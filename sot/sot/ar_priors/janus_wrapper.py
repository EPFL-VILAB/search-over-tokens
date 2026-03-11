"""Janus / Janus Pro AR prior wrapper.

Key differences from FlexTok:
  - Janus uses a VQ codebook (not flow matching) → decode is deterministic
  - Janus REQUIRES complete token sequences (576 tokens) for decoding;
    partial sequences are zero-padded automatically in decode_tokens()
  - Generation uses manual CFG (paired cond/uncond forward passes)
  - No external config/checkpoint — loads directly from HuggingFace Hub
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Union, Tuple, Optional
from PIL import Image

from sot.ar_priors.base import BaseARPrior, ARPriorFactory

logger = logging.getLogger("sot.janus")

# Import Janus model
try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from transformers import AutoModelForCausalLM
    JANUS_AVAILABLE = True
except ImportError:
    JANUS_AVAILABLE = False
    logger.warning(
        "Janus not available. Install it first: "
        "pip install -e path/to/Janus"
    )

__all__ = ["JanusARPrior", "JanusProARPrior"]


@ARPriorFactory.register("janus")
class JanusARPrior(BaseARPrior):
    """Wrapper for DeepSeek Janus AR model (1.3B).

    Generation logic:
      - CFG via paired cond/uncond forward passes
      - Token-by-token autoregressive generation with KV caching
      - VQ codebook decoding (decode_code)

    Config parameters:
        - model_version: HuggingFace model name (default: deepseek-ai/Janus-1.3B)
        - cfg_factor: Classifier-free guidance scale (default: 5.0)
        - temperature: Sampling temperature (default: 1.0)
    """

    # Model constants
    VOCAB_SIZE = 16384
    TOKEN_NUMBER = 576
    IMG_SIZE = 384
    PATCH_SIZE = 16

    def __init__(self, config: Dict, device: str = "cuda"):
        if not JANUS_AVAILABLE:
            raise ImportError("Janus package not found. Install it first.")

        super().__init__(config, device)

        model_version = config.get("model_version", "deepseek-ai/Janus-1.3B")
        self.cfg_factor = config.get("cfg_factor", 5.0)
        self.temperature = config.get("temperature", 1.0)

        logger.info(f"[Device {device}] Loading Janus model: {model_version}")

        # Load tokenizer / chat processor
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            model_version
        )
        self.text_tokenizer = self.vl_chat_processor.tokenizer

        # Load model (handle distributed if available)
        # The HF config has _attn_implementation="flash_attention_2" baked in
        # the language_config sub-model. Passing attn_implementation to
        # from_pretrained only overrides the top level; LlamaForCausalLM is
        # constructed from language_config directly inside Janus __init__.
        # We patch language_config after loading the config to avoid requiring
        # flash_attn.
        # Check for the *real* flash_attn (not the SDPA shim from infinity_wrapper).
        # The shim sets __version__ = "0.0.0".
        try:
            import flash_attn
            _attn = "flash_attention_2" if getattr(flash_attn, "__version__", "0.0.0") != "0.0.0" else "sdpa"
        except ImportError:
            _attn = "sdpa"

        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_version, trust_remote_code=True)
        if hasattr(hf_config, "language_config"):
            hf_config.language_config._attn_implementation = _attn
            hf_config.language_config._attn_implementation_internal = _attn

        _load_kwargs = dict(
            config=hf_config,
            trust_remote_code=True,
            use_safetensors=True,
        )
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            if rank == 0:
                self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_version, **_load_kwargs)
            torch.distributed.barrier()
            if rank != 0:
                self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_version, **_load_kwargs)
        else:
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_version, **_load_kwargs)

        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).to(device)

    # ------------------------------------------------------------------
    # Prompt formatting (Janus vs Pro differ here)
    # ------------------------------------------------------------------

    def _get_prompt(self, text: str) -> str:
        """Format text prompt for Janus model."""
        conversation = [
            {"role": "User", "content": text},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        return sft_format + self.vl_chat_processor.image_start_tag

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    def generate_next_tokens(
        self,
        prompt: Union[str, List[str]],
        current_tokens: torch.Tensor,
        num_new_tokens: int = 1,
        num_samples: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate next tokens with Janus AR model + CFG.

        Uses paired cond/uncond forward passes with KV caching.
        Diversity comes from multinomial sampling (or greedy with top_k=1).

        Args:
            prompt: Text prompt
            current_tokens: Current token sequence [batch, seq_len]
            num_new_tokens: Number of new tokens to generate
            num_samples: Number of diverse samples per beam element
            **kwargs:
                - seed: RNG seed for reproducibility
                - top_k: 1 for greedy, None for multinomial (default: None)

        Returns:
            (tokens, None): tokens shape [batch * num_samples, seq_len + num_new_tokens]
        """
        batch_size = current_tokens.size(0)
        current_len = current_tokens.size(1)
        parallel_size = batch_size * num_samples

        # Seed for reproducibility

        greedy = kwargs.get("top_k", None) == 1

        # Replicate beams for num_samples
        if num_samples > 1:
            expanded_tokens = current_tokens.repeat_interleave(num_samples, dim=0)
        else:
            expanded_tokens = current_tokens

        # Build prompt embeddings (paired cond/uncond interleaved)
        prompt_str = prompt if isinstance(prompt, str) else prompt[0]
        prompt_text = self._get_prompt(prompt_str)
        input_ids = self.text_tokenizer.encode(prompt_text, return_tensors="pt").to(
            self.device
        )
        prompt_len = input_ids.size(1)

        # [parallel_size * 2, prompt_len]: even=cond, odd=uncond (text padded)
        tokens_paired = input_ids.repeat(parallel_size * 2, 1)
        for i in range(1, parallel_size * 2, 2):
            tokens_paired[i, 1 : prompt_len - 1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens_paired)

        # Add image embeddings for existing tokens
        if current_len > 0:
            # Pair each expanded beam for cond/uncond: [parallel_size, len] → [parallel_size*2, len]
            paired_img_tokens = (
                expanded_tokens.unsqueeze(1)
                .repeat(1, 2, 1)
                .reshape(parallel_size * 2, -1)
            )
            img_embeds = self.vl_gpt.prepare_gen_img_embeds(paired_img_tokens)
            inputs_embeds = torch.cat([inputs_embeds, img_embeds], dim=1)

        # Token-by-token generation with KV caching
        generated_tokens = []
        token_log_probs = []  # per-step log-probs for the sampled token
        outputs = None

        for step in range(num_new_tokens):
            with torch.no_grad():
                outputs = self.vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if step > 0 else None,
                )
                hidden_states = outputs.last_hidden_state
                logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])

                # CFG: combine cond/uncond logits
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                combined = logit_uncond + self.cfg_factor * (logit_cond - logit_uncond)

            log_probs = torch.log_softmax(combined / self.temperature, dim=-1)
            probs = log_probs.exp()

            if greedy:
                next_token = probs.argmax(dim=-1)
            else:
                next_token = torch.multinomial(probs, num_samples=1, generator=self.rng).squeeze(-1)

            # Gather log-probability of the sampled token: [parallel_size]
            step_log_prob = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
            token_log_probs.append(step_log_prob)

            generated_tokens.append(next_token)

            # Prepare image embedding for next step (paired cond/uncond)
            paired_next = next_token.unsqueeze(1).repeat(1, 2).view(-1)
            img_embeds = self.vl_gpt.prepare_gen_img_embeds(paired_next)
            inputs_embeds = img_embeds.unsqueeze(1)

        # Assemble result: [parallel_size, current_len + num_new_tokens]
        new_tokens = torch.stack(generated_tokens, dim=-1)  # [parallel_size, num_new_tokens]
        result = torch.cat([expanded_tokens, new_tokens], dim=-1)

        # Sum log-probs over all generated tokens → [parallel_size]
        log_prob_sum = torch.stack(token_log_probs, dim=-1).sum(dim=-1)

        return result, log_prob_sum

    # ------------------------------------------------------------------
    # Token decoding
    # ------------------------------------------------------------------

    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> List:
        """Decode tokens to images using Janus VQ codebook decoder.

        Janus requires exactly TOKEN_NUMBER (576) tokens. Partial sequences
        are automatically zero-padded.

        For lookahead search, the search algorithm AR-completes the
        tokens before calling this method, so padding is minimal or none.

        Args:
            tokens: Token sequences [batch, seq_len] (may be < 576)
            **kwargs: Unused (kept for interface compatibility)

        Returns:
            List of PIL Images
        """
        batch_size = tokens.size(0)
        current_len = tokens.size(1)

        # Zero-pad partial sequences to TOKEN_NUMBER
        if current_len < self.TOKEN_NUMBER:
            padding = torch.zeros(
                (batch_size, self.TOKEN_NUMBER - current_len),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = torch.cat([tokens, padding], dim=1)

        # Decode VQ tokens → images
        with torch.no_grad():
            dec = self.vl_gpt.gen_vision_model.decode_code(
                tokens.int(),
                shape=[
                    batch_size,
                    8,
                    self.IMG_SIZE // self.PATCH_SIZE,
                    self.IMG_SIZE // self.PATCH_SIZE,
                ],
            )
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        # Normalize [−1, 1] → [0, 255] and convert to PIL
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        images = [Image.fromarray(dec[i]) for i in range(batch_size)]

        return images

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def get_max_tokens(self) -> int:
        return self.TOKEN_NUMBER


@ARPriorFactory.register("janus_pro")
class JanusProARPrior(JanusARPrior):
    """Janus Pro (7B) — same architecture, different model and prompt format.

    Config parameters:
        - model_version: HuggingFace model name (default: deepseek-ai/Janus-Pro-7B)
        - cfg_factor: Classifier-free guidance scale (default: 5.0)
        - temperature: Sampling temperature (default: 1.0)
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        # Default to Pro model version
        config = dict(config)  # don't mutate caller's dict
        config.setdefault("model_version", "deepseek-ai/Janus-Pro-7B")
        super().__init__(config, device)

    def _get_prompt(self, text: str) -> str:
        """Format text prompt for Janus Pro (different role tags)."""
        conversation = [
            {"role": "<|User|>", "content": text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        return sft_format + self.vl_chat_processor.image_start_tag
