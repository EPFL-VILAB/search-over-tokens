"""Infinity AR prior wrapper — integrates the Infinity multi-scale visual AR model.

Key characteristics of Infinity (vs. FlexTok/Janus):
  - Multi-scale visual autoregressive model (coarse-to-fine, not flat token sequence)
  - Uses BSQ-VAE with bit labels (binary codes packed into integers)
  - Generation proceeds scale-by-scale; each scale has (h × w) spatial positions
  - Requires T5 text encoder + Infinity transformer + BSQ-VAE

Token representation:
  Each spatial position at each scale produces a single integer code (packed from
  ``codebook_dim`` bits).  A flat token sequence concatenates codes across scales:
      [scale_0 (h0*w0 codes), scale_1 (h1*w1 codes), ...]
  Search algorithms (beam search, etc.) work with these flat sequences; the wrapper
  translates to/from Infinity's native scale-based representation internally.

Everything is auto-downloaded when paths are set to ``"auto"`` (the default):
  - Source code: cloned from GitHub (``FoundationVision/Infinity``)
  - Checkpoints: downloaded from HuggingFace (``FoundationVision/Infinity``)
  - T5 encoder: downloaded from HuggingFace (``google/flan-t5-xl``)
"""

import os
import torch
import numpy as np
import sys
import bisect
import logging
from typing import Dict, List, Union, Tuple, Optional
from PIL import Image
import torch.nn.functional as F

from soto.ar_priors.base import BaseARPrior, ARPriorFactory

logger = logging.getLogger("soto.infinity")

__all__ = ["InfinityARPrior"]

# ---------------------------------------------------------------------------
# flash_attn shim: Infinity's source imports flash_attn at the top level, but
# the actual attention code has a slow_attn (SDPA) fallback controlled by
# `self.using_flash`.  If flash_attn is not installed we register a shim
# module so the import succeeds, and set using_flash=False at runtime.
# ---------------------------------------------------------------------------
def _install_flash_attn_shim():
    """Register a fake ``flash_attn`` package in sys.modules if the real one
    is not installed.  This lets Infinity's ``from flash_attn import ...``
    succeed; the imported functions are SDPA-based fallbacks."""
    try:
        import flash_attn  # noqa: F401
        return  # real package available — nothing to do
    except ImportError:
        pass

    import types
    from torch.nn.functional import scaled_dot_product_attention as _sdpa

    def _flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, **kwargs):
        # flash_attn layout: [B, L, H, c] → SDPA layout: [B, H, L, c]
        out = _sdpa(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=softmax_scale, dropout_p=dropout_p,
        )
        return out.transpose(1, 2)

    def _flash_attn_varlen_kvpacked_func(
        q, kv, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=0.0, softmax_scale=None, **kwargs,
    ):
        B = cu_seqlens_q.shape[0] - 1
        k, v = kv.unbind(dim=1)
        H, c = q.shape[-2], q.shape[-1]
        q = q.reshape(B, max_seqlen_q, H, c).transpose(1, 2)
        k = k.reshape(B, max_seqlen_k, H, c).transpose(1, 2)
        v = v.reshape(B, max_seqlen_k, H, c).transpose(1, 2)
        out = _sdpa(q, k, v, scale=softmax_scale, dropout_p=dropout_p)
        return out.transpose(1, 2).reshape(-1, H, c)

    # Build shim modules with proper __spec__ so importlib.util.find_spec()
    # doesn't crash (used by diffusers, etc.)
    import importlib
    def _make_shim(name):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.__path__ = []  # mark as package for sub-module imports
        return mod

    fa = _make_shim("flash_attn")
    fa.flash_attn_func = _flash_attn_func
    fa.flash_attn_varlen_kvpacked_func = _flash_attn_varlen_kvpacked_func
    fa.__version__ = "0.0.0"  # some libraries check the version

    fa_ops = _make_shim("flash_attn.ops")
    fa_ops_ln = _make_shim("flash_attn.ops.layer_norm")
    fa_ops_rn = _make_shim("flash_attn.ops.rms_norm")
    fa_ops_fd = _make_shim("flash_attn.ops.fused_dense")

    sys.modules["flash_attn"] = fa
    sys.modules["flash_attn.ops"] = fa_ops
    sys.modules["flash_attn.ops.layer_norm"] = fa_ops_ln
    sys.modules["flash_attn.ops.rms_norm"] = fa_ops_rn
    sys.modules["flash_attn.ops.fused_dense"] = fa_ops_fd

    logger.info("flash_attn not installed — using PyTorch SDPA fallback for Infinity")

_install_flash_attn_shim()

# ---------------------------------------------------------------------------
# Auto-download defaults
# ---------------------------------------------------------------------------
_INFINITY_GIT_URL = "https://github.com/FoundationVision/Infinity.git"
_HF_REPO_ID = "FoundationVision/Infinity"

# Map model_type → HF filename (for checkpoint auto-download)
_HF_MODEL_FILES = {
    "infinity_2b": "infinity_2b_reg.pth",
}

# Map vae_type (codebook dim) → HF filename
_HF_VAE_FILES = {
    32: "infinity_vae_d32reg.pth",
}

# T5 text encoder (auto-downloaded by HuggingFace transformers)
_DEFAULT_T5_MODEL = "google/flan-t5-xl"

# Default cache location for the auto-cloned Infinity repo
_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "infinity")


def _ensure_infinity_repo(cache_dir: Optional[str] = None) -> str:
    """Ensure the FoundationVision Infinity source repo is available.

    Checks if the *correct* ``infinity`` package is importable (the one
    with ``infinity.models.infinity``).  If not, clones the GitHub repo
    into ``cache_dir`` (default: ``~/.cache/infinity/Infinity``).

    Returns:
        Path to the repo root (to be added to ``sys.path``).
    """
    # 1. Already importable (check for the specific sub-module to avoid
    #    collision with unrelated ``infinity`` pip packages)
    try:
        from infinity.models.infinity import Infinity  # noqa: F401
        import infinity as _inf
        return os.path.dirname(os.path.dirname(_inf.__file__))
    except (ImportError, ModuleNotFoundError):
        pass

    # 2. Clone to cache
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    repo_dir = os.path.join(cache_dir, "Infinity")

    if os.path.isdir(os.path.join(repo_dir, "infinity", "models")):
        logger.info(f"  Infinity repo already cached at {repo_dir}")
        return repo_dir

    logger.info(f"  Cloning Infinity repo from {_INFINITY_GIT_URL} → {repo_dir} ...")
    os.makedirs(cache_dir, exist_ok=True)

    import subprocess
    subprocess.check_call(
        ["git", "clone", "--depth", "1", _INFINITY_GIT_URL, repo_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info(f"  Cloned Infinity repo to {repo_dir}")
    return repo_dir


def _resolve_checkpoint(
    path: Optional[str], hf_filename: str, cache_dir: Optional[str] = None,
) -> str:
    """Return a local checkpoint path, downloading from HuggingFace if needed.

    Args:
        path: User-specified path.  If ``None``, empty, or ``"auto"``,
            the file is downloaded from ``FoundationVision/Infinity``.
        hf_filename: Filename within the HF repo.
        cache_dir: Optional cache directory for downloads.

    Returns:
        Resolved local file path.
    """
    if path and path != "auto" and os.path.exists(path):
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-downloading Infinity checkpoints. "
            "Install with: pip install huggingface_hub"
        )

    logger.info(f"  Auto-downloading {hf_filename} from {_HF_REPO_ID} ...")
    return hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=hf_filename,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Sampling utility (ported from reference infinity_wrapper.py)
# ---------------------------------------------------------------------------

def _sample_with_top_k_top_p(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng=None,
    num_samples: int = 1,
    replacement: bool = True,
) -> torch.Tensor:
    """Sample from logits with optional top-k / top-p filtering.

    Args:
        logits_BlV: Logits tensor ``[B, l, V]``.
        top_k: Top-k filtering threshold (0 = disabled).
        top_p: Nucleus (top-p) filtering threshold (0.0 = disabled).
        rng: Optional ``torch.Generator``.
        num_samples: Number of independent draws per position.
        replacement: Whether to sample with replacement.

    Returns:
        Sampled indices ``[B, l, num_samples]``.
    """
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(
                sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
            ),
            -torch.inf,
        )
    num_samples = abs(num_samples)
    return torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng,
    ).view(B, l, num_samples)


# ---------------------------------------------------------------------------
# Infinity AR Prior
# ---------------------------------------------------------------------------

@ARPriorFactory.register("infinity")
class InfinityARPrior(BaseARPrior):
    """Wrapper for the Infinity multi-scale visual AR model.

    Implements the SoTo ``BaseARPrior`` interface, enabling beam search,
    best-of-N, and lookahead search over Infinity's scale-based token space.

    Everything is auto-downloaded when paths are set to ``"auto"`` (default):
      - **Source code**: cloned from GitHub (``FoundationVision/Infinity``)
      - **Checkpoints**: downloaded from HuggingFace (``FoundationVision/Infinity``)
      - **T5 encoder**: downloaded from HuggingFace (``google/flan-t5-xl``)

    Config parameters:
        - model_path: Checkpoint path or ``"auto"`` (default)
        - vae_path: VAE checkpoint path or ``"auto"`` (default)
        - text_encoder_ckpt: T5 model name/path or ``"auto"`` (default)
        - cache_dir: Cache directory for downloads (default: ``~/.cache/infinity``)
        - model_type: ``"infinity_2b"`` or ``"infinity_8b"``
        - vae_type: Codebook bit-width (default: 32)
        - pn: Pixel-count string ``"0.06M"`` / ``"0.25M"`` / ``"1M"``
        - cfg_factor: Classifier-free guidance scale (default: 3.0)
        - cfg_insertion_layer: ``[0]`` = on logits, ``[-N]`` = N-th layer from end
    """

    # Model architecture configs (matches official run_infinity.py)
    _MODEL_CONFIGS = {
        "infinity_2b": dict(
            depth=32, embed_dim=2048, num_heads=2048 // 128,
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
        ),
        "infinity_8b": dict(
            depth=40, embed_dim=3584, num_heads=28,
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
        ),
    }

    def __init__(self, config: Dict, device: str = "cuda"):
        super().__init__(config, device)

        # ---- Ensure Infinity source code is available ----
        repo_path = _ensure_infinity_repo(config.get("cache_dir"))
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from infinity.models.infinity import Infinity
            from infinity.utils.dynamic_resolution import (
                dynamic_resolution_h_w, h_div_w_templates,
            )
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as e:
            raise ImportError(f"Infinity not available: {e}.")

        self._dynamic_resolution_h_w = dynamic_resolution_h_w
        self._h_div_w_templates = h_div_w_templates

        # ---- Config ----
        self.model_type = config.get("model_type", "infinity_2b")
        self.vae_type = config.get("vae_type", 32)
        self.pn = str(config.get("pn", "1M"))
        self.h_div_w_template = config.get("h_div_w_template", 1.0)
        self._vocab_size = 2 ** self.vae_type
        self.cfg_factor = config.get("cfg_factor", 3.0)
        self.temperature = config.get("temperature", 1.0)
        self.cfg_insertion_layer = list(config.get("cfg_insertion_layer", [0]))

        # ---- Resolve checkpoint paths (auto-download if needed) ----
        cache_dir = config.get("cache_dir")
        model_path = _resolve_checkpoint(
            config.get("model_path", "auto"),
            _HF_MODEL_FILES.get(self.model_type, f"{self.model_type}.pth"),
            cache_dir,
        )
        vae_path = _resolve_checkpoint(
            config.get("vae_path", "auto"),
            _HF_VAE_FILES.get(self.vae_type, f"infinity_vae_d{self.vae_type}reg.pth"),
            cache_dir,
        )
        text_encoder_ckpt = config.get("text_encoder_ckpt", _DEFAULT_T5_MODEL)
        if not text_encoder_ckpt or text_encoder_ckpt == "auto":
            text_encoder_ckpt = _DEFAULT_T5_MODEL

        # ---- Load models ----
        logger.info(f"[Device {device}] Loading Infinity ({self.model_type})")

        self.text_tokenizer, self.text_encoder = self._load_text_encoder(
            text_encoder_ckpt, AutoTokenizer, T5EncoderModel
        )
        self.vae = self._load_vae(vae_path)
        self.infinity = self._load_infinity(model_path, Infinity)

        # ---- Scale schedule & boundaries ----
        self.scale_schedule = self._get_scale_schedule()
        self.cfg_list = [self.cfg_factor] * len(self.scale_schedule)

        self._scale_sizes: List[int] = [h * w for _, h, w in self.scale_schedule]
        self._scale_cumulative: List[int] = []
        total = 0
        for s in self._scale_sizes:
            total += s
            self._scale_cumulative.append(total)

        logger.info(
            f"  Scale schedule: {len(self.scale_schedule)} scales, "
            f"total tokens: {self._scale_cumulative[-1]}"
        )
        self._current_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_text_encoder(self, ckpt, AutoTokenizer, T5EncoderModel):
        """Load T5 text encoder and tokenizer (auto-downloads from HF)."""
        logger.info(f"  Loading text encoder: {ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt, revision=None, legacy=True)
        tokenizer.model_max_length = 512
        encoder = T5EncoderModel.from_pretrained(ckpt, torch_dtype=torch.float16)
        encoder.to(self.device).eval().requires_grad_(False)
        return tokenizer, encoder

    def _load_vae(self, vae_path):
        """Load BSQ-VAE model."""
        logger.info(f"  Loading VAE: {vae_path}")
        from infinity.models.bsq_vae.vae import vae_model
        return vae_model(
            vae_path,
            schedule_mode="dynamic",
            codebook_dim=self.vae_type,
            codebook_size=2 ** self.vae_type,
            patch_size=16,
            encoder_ch_mult=[1, 2, 4, 4, 4],
            decoder_ch_mult=[1, 2, 4, 4, 4],
            test_mode=True,
        ).to(self.device)

    def _load_infinity(self, model_path, Infinity):
        """Load Infinity transformer model."""
        logger.info(f"  Loading Infinity transformer: {model_path}")

        if self.model_type not in self._MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Available: {list(self._MODEL_CONFIGS.keys())}"
            )

        with torch.cuda.amp.autocast(
            enabled=True, dtype=torch.bfloat16, cache_enabled=True
        ), torch.no_grad():
            infinity = Infinity(
                vae_local=self.vae,
                text_channels=2048, text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None,
                checkpointing="full-block",
                customized_flash_attn=False, fused_norm=True,
                pad_to_multiplier=128, use_flex_attn=False,
                add_lvl_embeding_only_first_block=1,
                use_bit_label=1, rope2d_each_sa_layer=1,
                rope2d_normalized_by_hw=2, pn=self.pn,
                apply_spatial_patchify=0, inference_mode=True,
                train_h_div_w_list=[1.0],
                **self._MODEL_CONFIGS[self.model_type],
            ).to(device=self.device)

        infinity.eval()
        infinity.requires_grad_(False)
        infinity.to(self.device)
        torch.cuda.empty_cache()

        state_dict = torch.load(model_path, map_location=self.device)
        infinity.load_state_dict(state_dict)
        infinity.rng = torch.Generator(device=self.device)
        return infinity

    def _get_scale_schedule(self):
        """Get scale schedule for configured aspect ratio and pixel count."""
        templates = self._h_div_w_templates
        closest = templates[np.argmin(np.abs(templates - self.h_div_w_template))]
        scales = self._dynamic_resolution_h_w[closest][self.pn]["scales"]
        return [(1, h, w) for (_, h, w) in scales]

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str):
        """Encode text prompt using T5 encoder.

        Returns:
            Tuple of ``(kv_compact, lens, cu_seqlens_k, Ltext)``.
        """
        captions = [prompt]
        tokens = self.text_tokenizer(
            text=captions,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device, non_blocking=True)
        mask = tokens.attention_mask.to(self.device, non_blocking=True)

        self.text_encoder = self.text_encoder.to(torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            text_features = self.text_encoder(
                input_ids=input_ids, attention_mask=mask
            )["last_hidden_state"].float()

        lens = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(
            mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0)
        )
        Ltext = max(lens)

        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)

        return (kv_compact, lens, cu_seqlens_k, Ltext)

    # ------------------------------------------------------------------
    # Scale ↔ token mapping
    # ------------------------------------------------------------------

    def _tokens_to_scale_idx(self, num_tokens: int) -> int:
        """Convert flat token count → number of *completed* scales.

        Examples (assuming cumulative = [1, 5, 14, 30, ...]):
          0 tokens → 0  (no scale completed)
          1 token  → 1  (scale 0 completed)
          5 tokens → 2  (scales 0-1 completed)
          3 tokens → 1  (scale 0 done, partially into scale 1)
        """
        if num_tokens <= 0:
            return 0
        idx = bisect.bisect_left(self._scale_cumulative, num_tokens)
        if idx < len(self._scale_cumulative) and self._scale_cumulative[idx] == num_tokens:
            return idx + 1
        return idx

    def get_scale_boundaries(self, max_steps: int) -> List[int]:
        """Return cumulative token counts per scale for beam search scheduling.

        This enables beam search to step through scales one at a time,
        matching Infinity's native generation granularity.

        Args:
            max_steps: Maximum number of steps (scales) to return.

        Returns:
            List of cumulative token counts, one entry per scale.
        """
        return self._scale_cumulative[:max_steps]

    # ------------------------------------------------------------------
    # Bit packing / unpacking
    # ------------------------------------------------------------------

    def _unpack_codes_to_idx_Bld_list(
        self,
        codes: torch.Tensor,
        up_to_scale: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Unpack integer codes → per-scale bit-label tensors.

        Args:
            codes: Packed integer codes, shape ``[B, L]``.
            up_to_scale: Number of leading scales to unpack (``None`` = all).

        Returns:
            List of tensors, each ``[B, 1, h, w, codebook_dim]``.
        """
        codebook_dim = self.infinity.codebook_dim

        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        Bprime, L = codes.shape
        device = codes.device

        # Unpack bits: [B, L] → [B, L, d]
        bit_positions = torch.arange(codebook_dim, device=device)
        bits = (codes.long().unsqueeze(-1) >> bit_positions) & 1

        if up_to_scale is None:
            up_to_scale = len(self.scale_schedule)

        out: List[torch.Tensor] = []
        ptr = 0
        for si in range(min(up_to_scale, len(self.scale_schedule))):
            _, h, w = self.scale_schedule[si]
            num_tokens = h * w
            scale_bits = bits[:, ptr : ptr + num_tokens, :]
            ptr += num_tokens
            if scale_bits.numel() == 0:
                out.append(
                    torch.zeros(
                        Bprime, 1, h, w, codebook_dim,
                        dtype=bits.dtype, device=device,
                    )
                )
                continue
            scale_bits = scale_bits.reshape(Bprime, h, w, codebook_dim)
            out.append(scale_bits.unsqueeze(1))  # [B, 1, h, w, d]

        return out

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def _enable_kv_caching(self):
        """Enable KV caching for all attention modules."""
        if hasattr(self.infinity, "block_chunks"):
            for bc in self.infinity.block_chunks:
                for m in bc.module:
                    (m.sa if hasattr(m, "sa") else m.attn).kv_caching(True)
        else:
            for b in self.infinity.unregistered_blocks:
                (b.sa if hasattr(b, "sa") else b.attn).kv_caching(True)

    def _disable_kv_caching(self):
        """Disable KV caching for all attention modules."""
        if hasattr(self.infinity, "block_chunks"):
            for bc in self.infinity.block_chunks:
                for m in bc.module:
                    (m.sa if hasattr(m, "sa") else m.attn).kv_caching(False)
        else:
            for b in self.infinity.unregistered_blocks:
                (b.sa if hasattr(b, "sa") else b.attn).kv_caching(False)

    # ------------------------------------------------------------------
    # Token generation (core interface)
    # ------------------------------------------------------------------

    def generate_next_tokens(
        self,
        prompt: Union[str, List[str]],
        current_tokens: torch.Tensor,
        num_new_tokens: int = 1,
        num_samples: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate next tokens (scale-by-scale).

        Translates between the SoTo flat-token interface and Infinity's
        scale-based generation.  The current token count determines which
        scales are already completed; new scales are generated until the
        total token count reaches ``current_len + num_new_tokens``
        (rounded up to the next scale boundary).

        Args:
            prompt: Text prompt.
            current_tokens: ``[batch, seq_len]`` packed integer codes.
            num_new_tokens: Target number of new tokens (aligned to scale
                boundaries internally).
            num_samples: Branching factor — candidates per beam element.
            **kwargs:
                - top_k (int): 1 for greedy; defaults to ``num_samples``.
                - replacement (bool): Sample with replacement (default True).

        Returns:
            ``(extended_tokens, None)``:
                extended_tokens — ``[batch * num_samples, new_seq_len]``.
        """
        # Cache prompt for decode_tokens AR completion
        prompt_str = prompt if isinstance(prompt, str) else prompt[0]
        self._current_prompt = prompt_str

        B = current_tokens.size(0) if current_tokens.dim() == 2 else 1
        current_len = current_tokens.size(1) if current_tokens.dim() == 2 else 0

        # Determine current and target scale indices
        current_scale_idx = self._tokens_to_scale_idx(current_len)
        target_len = current_len + num_new_tokens
        target_scale_idx = self._tokens_to_scale_idx(target_len)

        # Always generate at least one more scale
        if target_scale_idx <= current_scale_idx:
            target_scale_idx = current_scale_idx + 1
        target_scale_idx = min(target_scale_idx, len(self.scale_schedule))

        # Sampling parameters
        top_k = kwargs.get("top_k", num_samples)
        replacement = kwargs.get("replacement", True)


        # Normalise empty-input case
        if current_len == 0:
            selected_tokens = torch.zeros(
                (1, 0), dtype=torch.long, device=self.device
            )
            B = 1
        else:
            selected_tokens = current_tokens

        # Always process one beam at a time to bound peak GPU memory.
        #
        # _generate_scales replays ALL scales from 0 to target_scale_idx on every
        # call (Infinity needs the full conditioning chain), so the transformer batch
        # size is  2 * B * top_k  (CFG doubles it).  With beam_width=5 and
        # candidates_per_beam=2 that's a batch of 20 at scale 13 — easy OOM.
        #
        # Looping over beams caps peak batch size at  2 * top_k  regardless of how
        # many beams are active.  It also gives each beam its own independent
        # sampling chain, which is required for diversity when top_k > 1 and
        # multiple new scales are generated in one call (otherwise all candidates
        # inside a call share the same greedy visual conditioning).

        log_probs_all: Optional[torch.Tensor] = None

        with torch.no_grad():
            if B == 1:
                # Single beam: call directly, no loop overhead
                result, log_probs_all = self._generate_scales(
                    prompt=prompt_str,
                    selected_tokens=selected_tokens,
                    current_scale_idx=current_scale_idx,
                    target_scale_idx=target_scale_idx,
                    top_k=top_k,
                    replacement=replacement,
                    rng=self.rng,
                )
                if not isinstance(result, torch.Tensor):
                    result = torch.tensor(result, dtype=torch.long, device=self.device)
            else:
                # Multiple beams: process each beam independently to limit batch size
                chunks = []
                lp_chunks = []
                for i in range(B):
                    chunk, chunk_lp = self._generate_scales(
                        prompt=prompt_str,
                        selected_tokens=selected_tokens[i : i + 1],
                        current_scale_idx=current_scale_idx,
                        target_scale_idx=target_scale_idx,
                        top_k=top_k,
                        replacement=replacement,
                        rng=self.rng,
                    )
                    if not isinstance(chunk, torch.Tensor):
                        chunk = torch.tensor(chunk, dtype=torch.long, device=self.device)
                    chunks.append(chunk)
                    if chunk_lp is not None:
                        lp_chunks.append(chunk_lp)
                result = torch.cat(chunks, dim=0)
                if lp_chunks:
                    log_probs_all = torch.cat(lp_chunks, dim=0)

        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=torch.long, device=self.device)

        return result, log_probs_all

    # ------------------------------------------------------------------
    # Helpers for scale-by-scale generation
    # ------------------------------------------------------------------

    def _embed_codes_for_scale(
        self, summed_codes: torch.Tensor, si: int, bs: int, B: int,
    ) -> torch.Tensor:
        """Interpolate accumulated codes to scale ``si`` and embed."""
        x = F.interpolate(
            summed_codes, size=self.scale_schedule[si],
            mode=self.vae.quantizer.z_interplote_up,
        )
        x = x.squeeze(-3).reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = self.infinity.word_embed(self.infinity.norm0_ve(x))
        return x.repeat(bs // B, 1, 1)

    def _accumulate_codes(
        self, idx_Bld: torch.Tensor, summed_codes: Union[int, torch.Tensor],
    ) -> Union[int, torch.Tensor]:
        """Add bit-label codes to the running sum at full resolution."""
        codes = self.vae.quantizer.lfq.indices_to_codes(
            idx_Bld, label_type="bit_label"
        )
        return summed_codes + F.interpolate(
            codes, size=self.scale_schedule[-1],
            mode=self.vae.quantizer.z_interplote_up,
        )

    def _forward_blocks(
        self, last_stage: torch.Tensor, si: int,
        cond_BD_or_gss: torch.Tensor, ca_kv: tuple,
        use_cfg: bool, cfg_scale: float, B: int, cfg_layers: set,
    ) -> torch.Tensor:
        """Run all transformer blocks for one scale with optional CFG."""
        if hasattr(self.infinity, "block_chunks"):
            block_groups = [
                (idx, list(bc.module))
                for idx, bc in enumerate(self.infinity.block_chunks)
            ]
        else:
            block_groups = [
                (i, [b])
                for i, b in enumerate(self.infinity.unregistered_blocks)
            ]

        layer_idx = 0
        for group_idx, modules in block_groups:
            if group_idx == 0 or not self.infinity.add_lvl_embeding_only_first_block:
                last_stage = self.infinity.add_lvl_embeding(
                    last_stage, si, self.scale_schedule, need_to_pad=0
                )
            for m in modules:
                last_stage = m(
                    x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv,
                    attn_bias_or_two_vector=None, attn_fn=None,
                    scale_schedule=self.scale_schedule,
                    rope2d_freqs_grid=self.infinity.rope2d_freqs_grid,
                    scale_ind=si,
                )
                if use_cfg and layer_idx in cfg_layers:
                    last_stage = cfg_scale * last_stage[:B] + (1 - cfg_scale) * last_stage[B:]
                    last_stage = torch.cat((last_stage, last_stage), 0)
                layer_idx += 1
        return last_stage

    def _pack_bits_to_codes(
        self, idx_Bld: torch.Tensor, tmp_bs: int, tmp_seq_len: int, top_k: int,
    ) -> torch.Tensor:
        """Pack sampled bit-label indices into integer codes."""
        bits = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1, top_k).permute(0, 3, 1, 2)
        codes_packed = (
            bits.long()
            * (1 << torch.arange(bits.size(-1), device=bits.device))
        ).sum(-1, keepdim=True)
        return codes_packed.reshape(tmp_bs * top_k, -1)

    # ------------------------------------------------------------------
    # Core scale-by-scale generation
    # ------------------------------------------------------------------

    def _generate_scales(
        self,
        prompt: str,
        selected_tokens: torch.Tensor,
        current_scale_idx: int,
        target_scale_idx: int,
        top_k: int = 1,
        replacement: bool = True,
        rng=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scale-by-scale generation with CFG.

        Processes all scales from 0 to ``target_scale_idx``:
          - Scales ``< current_scale_idx``: condition on provided tokens
          - Scales ``>= current_scale_idx``: generate via sampling

        Returns:
            Tuple of (tokens, log_probs):
              - tokens: ``[B * top_k, total_tokens_up_to_target]``
              - log_probs: ``[B * top_k]`` summed log-probs, or ``None``
        """
        B = selected_tokens.size(0) if selected_tokens.dim() == 2 else 1

        # --- Encode prompt & replicate for batch ---
        kv_compact, lens, cu_seqlens_k, Ltext = self._encode_prompt(prompt)

        prev_idx_Bld_list = (
            self._unpack_codes_to_idx_Bld_list(selected_tokens, current_scale_idx)
            if selected_tokens.numel() > 0 and selected_tokens.size(1) > 0
            else []
        )

        if B > 1:
            kv_compact = kv_compact.expand(B, -1, -1).reshape(-1, kv_compact.shape[-1])
            total_length = cu_seqlens_k[-1].item()
            new_cu = [0]
            for i in range(B):
                for j in range(1, len(cu_seqlens_k)):
                    new_cu.append(cu_seqlens_k[j].item() + i * total_length)
            cu_seqlens_k = torch.tensor(
                new_cu, device=cu_seqlens_k.device, dtype=cu_seqlens_k.dtype
            )

        # --- CFG setup ---
        cfg_scale = self.cfg_list[min(current_scale_idx, len(self.cfg_list) - 1)]
        use_cfg = cfg_scale != 1.0
        bs = 2 * B if use_cfg else B

        if use_cfg:
            kv_compact_un = kv_compact.clone()
            total = 0
            for le in lens:
                kv_compact_un[total : total + le] = self.infinity.cfg_uncond[:le]
                total += le
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat(
                (cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0
            )

        # --- Transformer initialisation ---
        kv_compact = self.infinity.text_norm(kv_compact)
        cond_BD = self.infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, Ltext))
        ca_kv = (self.infinity.text_proj_for_ca(kv_compact), cu_seqlens_k, Ltext)
        last_stage = cond_BD.unsqueeze(1) + self.infinity.pos_start.expand(bs, 1, -1)

        with torch.cuda.amp.autocast(enabled=False):
            cond_BD_or_gss = self.infinity.shared_ada_lin(cond_BD.float()).float().contiguous()

        self._enable_kv_caching()

        # CFG insertion layers (set for O(1) lookup)
        cfg_layers: set = set()
        add_cfg_on_logits = False
        n_layers = (
            sum(len(bc.module) for bc in self.infinity.block_chunks)
            if hasattr(self.infinity, "block_chunks")
            else len(self.infinity.unregistered_blocks)
        )
        for item in self.cfg_insertion_layer:
            if item == 0:
                add_cfg_on_logits = True
            elif item < 0:
                cfg_layers.add(n_layers + item)

        summed_codes: Union[int, torch.Tensor] = 0
        output: Optional[torch.Tensor] = None
        # Accumulate log-probs for each (batch * top_k) sample
        log_prob_accum: Optional[torch.Tensor] = None

        # --- Scale-by-scale loop ---
        for si in range(target_scale_idx):
            _, h, w = self.scale_schedule[si]

            # ---- Visual conditioning ----
            if si == 0:
                pass  # last_stage already set to sos
            else:
                if si <= current_scale_idx:
                    summed_codes = self._accumulate_codes(
                        prev_idx_Bld_list[si - 1], summed_codes
                    )
                last_stage = self._embed_codes_for_scale(summed_codes, si, bs, B)

            # ---- Transformer forward ----
            last_stage = self._forward_blocks(
                last_stage, si, cond_BD_or_gss, ca_kv,
                use_cfg, cfg_scale, B, cfg_layers,
            )

            # ---- Sample at new scales ----
            if si >= current_scale_idx:
                if use_cfg and add_cfg_on_logits:
                    logits_BlV = self.infinity.get_logits(last_stage, cond_BD)
                    logits_BlV = cfg_scale * logits_BlV[:B] + (1 - cfg_scale) * logits_BlV[B:]
                else:
                    logits_BlV = self.infinity.get_logits(last_stage[:B], cond_BD[:B])

                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]

                # Reshape to binary logits for bit-label sampling: [B, positions*bits, 2]
                bit_logits = logits_BlV.reshape(B, -1, 2).clone()

                idx_Bld = _sample_with_top_k_top_p(
                    bit_logits,
                    top_k=0, top_p=0.0, rng=rng,
                    num_samples=top_k, replacement=replacement,
                )

                # --- Compute log-probabilities for the sampled bits ---
                # bit_logits: [B, n_bits, 2]  idx_Bld: [B, n_bits, top_k]
                log_probs_bits = torch.log_softmax(bit_logits, dim=-1)  # [B, n_bits, 2]
                # Gather log-prob of sampled index for each (batch, bit, sample)
                # idx_Bld: [B, n_bits, top_k] → need log_probs_bits per sample
                # Expand log_probs_bits to [B, n_bits, top_k, 2] is not needed;
                # instead gather along dim=-1 for each sample independently.
                # log_probs_bits: [B, n_bits, 2], idx_Bld: [B, n_bits, top_k]
                # For each sample k, gather: log_probs_bits[b, j, idx_Bld[b, j, k]]
                # Reshape: [B, n_bits, 1, 2] → expand → [B, n_bits, top_k, 2]
                lp_expanded = log_probs_bits.unsqueeze(2).expand(-1, -1, top_k, -1)
                sampled_lp = lp_expanded.gather(-1, idx_Bld.unsqueeze(-1)).squeeze(-1)
                # sampled_lp: [B, n_bits, top_k]
                # Sum over all bit positions → [B, top_k]
                scale_lp = sampled_lp.sum(dim=1)  # [B, top_k]
                # Flatten to [B * top_k]
                scale_lp_flat = scale_lp.reshape(-1)

                if log_prob_accum is None:
                    log_prob_accum = scale_lp_flat
                else:
                    log_prob_accum = log_prob_accum + scale_lp_flat

                output_tensor = self._pack_bits_to_codes(idx_Bld, tmp_bs, tmp_seq_len, top_k)

                # Accumulate output
                if output is None and selected_tokens.size(1) > 0:
                    output = selected_tokens.repeat_interleave(top_k, dim=0)
                output = (
                    torch.cat([output, output_tensor], dim=1)
                    if output is not None else output_tensor
                )

                # Update summed_codes with greedy (first) sample
                idx_Bld_greedy = idx_Bld[:, :, :1].reshape(B, h, w, -1).unsqueeze(1)
                summed_codes = self._accumulate_codes(idx_Bld_greedy, summed_codes)

        self._disable_kv_caching()
        return output, log_prob_accum

    # ------------------------------------------------------------------
    # Token decoding
    # ------------------------------------------------------------------

    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> List:
        """Decode tokens to PIL images via VAE.

        No AR completion is done here — the search algorithm is responsible
        for extending partial sequences (e.g. ``LookaheadSearch`` greedy-
        completes candidates).  This method just VAE-decodes whatever
        scales are present in ``tokens``.
        """
        batch_size = tokens.size(0)
        n_scales = self._tokens_to_scale_idx(tokens.size(1))

        with torch.no_grad():
            idx_list = self._unpack_codes_to_idx_Bld_list(tokens, n_scales)
            summed_codes: Union[int, torch.Tensor] = 0
            for si in range(n_scales):
                summed_codes = self._accumulate_codes(idx_list[si], summed_codes)

            # Decode in chunks to manage GPU memory
            max_batch = 10
            if batch_size <= max_batch:
                img = self.vae.decode(summed_codes.squeeze(-3))
            else:
                chunks = []
                for i in range(0, batch_size, max_batch):
                    chunks.append(
                        self.vae.decode(summed_codes[i : i + max_batch].squeeze(-3))
                    )
                img = torch.cat(chunks, dim=0)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).clamp_(0, 255).to(torch.uint8)
        return [Image.fromarray(img[i].cpu().numpy()) for i in range(batch_size)]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_max_tokens(self) -> int:
        return self._scale_cumulative[-1] if self._scale_cumulative else 0

