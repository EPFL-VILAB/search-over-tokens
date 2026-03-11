# Extending the Framework

Each new component requires only three steps:
1. Subclass the base class
2. Implement the abstract methods
3. Register with the factory decorator

## Adding a New AR Model

```python
# sot/sot/ar_priors/my_model_wrapper.py
from sot.ar_priors.base import BaseARPrior, ARPriorFactory

@ARPriorFactory.register("my_model")
class MyModelARPrior(BaseARPrior):
    def __init__(self, config, device="cuda"):
        super().__init__(config, device)
        self.model = load_my_model(config["checkpoint_path"])

    def generate_next_tokens(self, prompt, current_tokens, num_new_tokens=1, num_samples=1, **kwargs):
        """
        Args:
            prompt: Text prompt
            current_tokens: [batch, seq_len] current token IDs
            num_new_tokens: How many new tokens to generate
            num_samples: Branching factor (candidates per beam)
        Returns:
            (tokens, log_probs):
              tokens: [batch * num_samples, seq_len + num_new_tokens]
              log_probs: [batch * num_samples] or None
        """
        ...
        return extended_tokens, log_probs

    def decode_tokens(self, tokens, **kwargs):
        """
        Args:
            tokens: [batch, seq_len] token IDs
        Returns:
            List of PIL Images
        """
        ...
        return pil_images

    def get_vocab_size(self):
        return 32000

    def get_max_tokens(self):
        return 256
```

Add the import in `sot/sot/ar_priors/__init__.py` and create `sot/configs/components/ar_priors/my_model.yaml`:

```yaml
ar_prior:
  name: my_model
  checkpoint_path: /path/to/checkpoint.pth
```

## Adding a New Search Algorithm

```python
# sot/sot/search_algorithms/mcts.py
from sot.search_algorithms.base import BaseSearchAlgorithm, SearchAlgorithmFactory, SearchResult

@SearchAlgorithmFactory.register("mcts")
class MCTSSearch(BaseSearchAlgorithm):
    def __init__(self, ar_prior, verifier, config):
        super().__init__(ar_prior, verifier, config)
        self.num_simulations = config.get("num_simulations", 100)

    def search(self, prompt, num_results=1, **kwargs):
        # Use self.ar_prior.generate_next_tokens(...)
        # Use self.ar_prior.decode_tokens(...)
        # Use self.verifier.score(images, prompts)
        ...
        return SearchResult(tokens=best_tokens, images=best_images, scores=best_scores)
```

## Adding a New Verifier

```python
# sot/sot/verifiers/my_reward.py
from sot.verifiers.base import BaseVerifier, VerifierFactory

@VerifierFactory.register("my_reward")
class MyRewardVerifier(BaseVerifier):
    def __init__(self, config, device="cuda"):
        super().__init__(config, device)
        self.model = load_reward_model()

    def _score(self, images, prompts, **kwargs):
        """
        Args:
            images: List of PIL Images
            prompts: List of text prompts (one per image), or None
        Returns:
            Scores tensor of shape [num_images]
        """
        ...
        return torch.tensor(scores)
```

New verifiers also work immediately in ensembles:

```yaml
verifier:
  name: ensemble
  aggregation: rank
  verifiers:
    - name: my_reward
      weight: 2.0
    - name: clip
      weight: 1.0
```
