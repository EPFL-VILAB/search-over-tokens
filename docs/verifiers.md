# Verifiers

Verifiers (reward models) score generated images and guide search. All implement the `BaseVerifier` interface and are registered by name.

## Available Verifiers

| Name | What It Scores | Config | Implementation |
|------|---------------|--------|----------------|
| `clip` | CLIP image-text cosine similarity | [`clip.yaml`](../soto/configs/components/verifiers/clip.yaml) | [`clip_verifier.py`](../soto/soto/verifiers/clip_verifier.py) |
| `image_reward` | Learned human-preference alignment | [`image_reward.yaml`](../soto/configs/components/verifiers/image_reward.yaml) | [`image_reward_verifier.py`](../soto/soto/verifiers/image_reward_verifier.py) |
| `aesthetic` | Aesthetic quality prediction | [`aesthetic.yaml`](../soto/configs/components/verifiers/aesthetic.yaml) | [`aesthetic_verifier.py`](../soto/soto/verifiers/aesthetic_verifier.py) |
| `pickscore` | Pick-a-Pic human preference | [`pickscore.yaml`](../soto/configs/components/verifiers/pickscore.yaml) | [`pickscore_verifier.py`](../soto/soto/verifiers/pickscore_verifier.py) |
| `hpsv2` | Human Preference Score v2 | [`hpsv2.yaml`](../soto/configs/components/verifiers/hpsv2.yaml) | [`hpsv2_verifier.py`](../soto/soto/verifiers/hpsv2_verifier.py) |
| `cyclereward` | Cycle-consistent reward | [`cyclereward.yaml`](../soto/configs/components/verifiers/cyclereward.yaml) | [`cyclereward_verifier.py`](../soto/soto/verifiers/cyclereward_verifier.py) |
| `likelihood` | AR log-probability of the token sequence | [`likelihood.yaml`](../soto/configs/components/verifiers/likelihood.yaml) | [`likelihood_verifier.py`](../soto/soto/verifiers/likelihood_verifier.py) |
| `grounded_sam` | Spatial compositionality (counting, colors, relations) | [`spatial.yaml`](../soto/configs/components/verifiers/spatial.yaml) | [`spatial_verifier.py`](../soto/soto/verifiers/spatial_verifier.py) |
| `dreamsim` | Perceptual similarity to reference image | [`dreamsim.yaml`](../soto/configs/components/verifiers/dreamsim.yaml) | [`dreamsim_verifier.py`](../soto/soto/verifiers/dreamsim_verifier.py) |
| `ensemble` | Combines any of the above | [`ensemble.yaml`](../soto/configs/components/verifiers/ensemble.yaml) | [`ensemble_verifier.py`](../soto/soto/verifiers/ensemble_verifier.py) |

All models auto-download from HuggingFace on first use.

## Ensemble Verifier

The ensemble combines multiple verifiers with two aggregation strategies:
- **`rank`** (default): Each verifier ranks candidates independently; final score = negative sum of ranks.
- **`weighted`**: Min-max normalize each verifier's scores, then compute weighted sum.

```yaml
# soto/configs/components/verifiers/ensemble.yaml
verifier:
  name: ensemble
  aggregation: rank
  verifiers:
    - name: clip
      weight: 1.0
    - name: image_reward
      weight: 1.0
    - name: aesthetic
      weight: 1.0
    - name: grounded_sam
      weight: 1.0
      part_by_part: true
      spatial_metric: "pse"
    - name: likelihood
      weight: 1.0
```

```bash
python soto/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ensemble
```
