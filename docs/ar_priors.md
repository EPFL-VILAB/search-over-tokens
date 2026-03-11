# AR Priors

`SoT` wraps multiple AR image generation models behind a unified `BaseARPrior` interface. Each model is registered by name and auto-loads its default YAML config.

## FlexTok AR Model Zoo

FlexTok AR is the primary AR model from our paper. It uses [FlexTok](https://github.com/apple/ml-flextok) tokenizers to represent images as 1D ordered sequences of 256 tokens, decoded via a flow-matching diffusion decoder.

For **standalone use** (image generation without the full SoT framework), see [`flextok_ar/README.md`](../flextok_ar/README.md).

| Config Name | Model | Params | HuggingFace | Token Type | Config | Wrapper |
|-------------|-------|--------|-------------|------------|--------|---------|
| `flextok_ar_113m` | FlexAR-113M | 113M | [`EPFL-VILAB/FlexAR-113M-T2I`](https://huggingface.co/EPFL-VILAB/FlexAR-113M-T2I) | 1D ordered (256 tokens) | [`flextok_ar_113m.yaml`](../sot/configs/components/ar_priors/flextok/flextok_ar_113m.yaml) | [`flextok_wrapper.py`](../sot/sot/ar_priors/flextok_wrapper.py) |
| `flextok_ar_382m` | FlexAR-382M | 382M | [`EPFL-VILAB/FlexAR-382M-T2I`](https://huggingface.co/EPFL-VILAB/FlexAR-382M-T2I) | 1D ordered (256 tokens) | [`flextok_ar_382m.yaml`](../sot/configs/components/ar_priors/flextok/flextok_ar_382m.yaml) | [`flextok_wrapper.py`](../sot/sot/ar_priors/flextok_wrapper.py) |
| `flextok_ar_1b` | FlexAR-1B | 1.15B | [`EPFL-VILAB/FlexAR-1B-T2I`](https://huggingface.co/EPFL-VILAB/FlexAR-1B-T2I) | 1D ordered (256 tokens) | [`flextok_ar_1b.yaml`](../sot/configs/components/ar_priors/flextok/flextok_ar_1b.yaml) | [`flextok_wrapper.py`](../sot/sot/ar_priors/flextok_wrapper.py) |
| `flextok_ar_3b` | FlexAR-3B | 3.06B | [`EPFL-VILAB/FlexAR-3B-T2I`](https://huggingface.co/EPFL-VILAB/FlexAR-3B-T2I) | 1D ordered (256 tokens) | [`flextok_ar_3b.yaml`](../sot/configs/components/ar_priors/flextok/flextok_ar_3b.yaml) | [`flextok_wrapper.py`](../sot/sot/ar_priors/flextok_wrapper.py) |
| `flextok_ar_3b` (uncond) | FlexAR-3B | 3.06B | [`EPFL-VILAB/FlexAR-3B-T2I`](https://huggingface.co/EPFL-VILAB/FlexAR-3B-T2I) | 1D ordered (256 tokens) | [`flextok_ar_3b_uncond.yaml`](../sot/configs/components/ar_priors/flextok/flextok_ar_3b_uncond.yaml) | [`flextok_wrapper.py`](../sot/sot/ar_priors/flextok_wrapper.py) |
| `gridtok_ar_3b` | GridAR-3B | 3.06B | — | 2D grid (256 tokens) | [`gridtok_ar_3b.yaml`](../sot/configs/components/ar_priors/gridtok/gridtok_ar_3b.yaml) | [`flextok_wrapper.py`](../sot/sot/ar_priors/flextok_wrapper.py) |
| `uniform` | — | — | — | 1D (random) | [`uniform.yaml`](../sot/configs/components/ar_priors/uniform.yaml) | [`uniform.py`](../sot/sot/ar_priors/uniform.py) |

> **Note:** Model weights are currently hosted under `ZhitongGao` on HuggingFace and will be moved to the [`EPFL-VILAB`](https://huggingface.co/EPFL-VILAB) organization upon public release.

## Extended AR Models

The SoT framework also supports third-party AR models via the same interface.

| Config Name | Model | HuggingFace | GitHub | Token Type | Config | Wrapper |
|-------------|-------|-------------|--------|------------|--------|---------|
| `janus` | Janus 1.3B | [`deepseek-ai/Janus-1.3B`](https://huggingface.co/deepseek-ai/Janus-1.3B) | [Janus](https://github.com/deepseek-ai/Janus) | 2D grid VQ (576 tokens) | [`janus.yaml`](../sot/configs/components/ar_priors/janus/janus.yaml) | [`janus_wrapper.py`](../sot/sot/ar_priors/janus_wrapper.py) |
| `janus_pro` | Janus Pro 7B | [`deepseek-ai/Janus-Pro-7B`](https://huggingface.co/deepseek-ai/Janus-Pro-7B) | [Janus](https://github.com/deepseek-ai/Janus) | 2D grid VQ (576 tokens) | [`janus_pro.yaml`](../sot/configs/components/ar_priors/janus/janus_pro.yaml) | [`janus_wrapper.py`](../sot/sot/ar_priors/janus_wrapper.py) |
| `infinity` | Infinity 2B | [`FoundationVision/Infinity`](https://huggingface.co/FoundationVision/Infinity) | [Infinity](https://github.com/FoundationVision/Infinity) | Multi-scale BSQ (13 scales) | [`infinity.yaml`](../sot/configs/components/ar_priors/infinity/infinity.yaml) | [`infinity_wrapper.py`](../sot/sot/ar_priors/infinity_wrapper.py) |

## Key Differences

- **FlexTok**: 1D ordered token sequence (256 tokens). Coarse-to-fine structure: first token captures global scene, each subsequent token progressively refines details. Decoded via flow-matching diffusion (stochastic, noise-controlled). **This is the primary model studied in the paper.**
- **GridTok**: Same AR architecture as FlexTok, but uses a 2D grid tokenizer (16×16 = 256 tokens, raster-scan order). Serves as the controlled 2D baseline for comparing against FlexTok's 1D ordering.
- **Janus / Janus Pro**: 2D grid token sequence (576 tokens), raster-scan order. Decoded via VQ codebook (deterministic).
- **Infinity**: Multi-scale token hierarchy (coarse-to-fine across 13 scales).
- **Uniform**: No AR model; direct sample from the FlexTok codebook (used for training-free generation experiments).

## Config Format

Each AR model has a YAML config under `sot/configs/components/ar_priors/`. Example (`flextok_ar_3b.yaml`):

```yaml
ar_prior:
  name: flextok_ar_3b
  model_id: EPFL-VILAB/FlexAR-3B-T2I

  generation_kwargs:
    replacement: false
    return_probs: true

  decode_kwargs:
    timesteps: 20
    guidance_scale: 7.5
    perform_norm_guidance: true
    use_same_noise_per_prompt: true
```
