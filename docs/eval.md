# Running Experiments & Datasets

## Running Experiments

All configuration is Hydra-based. Experiments are composed from three component configs (AR prior, search algorithm, verifier) plus a dataset config.

### 1. Use pre-built eval configs (recommended)

```bash
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_bon50_ir
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_la8_ir
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ensemble
python sot/run_search.py --config-name eval/janus/geneval_janus_pro_beam_ir
python sot/run_search.py --config-name eval/infinity/geneval_infinity_beam_ir
```

### 2. Override on the command line

```bash
# Change the verifier
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir \
    verifier=ensemble

# Change search parameters
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir \
    search.beam_width=10 search.max_steps=12

# Limit dataset size for testing
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir \
    dataset.num_samples=10

# Use custom prompts instead of a dataset
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir \
    prompts='["A painting of a mountain lake at sunset"]'
```

### 3. Write your own eval config

```yaml
# sot/configs/eval/my_experiment.yaml
# @package _global_

defaults:
  - /eval/base
  - /components/ar_priors/janus/janus_pro
  - /components/search_algorithms/lookahead_search
  - /components/verifiers/ensemble

search:
  name: lookahead
  beam_width: 10
  max_steps: 9
  candidates_per_beam: 5
  lookahead_number: -1

dataset:
  name: geneval
  num_samples: 50

seed: 42
output_dir: results
resume: true
num_results: 1
```

```bash
python sot/run_search.py --config-name eval/my_experiment
```

## Datasets

All datasets auto-download on first use:

| Dataset | Samples | Task | Config Key |
|---------|---------|------|------------|
| GenEval | 553 | Compositional T2I evaluation | `dataset.name: geneval` |
| DreamBench++ | 1350 | Subject-driven generation | `dataset.name: dreambench` |
| COCO | 300 (we use a subset) | General T2I | `dataset.name: coco` |

Or use custom prompts directly:

```bash
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir \
    prompts='["your prompt here"]'
```
