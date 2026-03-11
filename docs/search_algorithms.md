# Search Algorithms

`SoT` provides three search strategies, all implementing the `BaseSearchAlgorithm` interface. Each is registered by name and configured via YAML.

## Best-of-N (`best_of_n`)

**Implementation:** [`best_of_n.py`](../sot/sot/search_algorithms/best_of_n.py) | **Config:** [`best_of_n.yaml`](../sot/configs/components/search_algorithms/best_of_n.yaml)

The simplest strategy: generate N complete sequences, score all, return the best.

```yaml
# sot/configs/components/search_algorithms/best_of_n.yaml
search:
  name: best_of_n
  n_samples: 50              # Number of full sequences to generate
  # decode_timesteps is controlled by ar_prior.decode_kwargs.timesteps
```

```bash
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_bon50_ir
```

## Beam Search (`beam`)

**Implementation:** [`beam_search.py`](../sot/sot/search_algorithms/beam_search.py) | **Config:** [`beam_search.yaml`](../sot/configs/components/search_algorithms/beam_search.yaml)

Incrementally generates tokens while maintaining the top-K candidates at each step. The token schedule controls how many tokens are added per step.

```yaml
# sot/configs/components/search_algorithms/beam_search.yaml
search:
  name: beam
  beam_width: 5              # Number of beams to keep
  max_steps: 9               # Number of search steps
  candidates_per_beam: 10    # Branching factor per beam
  token_schedule: geometric  # "geometric", "linear", "fixed", or list of ints
```

**Token schedules:**
- `geometric`: Tokens double each step (1, 2, 4, 8, ..., 256). Well-suited for FlexTok's coarse-to-fine structure.
- `linear`: Fixed increment per step (e.g., 64 tokens/step). Suited for Janus.
- `fixed`: One token per step.
- list of ints: Explicit cumulative token counts per step. When the AR prior exposes scale boundaries (e.g. Infinity), values are interpreted as 1-indexed scale numbers and the schedule is built from them.

```bash
# FlexTok beam search (geometric schedule)
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir

# Janus Pro beam search (linear schedule)
python sot/run_search.py --config-name eval/janus/geneval_janus_pro_beam_ir

# Infinity beam search (scale-indexed schedule)
python sot/run_search.py --config-name eval/infinity/geneval_infinity_beam_ir
```

## Lookahead Beam Search (`lookahead`)

**Implementation:** [`lookahead_search.py`](../sot/sot/search_algorithms/lookahead_search.py) | **Config:** [`lookahead_search.yaml`](../sot/configs/components/search_algorithms/lookahead_search.yaml)

Same as beam search, but before scoring, partial token sequences are AR-completed to give the decoder more context. Only the original (non-extended) tokens are kept in the beam.

```yaml
# sot/configs/components/search_algorithms/lookahead_search.yaml
search:
  name: lookahead
  beam_width: 5
  max_steps: 9
  candidates_per_beam: 10
  token_schedule: geometric
  lookahead_number: -1       # -1 = complete to end
  max_lookahead_step: -1     # -1 = apply at all steps
```

**Typical configurations per model:**
- FlexTok: `lookahead_number: 8` — extend by 8 extra tokens
- Janus: `lookahead_number: -1` — always complete to full sequence
- Infinity: `lookahead_number: -1, max_lookahead_step: 8` — complete for early scales only

```bash
python sot/run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_la8_ir
```
