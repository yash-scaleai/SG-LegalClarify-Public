# SG-LegalClarify

Paper: [arXiv link TBD]

For each scenario, the model under test first decides whether the user's question gives
enough facts for a decisive answer. If it does, the model answers in one turn. If not, it
can ask clarifying questions for up to 8 turns, answered by an LLM user simulator that
holds the scenario's ground-truth facts. Each final conversation is then graded against
the scenario's rubric by a panel of three LLM judges.

## Contents

| Path | What it does |
|---|---|
| `run_eval.py` | Generates the conversations: gate, multi-turn loop, model routing, result files. |
| `user_simulator.py` | Tool-gated user simulator (fact selector, answer rendering, handoff) and turn classifier. |
| `excel_scenarios.py` | Loads scenarios, rubrics and ground-truth variables from `SG-LegalClarify_public.xlsx`. |
| `web_search/` | Web search for the model under test (`--web-search`). See its README. |
| `scripts/run_paper_3seeds.sh` | Runs the full sweep for seeds 42, 43 and 44. |
| `scripts/run_sharded_experiment.sh` | Runs one seed as N parallel shards, after a preflight check. |
| `scripts/preflight_models.py` | Sends one request to each model before a long run. |
| `scripts/serve_*.sh` | Serves the three local models with vLLM. |
| `scripts/run_panel_grading.py` | Grades a results folder with the 3-judge panel (`--heal` re-asks missing verdicts, `--apply` writes the scores). |

## Setup

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in the keys
```

All API models are called through a [LiteLLM](https://docs.litellm.ai/) proxy, which
needs keys for the providers you want to run. Model ids in `run_eval.MODELS` are LiteLLM
route names. `docs/reproducing.md` lists every model, its route and the settings
sent to it.

The public split is `SG-LegalClarify_public.xlsx`: 550 of the 1,100 source questions
(half of each category, with every underspecified question kept with its fully specified
counterpart), giving 930 evaluation instances. The other half is held out for private
evaluation.

## Quick start

Generate conversations for one model on a few scenarios, then grade them:

```bash
echo '["<task_id_1>", "<task_id_2>"]' > subset.json
uv run python run_eval.py --models anthropic/claude-opus-4-8 \
  --scenario-subset subset.json --results-root results_demo --seed 42 --run-id 1
uv run python scripts/run_panel_grading.py --results-dir results_demo
uv run python scripts/run_panel_grading.py --results-dir results_demo --apply
```

`docs/reproducing.md` gives the full procedure for the paper's runs.

## Output format

Each scenario gets `<results-root>/<scenario_key>/<scenario_key>_base.json`:

```
{timestamp, scenario, trials: [
  {model, model_label, prompt_key, conversation, strategy, selected_mode,
   variables_asked, num_vars_asked, simulator_selections, rubric_results, mandatory_passed,
   mandatory_total, score, panel_grading, error, duration_seconds, ...}
]}
```

- `selected_mode` is the gate decision: `decisive` or `facts_gathering`.
- `strategy` is `clarified` if the model asked at least one clarifying question.
- `variables_asked` marks each hidden variable the user simulator selected, i.e. that
  the model asked for; `num_vars_asked` counts them. Variable coverage is computed from it.
- `simulator_selections` lists every simulator reply: the assistant turn it answered
  (`assistant_turn`), the variables its fact selector chose (`selected_variables`), and
  whether the reply was the decisive handoff (`handoff`).
- `rubric_results[*].passed` is the panel majority and `votes` holds each judge's verdict.
- `score` is the percentage of Mandatory rubric items passed.

Grading also writes `<scenario_key>_panel_votes.jsonl` (one record per trial and judge).
Generation writes one chain-of-thought JSONL per process under `data/`, one record per
trial, with per-turn reasoning, the simulator selections and, for web-search runs, every
search and page visit.

## Citation

TBD

## License

TBD
