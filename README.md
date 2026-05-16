# ClassEval-Pro: A Cross-Domain Benchmark for Class-Level Code Generation

ClassEval-Pro is a benchmark and evaluation framework for **class-level code
generation** with Large Language Models (LLMs). It targets *compositional code
creation* — building a complete, internally structured class from a
specification — and evaluates it under five generation strategies (Holistic,
Bottom-Up, Top-Down, Incremental, Compositional) with a unified pass@k
evaluator. The benchmark is produced by a fully automated, contamination-
resistant construction pipeline (post-cutoff GitHub mining, cross-domain class
composition, LLM-judge filtering, and coverage-verified reference solutions).

> This repository accompanies the paper *"ClassEval-Pro: A Cross-Domain
> Benchmark for Class-Level Code Generation,"* accepted to **AIware 2026**
> ([arXiv:2604.26923](https://arxiv.org/abs/2604.26923)).

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Benchmark Construction](#benchmark-construction)
- [Running an Experiment](#running-an-experiment)
- [Evaluation](#evaluation)
- [Generation Strategies](#generation-strategies)
- [Citation](#citation)
- [License](#license)

## Repository Structure

The five generation strategies share a single scaffolding package (`common/`).
Each strategy directory contains only its own `inference.py` (the strategy
logic); `main.py`, `utils.py`, and `test.py` are thin shims that re-export from
`common/`, giving one source of truth for the driver, model client, and
evaluator. All package and module names follow PEP 8 (lowercase `snake_case`).

```
ClassEval-Pro/
├── benchmark_construction/     # Dataset construction pipeline (paper Sec. 3)
│   ├── config.py               # Paths, thresholds, ratios, API/model settings
│   ├── stage1_mine.py          # GitHub repo search (11-domain matrix)
│   ├── stage2_extract.py       # Clone + AST extract + 3-level validation + structure
│   ├── stage3_merge.py         # The Stack-style dedup + id assignment + stats
│   ├── stage4_compose.py       # Intra/cross-domain pairing + AST-augmented skeletons
│   ├── stage5_generate.py      # Skeleton fusion + tests + 3-judge + solution + coverage
│   ├── run.py                  # Module entrypoint (runs stages 1..5)
│   └── run_pipeline.sh         # One-command full data-generation script
├── common/                     # Shared scaffolding (single copy)
│   ├── model_client.py         # ModelClient + code extraction + IO helpers
│   ├── evaluate.py             # AutoTest + pass@k computation (run_evaluation)
│   └── runner.py               # Generic multi-threaded inference driver
├── holistic/
│   ├── inference.py            # Holistic generation pipeline (strategy logic)
│   ├── main.py                 # Shim → common.runner (injects the pipeline class)
│   ├── utils.py                # Shim → common.model_client
│   ├── test.py                 # Shim → common.evaluate
│   └── run.sh                  # Run script
├── bottom_up/                  # Same 5-file shape (inference.py = bottom-up logic)
├── top_down/                   # Same 5-file shape (inference.py = top-down logic)
├── incremental/                # Same 5-file shape (inference.py = incremental logic)
├── compositional/              # Same 5-file shape (inference.py = compositional logic)
├── data.json                   # The released benchmark dataset
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md
```

## Installation

Python 3.10+ is required (the construction pipeline uses PEP 604 `X | None` type hints).

```bash
pip install -r requirements.txt
```

## Dataset

`data.json` is the released benchmark. Each item contains:

| Field | Description |
|-------|-------------|
| `task_id` | Unique task identifier |
| `class_name` | Target class name |
| `skeleton` | Class specification (signatures + docstrings) |
| `test` | Unit test suite |
| `test_classes` | List of test classes to run |
| `solution_code` | Coverage-verified reference implementation |

Every task is validated during construction by an LLM-judge ensemble and must
pass its test suite with > 90% line coverage, ensuring both the specification
and the reference solution are functionally correct.

## Benchmark Construction

`benchmark_construction/` reconstructs the dataset from scratch (paper Sec. 3)
in five chained stages. Each stage reads the previous stage's artifact from
`benchmark_construction/artifacts/`, so the pipeline is resumable.

| Stage | Module | Step |
|-------|--------|------|
| 1 | `stage1_mine.py` | GitHub repo search across the 11-domain matrix (post-cutoff, star-stratified) |
| 2 | `stage2_extract.py` | Clone → AST-extract stdlib-only classes → 3-level validation → structure to JSON |
| 3 | `stage3_merge.py` | The Stack-style dedup (key + exact SHA + MinHash near-duplicate) |
| 4 | `stage4_compose.py` | Intra/cross-domain pairing + AST-augmented skeleton rendering |
| 5 | `stage5_generate.py` | Skeleton fusion + test generation + 3-judge filter + reference solution + coverage filter |

No credentials or model names are hardcoded; they are read from environment
variables (placeholders default to `YOUR_*`, and the run script fails fast if
left unfilled):

| Variable | Used by | Purpose |
|----------|---------|---------|
| `GITHUB_TOKEN` | stage 1 | GitHub Search API token |
| `AZURE_API_KEY`, `AZURE_ENDPOINT` | stage 5 | Azure OpenAI credentials |
| `SKELETON_MODEL` | stage 5 | Skeleton-fusion model deployment name |
| `TESTCASE_MODEL` | stage 5 | Test-generation model deployment name |
| `SOLUTION_MODEL` | stage 5 | Reference-solution model deployment name |
| `JUDGE_MODEL_1/2/3` | stage 5 | Three independent LLM-judge deployment names |

Tunable thresholds (LOC bounds, method count, dedup similarity, the 67/233
intra/cross split, coverage cutoff, etc.) live in
`benchmark_construction/config.py`.

```bash
# from the repository root
export GITHUB_TOKEN=...    AZURE_API_KEY=...    AZURE_ENDPOINT=...
export SKELETON_MODEL=...  TESTCASE_MODEL=...   SOLUTION_MODEL=...
export JUDGE_MODEL_1=...   JUDGE_MODEL_2=...    JUDGE_MODEL_3=...

bash benchmark_construction/run_pipeline.sh            # stages 1..5
bash benchmark_construction/run_pipeline.sh --from 4   # resume at stage 4
bash benchmark_construction/run_pipeline.sh --only 4 5
```

Equivalent module entrypoint:
`python -m benchmark_construction.run [--from N] [--to N] [--only N ...]`.
The reconstructed benchmark is written to
`benchmark_construction/artifacts/dataset.json`; the released top-level
`data.json` is never overwritten.

## Running an Experiment

Each strategy is run from its own directory and shares the same CLI through
`common/runner.py`; only `inference.py` differs across strategies.

### Quick start

```bash
cd holistic
# set credentials + model deployment (or edit run.sh):
export AZURE_API_KEY=... AZURE_ENDPOINT=... MODEL_NAME=...
bash run.sh
```

### Manual run

```bash
cd holistic
python main.py \
    --output_dir ./output_holistic \
    --data_path ../data.json \
    --model your-model-name \
    --azure_api_key YOUR_API_KEY \
    --azure_endpoint YOUR_API_ENDPOINT \
    --temperature 0.2 \
    --max_length 16384 \
    --sample 5 \
    --num_workers 10 \
    --auto_test 1
```

| Argument | Description |
|----------|-------------|
| `--data_path` | Path to the benchmark dataset JSON file |
| `--output_dir` | Directory for generation results and metrics |
| `--model` | Model name or deployment name |
| `--temperature` | Sampling temperature |
| `--max_length` | Maximum generation length |
| `--sample` | Samples per task (for pass@k) |
| `--num_workers` | Concurrent worker threads |
| `--auto_test` | If 1, run evaluation automatically after generation |

To run another strategy, replace `holistic` with `bottom_up`, `top_down`,
`incremental`, or `compositional`.

## Evaluation

Evaluation runs automatically with `--auto_test 1`, or manually:

```bash
cd holistic
python test.py \
    --source_file_name ./output_holistic/results.json \
    --eval_data ../data.json \
    --model_name your-model-name
```

The evaluator computes pass@1, pass@3, pass@5, and average pass@1 at both the
function level and the class level, following the unbiased combinatorial
estimator from the Codex paper. Results are written to `pass_at_k_result.json`
and `detailed_result.json` in the output directory. Per-strategy temporary test
files are written under a `tmp/` directory and cleaned up afterward.

## Generation Strategies

| Strategy | Description |
|----------|-------------|
| **Holistic** | Generates the complete class in one model call. |
| **Bottom-Up** | Analyzes method dependencies, then implements from Level 0 (independent) upward. |
| **Top-Down** | Analyzes dependencies, then implements from the highest-level methods down to their helpers. |
| **Incremental** | Extracts the method list, then generates one method at a time, appending to accumulated context. |
| **Compositional** | Generates each method independently (with the skeleton as context) and assembles them. |

## Citation

If you use ClassEval-Pro, please cite our paper:

```bibtex
@misc{chen2026classevalprocrossdomainbenchmarkclasslevel,
      title={ClassEval-Pro: A Cross-Domain Benchmark for Class-Level Code Generation}, 
      author={Yeheng Chen and Chaoxiang Xie and Yuling Shi and Wenhao Zeng and Yongpan Wang and Hongyu Zhang and Xiaodong Gu},
      year={2026},
      eprint={2604.26923},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2604.26923}, 
}
```

## License

Released under the [MIT License](LICENSE), as part of the AIware 2026 paper
*ClassEval-Pro: A Cross-Domain Benchmark for Class-Level Code Generation*.
