# PGIM

Source code for “Web API Recommendation via High-Order Interaction Modeling and Semantic Integration with Large Language Models.”

## Requirements

Use Python 3.10. The Python dependencies are listed in `code/requirements.txt`:

| Package | Version |
| --- | --- |
| PyTorch (`torch`) | 2.8.0 |
| NumPy (`numpy`) | 2.2.6 |
| SciPy (`scipy`) | 1.15.2 |
| PyYAML (`PyYAML`) | 6.0.3 |

## Installation with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the following commands from the repository root:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r code/requirements.txt
```

These commands create a virtual environment in `.venv` and install the required packages. See the [uv environment guide](https://docs.astral.sh/uv/pip/environments/) and [package installation guide](https://docs.astral.sh/uv/pip/packages/) for details.

## Data

The `data/` directory contains two split settings:

| Directory | Split | Contents |
| --- | --- | --- |
| `data/leave_out/` | Leave-one-out setting | `train_dict.pkl`, `test_dict.pkl`, `mashup_api_dict.pkl`, `mashup_desc_dict.pkl`, and `api_desc_dict.pkl`. |
| `data/cold_start/` | Mashup-level 8:1:1 train/validation/test split | `train_cases.jsonl`, `valid_cases.jsonl`, and `test_cases.jsonl` . |

The cold-start split uses disjoint Mashup sets for training, validation, and testing.

## Training

```bash
python code/train.py \
    --seed 42 \
    --device cuda \
    --output outputs/seed42
```

CUDA training requires a compatible NVIDIA GPU and driver. The output directory must be new or empty. 