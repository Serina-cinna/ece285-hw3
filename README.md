# ECE 285 Homework 3

This repository trains a DDPM on the 5,179-image Histopathology subset of the assigned Kaggle dataset.

## Run

Install dependencies with `pip install -r requirements.txt`, then run:

```bash
python code/prepare_data.py --data-root data
python code/train_color_whitened.py --data data/Histopathology --epochs 100
python code/evaluate_rgb_whitened.py --data data/Histopathology --checkpoint outputs/ddpm128_rgb_whitened/checkpoints/epoch_025.pt --color-transform outputs/ddpm128_rgb_whitened/color_transform.json --output outputs/evaluation/epoch25
```

The report and selected epoch-25 results are in `report/` and `outputs/`.
