# Training data

`synthetic_full.csv` is generated from `ML/data/seeds/synthetic_seed.csv` by `python3 ML/data/generate_synthetic.py`. It is checked in as a reproducible training artifact. `baseline_metrics.json` records metrics for the locally trained baseline. The pickle model files are ignored by Git; regenerate them with `python3 ML/training/train_baseline.py`.

Optional external data can be fetched with `python3 ML/data/download_datasets.py` after installing the optional `datasets` dependency. Review dataset terms and suitability before redistribution or production use. Generated examples are synthetic and are not representative of real-world performance.
