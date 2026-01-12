# Final Project (MNIST): Training, Compression, and TFLite Micro Export

This project trains a small CNN on MNIST and produces multiple compressed variants, evaluates them, and exports the best INT8 model for microcontroller use. It is made to run on native device hardware.

## What It Builds
- Baseline FP32
- PTQ INT8 (post‑training quantization)
- QAT INT8 (quantization‑aware training)
- Pruned FP32 (~50% sparsity target)
- Pruned + PTQ INT8
- Pruned + QAT INT8

For each variant it reports:
- Size (KB), Accuracy (%), Loss
- Latency (ms/sample)
- Sparsity (% zeros)

It then exports:
- `models/best_tflm_model.tflite` (flatbuffer)
- `models/best_tflm_model.h` (C array for TFLite Micro)

## Repo Contents
- `model.py` — end‑to‑end pipeline (train, compress, convert, evaluate, export)
- `requirements.txt` — Python dependencies
- `scripts/setup_windows.ps1` — Windows setup script (venv + deps; optional run)
- `scripts/setup_wsl.sh` — WSL/Ubuntu setup script (venv + deps; optional run)
- `README.md` 
- `models/` — created on first run; stores outputs and plots

## Run It

Use the run scripts at the repo root. They set up a virtual environment, install dependencies, and run the pipeline.

- Windows PowerShell:
  ```powershell
  .\run.ps1
  ```

- WSL/Ubuntu:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

## Outputs (in `models/`)
- Trained/checkpointed models: `baseline_fp32.h5`, `pruned_fp32.h5`, …
- INT8 files: `ptq_int8.tflite`, `qat_int8.tflite`, `pruned_ptq_int8.tflite`, `pruned_qat_int8.tflite`
- Comparison plots and `model_comparison_summary.csv`
- Export: `best_tflm_model.tflite` and `best_tflm_model.h`

## Notes
- MNIST is downloaded automatically by TensorFlow on first run.
- Epochs for all phases are set to 10 in `model.py`.
- If you hit legacy Keras issues, clear the flag and rerun:
  - PowerShell: `Remove-Item Env:TF_USE_LEGACY_KERAS -ErrorAction SilentlyContinue`
  - WSL: `unset TF_USE_LEGACY_KERAS`
 
## My Results
- Best INT8 variant: Pruned + PTQ INT8
- Accuracy: ~99.1%
- Latency: ~0.09 ms/sample (WSL CPU)
- Model size: ~60.7 KB (TFLite INT8)
- Sparsity: ~49.9% zeros 

Note: Exact numbers vary slightly between runs and machines; see `models/model_comparison_summary.csv` for your run’s metrics and the generated plots in `models/`. I Averaged several runs on my machine.

## License / Attribution
MNIST dataset (Yann LeCun et al.) and TensorFlow Model Optimization Toolkit.

