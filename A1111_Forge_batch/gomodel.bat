@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set CUDA_MODULE_LOADING=LAZY
set COMMANDLINE_ARGS=--xformers --api --ckpt-dir "C:\SDXL-Minimal\stable-diffusion-webui\models\Stable-diffusion" --lora-dir "C:\SDXL-Minimal\stable-diffusion-webui\models\Lora" --styles-file="kstyles.csv" --styles-file="styles.csv"

call webui.bat