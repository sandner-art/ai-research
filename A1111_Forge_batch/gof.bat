@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set CUDA_MODULE_LOADING=LAZY
set COMMANDLINE_ARGS=--pin-shared-memory --cuda-malloc --cuda-stream --xformers --api --styles-file="kstyles.csv" --styles-file="styles.csv"

call webui.bat
