@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set CUDA_MODULE_LOADING=LAZY
set COMMANDLINE_ARGS=--xformers --api

call webui.bat
