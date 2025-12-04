@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%

if not exist .venv ( py -3 -m venv .venv )
call .venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

py -m src.datos.prepare_kvasir --origen ".\Dataset" --salida ".\data" --val 0.15 --test 0.15
py -m src.modelo.train --epochs 15 --batch 32 --lr 0.0007 --salida ".\models\kvasir_resnet18.pt"