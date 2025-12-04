@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%

if not exist .venv ( py -3 -m venv .venv )
call .venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

if not exist "data\split" (
  py -m src.datos.prepare_kvasir --origen ".\Dataset" --salida ".\data"
)

if not exist "models\kvasir_resnet18.pt" (
  py -m src.modelo.train --epochs 3 --batch 16 --lr 0.001 --salida ".\models\kvasir_resnet18.pt"
)

py -m streamlit run src\app.py --server.headless true