@echo off
echo Installing dependencies...

:: Rimuovi versioni precedenti
pip uninstall -y grpcio grpcio-tools protobuf mediapipe tensorflow

:: Installa wheels pre-compilati per Windows
pip install --no-cache-dir -r requirements.txt

echo Installation complete! 