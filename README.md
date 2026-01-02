# Rubik's Cube Detector & Solver

A DearPyGui-based application that scans a physical Rubik's Cube with a webcam using YOLO for detection, reconstructs face colors, and produces a solution using Kociemba's algorithm.

https://github.com/user-attachments/assets/de32d4c6-ca91-43ef-88c4-ae9894e5c7e0

## Requirements
- Python 3.9+
- Webcam (for scanning)
- Recommended GPU support for PyTorch (falls back to CPU)
- Python packages: in requirements.txt

Install the dependencies (PowerShell example):
```powershell
python -m pip install -r requirements.txt
```
Adjust the PyTorch install command for your CUDA version or use the CPU-only wheel if needed.

## Key Files
- `main.py` — main GUI scanner/solver; runs the webcam pipeline and visualization.
- `rubik_face_read.py` — cube state representation, move application, and color extraction helpers.
- `YOLO.pt` — pretrained YOLO model for cube detection.

## Running the Scanner & Solver
From the repository root:
```powershell
python main.py
```

Controls inside the app:
- `SPACE` — capture the current face
- `R` — redo the last captured face
- `C` — clear all captured faces
- `V` — open 3D view
- `A` — open solution animator
- Arrow keys — rotate the 3D/animation view
- `Q` — quit

Workflow:
1) Align a cube face in the camera view (follow the order) and press `SPACE` to capture each face. Guide:
- White (on top is blue) -> Yellow (on top is green) -> Green (on top is white) -> Orange (on top is white) -> Blue (on top is white) -> Red (on top is white).
2) The app resolves colors, validates the state, and runs Kociemba to compute a solution string.
3) Use `V` to inspect the reconstructed cube in 3D.
4) Use `A` to play/pause/step through the solution animation.

## Troubleshooting
- **Webcam not opening**: ensure no other app is using the camera; check the `cv2.VideoCapture(0)` index in `main.py`.
- **Slow inference**: confirm PyTorch is using CUDA; otherwise it will run on CPU.
- **Color misreads**: verify lighting and try recapturing faces; adjust thresholds in `rubik_face_read.py` if needed.
- **Kociemba install error**: This is a classic error when programming Python on Windows. Cause: The `kociemba` library (Rubik solving algorithm) is written in C for speed. When you use `pip install`, it tries to compile this C source code on your machine, but your machine lacks the compiler toolset named Microsoft Visual C++ Build Tools.

## License
MIT License (see `LICENSE`).
