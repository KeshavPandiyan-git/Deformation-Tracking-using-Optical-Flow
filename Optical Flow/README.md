# Vision-Based Deformation Tracking

Computer vision system for tracking material deformation using optical flow with vector visualization and heatmap display.

## Setup

### 1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Files

- **`vectorlines_heatmap.py`** - Computer vision script for tracking deformation using optical flow with vector visualization and heatmap display

## Usage

### Quick Start

Using the convenience script:
```bash
./run_vision.sh
```

### Vector Lines Heatmap

Manual activation and run:
```bash
source venv/bin/activate
python vectorlines_heatmap.py
```

Or with the convenience script:
```bash
./run_vision.sh --cam 0 --width 640 --height 480 --interactive_roi --csv output.csv
```

**Arguments:**
- `--cam`: Camera index (default: 0)
- `--width`: Camera width (default: 640)
- `--height`: Camera height (default: 480)
- `--interactive_roi`: Draw ROI on first frame
- `--cells`: Grid cells [cols rows] (default: [10,8])
- `--kpc`: Max corners per cell (default: 20)
- `--ql`: Quality level for feature detection (default: 0.004)
- `--minDist`: Minimum distance between features (default: 3)
- `--fb_thresh`: Forward-backward error threshold (default: 1.0)
- `--k_nn`: Neighbors for edge-strain calculation (default: 4)
- `--csv`: Output CSV file (default: "vector_log.csv")
- `--a`: Force scale factor (default: 1.0)
- `--b`: Force offset (default: 0.0)

**Controls:**
- Press `q` or `ESC` to quit

## Dependencies

- opencv-python (4.10.0.84)
- numpy (1.26.4)

## Output

The vision system outputs:
- Real-time visualization windows showing vector lines and deformation heatmap
- CSV log file containing frame-by-frame deformation metrics

