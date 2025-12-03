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

- **`optical_flow.py`** - Main script for tracking deformation using optical flow with vector visualization, heatmap display, and deformation region detection
- **`run_vision.sh`** - Convenience script to activate venv and run the optical flow tracker

## Usage

### Quick Start

Using the convenience script:
```bash
./run_vision.sh
```

### Optical Flow Tracker

Manual activation and run:
```bash
source venv/bin/activate
python optical_flow.py --cam 0 --interactive_roi
```

Or with the convenience script:
```bash
./run_vision.sh --cam 0 --interactive_roi
```

**Arguments:**
- `--cam`: Camera index (default: 0)
- `--width`: Camera width (default: 640)
- `--height`: Camera height (default: 480)
- `--interactive_roi`: Enable interactive ROI selection on first frame
- `--cells`: Grid cells [cols rows] (default: [10,8])
- `--kpc`: Max corners per cell (default: 20)
- `--ql`: Quality level for feature detection (default: 0.01)
- `--minDist`: Minimum distance between features (default: 5)
- `--fb_thresh`: Forward-backward error threshold (default: 3.0)
- `--k_nn`: Neighbors for edge-strain calculation (default: 4)
- `--csv`: Output CSV file (default: "vector_log.csv")
- `--deform_k_mad`: MAD multiplier for deformation threshold (default: 2.0, lower=more sensitive)
- `--deform_min_cluster`: Minimum points per deformation cluster (default: 3)
- `--deform_eps`: Clustering distance in pixels (default: 20.0)

**Keyboard Controls:**
- `b` - Set/update baseline (use when undeformed)
- `m` - Mark max deformation frame (detects and visualizes deformation regions)
- `q` or `ESC` - Quit

## Dependencies

- opencv-python (4.10.0.84)
- numpy (1.26.4)
- scipy (>=1.11.0)
- scikit-learn (optional, for DBSCAN clustering - falls back to connected components if not available)

## Output

The vision system outputs:
- **Real-time visualization:**
  - `vector-lines` window: Shows displacement vectors with color-coded confidence
  - `deform-heat` window: Heatmap visualization of deformation magnitude
- **CSV log files:**
  - `vector_log.csv`: Frame-by-frame deformation metrics
  - `vector_log_marked.csv`: Marked frames with deformation region statistics

## Features

- **Optical flow tracking** with forward-backward validation
- **Global affine motion removal** to isolate local deformation
- **Baseline tracking** - measure displacement from undeformed state
- **Deformation region detection** - automatically finds and clusters high-displacement areas
- **Bounding box visualization** - shows detected deformation patches with statistics
- **Robust statistics** - uses median and MAD for outlier-resistant thresholding

