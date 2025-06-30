# Enhanced Multi-Video Player Mapping Pipeline

This repository provides a complete, GPU-accelerated pipeline for detecting, tracking, and mapping players across multiple video feeds (e.g., broadcast and tacticam) using a custom YOLOv11 model. The pipeline extracts visual, spatial, and temporal features, computes a similarity matrix, and applies the Hungarian algorithm for optimal track-to-track matching. 

## Features

* **YOLOv11 Model Loading** with automatic GPU/CPU detection
* **Video Loading & Frame Extraction** (broadcast & tacticam)
* **Player Detection** with configurable confidence threshold and frame skipping
* **Visual & Spatial Feature Extraction** per detection
* **Temporal Feature Extraction** for motion analysis
* **Track Building** with distance and frame-gap constraints
* **Similarity Calculation** combining cosine, Euclidean, and correlation metrics
* **Optimal Mapping** using the Hungarian (linear sum assignment) algorithm
* **GPU-Optimized Pipeline** with adjustable parameters and memory management
* **Final Outputs**: frame-by-frame CSV and similarity heatmap visualization

## Requirements

* Python 3.8 or higher
* PyTorch
* OpenCV (cv2)
* NumPy
* SciPy
* scikit-learn
* pandas
* matplotlib
* seaborn

Install required packages via:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the repository**

   ```bash
   ```

git clone [https://github.com/your-username/player-mapping-pipeline.git](https://github.com/your-username/player-mapping-pipeline.git)
cd player-mapping-pipeline

````

2. **Set up a virtual environment** (optional but recommended)
    ```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\\Scripts\\activate  # Windows
````

3. **Install dependencies**

   ```bash
   ```

pip install -r requirements.txt

````

## Configuration

Adjust parameters in `config.py` or within the `Config` class:

| Parameter               | Description                                               | Default      |
|-------------------------|-----------------------------------------------------------|--------------|
| `MODEL_PATH`            | Path to custom YOLOv11 model file                         | `models/best.pt` |
| `BROADCAST_VIDEO`       | Path to broadcast video file                              | `data/broadcast.mp4` |
| `TACTICAM_VIDEO`        | Path to tacticam video file                               | `data/tacticam.mp4` |
| `CONFIDENCE_THRESHOLD`  | Minimum confidence for detections                         | `0.3`        |
| `FRAME_SKIP`            | Number of frames to skip between detections               | `3`          |
| `FEATURE_DIM`           | Dimension of visual feature vectors                       | `unknown`    |
| `OUTPUT_DIR`            | Directory for CSV and image outputs                       | `outputs/`   |

## Usage

```bash
# Basic detection and mapping pipeline
git clone https://github.com/Kxngh/Linked-Angles
cd player-mapping-pipeline
````

## Output

* **`outputs/tracking_results.csv`**: Frame-by-frame tracking data with matched `player_id`s
* **`outputs/similarity_matrix.png`**: Heatmap of similarity scores between broadcast and tacticam tracks
* **`outputs/gpu_optimized_*`**: GPU-accelerated results and visualizations


