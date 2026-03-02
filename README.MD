# Car Tracking and Analytics Pipeline
## Overview
![alt text](assets/image.png)
**Project description:**
- This repository provides a computer-vision pipeline for real-time vehicle-detection, tracking, color-classification, and license-plate recognition.
- The application processes video-streams and exports analytical-data into a structured CSV-format alongside a rendered MP4 video-file.
- The architecture is optimized for edge-devices and headless-execution.

**Execution modes:**
- Executing this pipeline via Docker runs strictly in headless-mode without a live graphical user-interface (GUI), because Docker containers lack native display-servers by default.
- The Docker-execution will silently process the stream and output the final artifacts (`output.mp4` and `report.csv`) to the `artifacts` directory.
- If you want to see the real-time visual tracking-interface, you must execute the pipeline locally using the Python-environment method, since local execution has direct access to the host window-manager.

---

## Core Features & Task Compliance
**Fulfillment of requirements:**
- **Real-time tracking:** Assigns a persistent unique-ID to every new vehicle entering the frame using the ByteTrack algorithm.
- **Dynamic visualization:** Bounding-boxes are drawn using the exact RGB-color identified by the classification-model, accompanied by the recognized license-plate text directly on the frame.
- **Edge-optimization:** The entire pipeline runs strictly on CPU using ONNX Runtime, avoiding heavy PyTorch dependencies to maximize realtime-performance.
- **Bonus implementation (Stream filtering):** Frames without detected vehicles instantly bypass the heavy inference-pipeline, and vehicles below specific resolution-thresholds are completely skipped to prevent useless OCR-attempts on unreadable plates.

---

## Prerequisites
**System requirements:**
- Docker
- Docker Compose
- Python 3.10 or higher (for local-execution)
- FFmpeg utility (only required for local RTSP-simulation)

## Deployment Methods

### Method 1: Docker with real RTSP-stream
**Recommended approach:**
This is the recommended setup for production-environments connecting to live IP-cameras.

Open a terminal and launch the containerized-pipeline by passing your actual camera URL:
```bash
RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream docker compose up --build
```
**Windows PowerShell users:**
```powershell
$env:RTSP_URL="rtsp://admin:password@192.168.1.100:554/stream"; docker compose up --build
```

### Method 2: Docker with FFmpeg simulated RTSP-stream
**Network simulation:**
This approach tests the network-capabilities using a local video-file.

1. Initialize the local RTSP-server using Docker in a new terminal-window:
    ```bash
    docker run --rm -it -p 8554:8554 bluenviron/mediamtx
    ```
2. If FFmpeg is not installed on your system, install it using the appropriate package-manager:
    - **Ubuntu/Debian:** `sudo apt install ffmpeg`
    - **macOS:** `brew install ffmpeg`
    - **Windows:** `winget install ffmpeg`
3. Stream the local video to the RTSP-server using FFmpeg in a second terminal-window:
    ```bash
    ffmpeg -re -stream_loop -1 -i "/absolute/path/to/videos/traffic_1.mp4" -c copy -rtsp_transport tcp -f rtsp rtsp://localhost:8554/mystream
    ```
4. Launch the computer-vision pipeline targeting the RTSP-stream in a third terminal-window:
    ```bash
    RTSP_URL=rtsp://host.docker.internal:8554/mystream docker compose up --build
    ```
**Windows PowerShell users:**
```powershell
$env:RTSP_URL="rtsp://host.docker.internal:8554/mystream"; docker compose up --build
```

### Method 3: Docker with local video-file
**Simple container testing:**
This approach is the simplest way to test the pipeline without setting up a network-stream.

- Place the target video-file into the `videos` directory and name it `traffic_1.mp4`.
- Execute the following command to start the containerized-pipeline using the default fallback-path:
    ```bash
    docker compose up --build
    ```
- The processing-results will be saved automatically in the `artifacts` directory as `report.csv` and `output.mp4`.

### Method 4: Local Python-environment execution
**Local development (GUI enabled):**
This method is suitable for local-development and visual-debugging without Docker.  
**This is the only method that will display the live video-feed with detection-overlays.**

1. Create a virtual-environment:
    ```bash
    python -m venv .venv
    ```
2. **Windows:** Activate the virtual-environment:
    ```powershell
    .venv\Scripts\activate
    ```
   **macOS/Linux:** Activate the virtual-environment:
    ```bash
    source .venv/bin/activate
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. **Windows:** Run the main script using the default video-path:
    ```powershell
    .venv\Scripts\python.exe -m src.main
    ```
   **macOS/Linux:** Run the main script using the default video-path:
    ```bash
    .venv/bin/python -m src.main
    ```
5. **Windows:** Alternatively run the script by explicitly passing the video-file or RTSP-link:
    ```powershell
    .venv\Scripts\python.exe -m src.main --rtsp videos/traffic_1.mp4
    ```
   **macOS/Linux:** Alternatively run the script by explicitly passing the video-file or RTSP-link:
    ```bash
    .venv/bin/python -m src.main --rtsp videos/traffic_1.mp4
    ```

## Configuration
**Environment variables:**
- The repository includes a pre-configured `.env` file. It contains no confidential information and provides ready-to-use defaults. You can modify it directly or override variables via inline CLI-commands during the Docker-build.
- `HEADLESS_MODE`: Disables the graphical user-interface for server-execution (default in Docker).
- `SAVE_VIDEO`: Enables saving the processed frames to `artifacts/output.mp4` when set to `True`.
- `RTSP_URL`: Defines the source-path or network-URL for the video-stream.
- `PROFILE`: Runtime profile for inference (`cpu` or `gpu`). Defaults to `cpu`.
- `FRAME_SKIP`: Process every N-th frame to reduce CPU load. Defaults to `1` (every frame).
- `VEHICLE_CONF_THRESHOLD`: Minimum confidence for vehicle detections. Defaults to `0.20`.
- `PLATE_CONF_THRESHOLD`: Minimum confidence for plate detections. Defaults to `0.30`.
- `OCR_MIN_VEHICLE_WIDTH`: Minimum vehicle bounding-box width in pixels before OCR is attempted. Defaults to `400`.
- `OCR_MIN_VEHICLE_AREA`: Minimum vehicle bounding-box area in pixels before OCR is attempted. Defaults to `120000`.

---

## Algorithmic Approaches & Pipeline Workflow

### Step 1 — Video ingestion (`src/io/rtsp_capture.py`)

**What:** A dedicated background thread pulls frames from OpenCV's `VideoCapture` into a small ring-buffer.  
**Why:** Decoupling I/O from processing prevents the main pipeline from blocking on slow network reads. The ring-buffer drops stale frames so the pipeline always operates on the freshest available image, which is critical for real-time performance. Automatic exponential-backoff reconnection handles transient RTSP drops without human intervention, and prevents CPU-overload and network-flooding by gradually increasing the reconnect-delay when the RTSP-source goes offline.

---

### Step 2 — Vehicle detection (`src/inference/onnx_yolo.py`, `onnx/yolo10nano_car.onnx`)

**What:** YOLOv10-nano exported to ONNX is used for vehicle detection.  
**Why:**
- **YOLOv10** eliminates the traditional NMS post-processing step by using a dual-assignment head, which reduces latency per frame compared with YOLOv5/v8 equivalents.
- **Nano** variant gives the best speed / accuracy trade-off on edge hardware (Raspberry Pi, Jetson Nano, modest CPUs).
- **ONNX Runtime** with `ORT_ENABLE_ALL` graph optimisations lets the same weights run on CPU, CUDA, TensorRT, or CoreML by simply switching the `ExecutionProvider` — no code change is needed.
- Before passing a frame to the detector the frame is scaled so its longest side is at most 1000 px. This roughly halves the input area for typical 1080p streams with negligible accuracy loss.

---

### Step 3 — Multi-object tracking (`src/pipeline/vehicle_tracker.py`, `src/pipeline/tracker_core/`)

**What:** ByteTrack algorithm with a Kalman filter for state estimation.  
**Why:**
- **ByteTrack** associates *all* detection boxes — including low-confidence ones — using IoU matching in two stages. This dramatically reduces ID switches when a vehicle is partially occluded or temporarily undetected.
- The Kalman filter predicts the next bounding box position between frames, so tracking is robust to short detection gaps (e.g. missed detections in the keyframe).
- Keeps a `track_buffer` of 90 frames (~3 seconds at 30 fps), meaning a vehicle that disappears briefly keeps its ID on reappearance.

---

### Step 4 — License-plate detection and OCR (`src/inference/onnx_yolo.py`, `src/pipeline/plate_ocr.py`, `src/inference/plate_recognizer.py`)

**What:** A custom fine-tuned YOLOv10-nano model (`yolo10nano_plate.onnx`) localises the plate region within each vehicle crop. A PaddleOCR-based recogniser (detection + recognition ONNX models) then reads the text.  
**Why:**
- Two-stage approach (detect plate → OCR) is more efficient than running full-image OCR on every frame.
- The plate model was fine-tuned on traffic-camera data because base-models lack the precision required for small objects at traffic-camera angles.
- **Frame-level filter:** OCR is skipped for vehicles whose bounding box is narrower than `OCR_MIN_VEHICLE_WIDTH` (400 px) **and** smaller than `OCR_MIN_VEHICLE_AREA` (120 000 px²). Small or distant cars cannot produce readable plates — skipping them saves CPU cycles and avoids injecting garbage text (this addresses the optional filter requirement).
- **Per-track result caching** (`PlateOCRPipeline`): once a clean plate string has been recognised for a given track ID it is reused for subsequent frames without re-running OCR. A negative-caching mechanism permanently stops attempting to read a specific vehicle's plate after 10 consecutive failed tries, preventing infinite processing-loops on blurred or obstructed plates. Only higher-confidence re-reads can supersede a cached result.
- Regex post-validation (`_is_valid_plate`, `ocr_regex_patterns`) discards OCR hallucinations that look nothing like a real plate (no digits, all-same characters, etc.).

---

### Step 5 — Color detection (`src/analytics/color_detector.py`)

**What:** HSV-space thresholding over the upper 75 % of the vehicle crop.  
**Why:**
- Fully deterministic; takes less than 0.5 ms per frame and avoids the overhead of running a secondary neural-network for simple color-matching.
- HSV separates hue from lighting, making rules like "blue: H ∈ [100, 140]" stable under varying exposure.
- The upper-body crop excludes the road surface and shadow under the chassis, which would otherwise skew the dominant-color vote.
- Per-track exponential moving average smooths transient misreadings without introducing a model dependency.

---

### Step 6 — Visualisation (`src/analytics/visualizer.py`, `src/ui/overlay.py`)

**What:** Each vehicle bounding box and label (ID + plate) is drawn in the detected vehicle color. A separate analytics panel shows the per-session color histogram and the most recent plate readings.  
**Why:** Coloring the box to match the real car color was explicitly required and also makes the stream easy to interpret at a glance. A separate analytics window avoids obscuring the main video feed.

---

### Step 7 — Output and reporting (`src/app.py`)

**What:** A CSV report is written incrementally as tracks expire (after 5 s of absence). An optional MP4 output is written by a dedicated background thread (`_AsyncVideoWriter`).  
**Why:** Incremental writing prevents losing data if the process is killed. The async writer prevents the video-encoding step from adding latency to the main processing loop. Frame-padding synchronizes the output video-speed with real-time by dynamically duplicating frames to compensate for hardware-lag or network buffer-drops.

---

## Architecture Decisions

**Additional pipeline optimizations:**
- **Model Fine-Tuning:** I executed custom fine-tuning on the YOLOv10 Nano architecture specifically for license-plate detection, because base-models lack the precision required for small objects at traffic-camera angles; the full training-pipeline is publicly documented in the [Kaggle Notebook](https://www.kaggle.com/code/ituvtu/yolov10nano).
- **Graceful Shutdown:** Ensures data-integrity by saving the final CSV-report and finalizing the MP4-container even when the process is interrupted via Ctrl+C or SIGTERM.
- **Garbage Collection:** Prevents memory-leaks during continuous RTSP-streams by incrementally flushing inactive vehicle-tracks to disk and clearing them from system RAM after 5 seconds of absence.
- **Frame Padding:** Synchronizes output video-speed with real-time by dynamically duplicating frames to compensate for hardware-lag or network buffer-drops.
- **Exponential Backoff:** Prevents CPU-overload and network-flooding by gradually increasing the reconnect-delay when the RTSP-source goes offline.

**Edge / real-time optimisation summary:**

| Technique | Benefit |
|---|---|
| YOLOv10-nano ONNX | Fastest YOLO variant; no post-NMS; hardware-agnostic |
| Frame downscale before detection | ~2× fewer pixels → ~2× faster inference |
| `FRAME_SKIP` env variable | Drop frames to match hardware capability |
| Per-track OCR cache + negative cache | OCR runs at most once per track per result-quality level |
| Async video writer thread | Encoding never blocks the pipeline |
| Async RTSP reader thread | I/O never blocks the pipeline |
| ByteTrack dual-stage association | Fewer lost tracks → fewer redundant OCR calls |
| Small/distant vehicle filter | Avoids wasteful OCR on unreadable crops |
| HSV color thresholding | <0.5 ms per vehicle; no secondary neural network |

---

## Tools & Utilities
**Development scripts:**
- The `tools/` directory contains helper-scripts for the MLOps-pipeline.
- `download.py`: Automates the downloading of pre-trained weights and testing video-assets.
- `convert_to_onnx.py`: Handles the conversion of PyTorch-models (including the fine-tuned YOLOv10) to the optimized ONNX-format for CPU-inference.
- `setup_ocr_models.py`: Initializes the specific OCR-models required for license-plate text-recognition.
- These scripts demonstrate the model-lifecycle and are strictly optional for the end-user, because the Docker-build process or initial environment-setup handles these dependencies automatically.

---

## Acknowledgments
**Credits and references:**
- **Base tracker logic:** The raw mathematical-logic for the Kalman-filter and bounding-box matching was extracted from the [Kazuhito00/ByteTrack-ONNX-Sample](https://github.com/Kazuhito00/ByteTrack-ONNX-Sample) repository and integrated into the custom tracker, because it provides a lightweight, PyTorch-free mathematical foundation for CPU-execution.
- **Custom integration:** All other tracking-components (state-management, memory-buffers, garbage-collection, and headless-integration) were built from scratch around this extracted core, since the architecture required specific adaptations for real-time tracking-performance without heavy framework-dependencies.
