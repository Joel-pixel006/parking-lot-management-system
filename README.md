# AI-Powered Parking Lot Management System

An AI-based parking lot management system that uses **computer vision and YOLO object detection** to detect vehicles and monitor parking spaces from video footage.

The project is designed to automate parking-space monitoring and reduce the need for manual checking of available and occupied parking spots.

---

##  Project Overview

Traditional parking systems often rely on manual monitoring or basic sensors to determine whether parking spaces are occupied.

This project explores a **computer-vision-based approach** where a camera/video feed is processed to identify vehicles and determine the occupancy of predefined parking spaces.

The system uses **YOLO-based object detection** along with manually configured parking-space coordinates.

### Main workflow

```text
Camera / Video
      ↓
Video Frame
      ↓
YOLO Object Detection
      ↓
Vehicle Detection
      ↓
Parking Space Analysis
      ↓
Occupied / Available Status
```

---

##  Features

- 🚘 Vehicle detection using YOLO
- 🅿️ Parking-space monitoring
- 📹 Video-based parking analysis
- 📍 Custom parking-space configuration
- 🤖 Computer-vision-based occupancy detection
- 🧠 YOLO model training support
- ⚙️ Configurable dataset and detection settings
- 🐍 Python-based implementation

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO | Vehicle/object detection |
| Ultralytics | YOLO model implementation |
| OpenCV | Video and image processing |
| Computer Vision | Parking-space analysis |
| Git | Version control |
| GitHub | Source-code hosting |

---

##  Project Structure

```text
Parking_Project/
│
├── .gitignore
│
├── data.yaml
│
├── main.py
│
├── setup_spots.py
│
├── train.py
│
├── dataset/              # Ignored by Git
│
├── sample_video.mp4      # Ignored by Git
│
├── yolov8n.pt            # Ignored by Git
├── yolov8m.pt            # Ignored by Git
└── yolov8x.pt            # Ignored by Git
```

---

##  File Description

### `main.py`

The main application file responsible for running the parking-lot detection system.

It processes the input video and uses the trained/pre-trained YOLO model to detect vehicles and analyze parking spaces.

---

### `setup_spots.py`

Used to configure the parking spaces that the system should monitor.

Parking-space coordinates can be defined based on the camera/video perspective.

This allows the system to work with a specific parking-lot layout.

---

### `train.py`

Contains the training workflow for the YOLO model.

It can be used to train a YOLO model using the project's custom dataset.

---

### `data.yaml`

Configuration file used by the YOLO training pipeline.

It contains information required to locate the dataset and define the object classes used during training.

---

### `.gitignore`

Specifies files and folders that should not be uploaded to GitHub.

Large machine-learning files such as model weights, datasets, and videos are excluded from the repository.

---

##  Computer Vision Approach

The project follows a computer-vision pipeline for parking monitoring.

### 1. Input

The system receives a video containing the parking area.

```text
Video → Individual Frames
```

### 2. Vehicle Detection

Each frame is processed using a YOLO object-detection model.

The model identifies vehicles present in the frame.

```text
Frame
 ↓
YOLO
 ↓
Vehicle Bounding Boxes
```

### 3. Parking-Space Configuration

Parking spaces are defined using coordinates corresponding to the camera view.

```text
Parking Lot
 ├── Spot 1
 ├── Spot 2
 ├── Spot 3
 ├── Spot 4
 └── ...
```

### 4. Occupancy Analysis

The detected vehicle positions are compared with the configured parking spaces.

The system can then determine whether individual parking spaces are occupied or available.

---

##  Dataset

The project supports training with a custom dataset.

The dataset is intentionally excluded from the GitHub repository because datasets and generated training files can be large.

The dataset configuration is defined through:

```text
data.yaml
```

If you want to reproduce the training process, place the required dataset in the expected directory structure before running the training script.

---

##  YOLO Models

The project uses YOLO model weights for object detection and training.

The local project may contain model files such as:

```text
yolov8n.pt
yolov8m.pt
yolov8x.pt
```

These model-weight files are excluded from GitHub using `.gitignore` because they can be relatively large.

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/parking-lot-management-system.git
```

Move into the project directory:

```bash
cd parking-lot-management-system
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages used by the project.

For example:

```bash
pip install ultralytics opencv-python
```

---

##  Running the Project

After installing the dependencies and placing the required model/video files in the project directory, run:

```bash
python main.py
```

The application will process the configured video input and perform vehicle/parking-space detection.

---

## 🏋️ Training the Model

If you want to train the model using the custom dataset:

```bash
python train.py
```

Make sure:

- The dataset is available locally.
- `data.yaml` points to the correct dataset paths.
- The required YOLO dependencies are installed.

---

##  Configuring Parking Spaces

Before running the complete system, parking spaces may need to be configured for the selected camera view.

Run:

```bash
python setup_spots.py
```

The configured parking positions can then be used by the main detection system.

---

##  Files Excluded from GitHub

The following files are intentionally excluded from version control:

```text
dataset/
*.pt
*.mp4
```

This prevents large datasets, model weights, and video files from unnecessarily increasing repository size.

---

##  Future Improvements

Potential improvements include:

- 📱 Web-based parking dashboard
- 🌐 Real-time camera-stream support
- 🅿️ Real-time available-space counter
- 📊 Parking analytics and statistics
- 🗄️ Database integration
- 📍 Multiple parking-lot support
- ☁️ Cloud deployment
- 📷 CCTV/IP-camera integration
- 📈 Historical parking-occupancy reports
- 🔔 Notifications for parking availability
- 👤 User and administrator dashboards

---

## Learning Objectives

This project provides practical experience with:

- Computer vision
- Object detection
- YOLO
- Dataset preparation
- Model training
- Video processing
- Python programming
- Git and GitHub
- Machine-learning project organization

---

## Author

**Joel Jose**

B.Tech Artificial Intelligence & Data Science

---

##  Contributing

Contributions, suggestions, and improvements are welcome.

If you would like to contribute:

```bash
git clone <repository-url>
```

Create a new branch:

```bash
git checkout -b feature/your-feature
```

Make your changes, commit them, and submit a pull request.

---

##  License

This project is intended for educational and portfolio purposes.

If you plan to distribute or use the project commercially, add an appropriate open-source license to the repository.
