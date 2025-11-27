# Traffic Detection MLOps Pipeline

An end-to-end machine learning operations (MLOps) system for real-time traffic object detection using YOLOv8. This project demonstrates the complete ML lifecycle from training to deployment, monitoring, and retraining.

## URLs

Once running locally, access:

| Service | URL | Description |
|---------|-----|-------------|
| **FastAPI Root** | `http://127.0.0.1:8001/` | API homepage, system status amd model info |
| **API Documentation** | `http://localhost:8001/docs` | Interactive Swagger UI for image upload and detection |
| **Streamlit UI** | `http://localhost:8501/` | streamlit web interface |


## Project Description

This project implements a **traffic object detection system** that identifies vehicles and pedestrians in images using a custom-trained YOLOv8 model.

### Key Features

- **Real-time Object Detection**: Detects cars, bicycles, motorcycles, buses, and people
- **REST API**: FastAPI backend serving predictions with 200ms average response time
- **Web UI**: Streamlit interface with data visualizations and insights
- **Model Retraining**: Automated retraining pipeline with new data
- **Model Monitoring**: Track uptime, performance metrics, and detection statistics
- **Load Testing**: Locust integration for stress testing

### Detected Objects

The model identifies 5 traffic object classes:
- **Car**
- **Bicycle**
- **Motorcycle**
- **Bus**
- **Person**

### System Output

Given a traffic image, the system returns:
- **Detection count** by object type (e.g., "10 cars, 2 bicycles, 1 person")
- pixel coordinates
- **Confidence scores** for each detection (typically 80-95%)


##  Project Structure


traffic-detection-mlops/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebook/
│   └── traffic_detection.ipynb       # Model training notebook
│
├── src/
│   ├── model.py                       # Training/retraining module
│   ├── preprocessing.py               # Data preprocessing functions
│   └── prediction.py                  # Prediction utilities
│
├── api/
│   └── app.py                         # FastAPI application
│
├── models/
│   └── traffic_detection_model.pt    # Trained YOLOv8 model
│
├── data/
│   ├── train/                         # Training images and labels
│   ├── valid/                         # Validation data
│   ├── test/                          # Test images
│   └── data.yaml                      # Dataset configuration
│
├── ui/streamlit_app.py                # Streamlit web interface
│
└── test_retraining.py                 # Retraining test script




## Setup Instructions 

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd MLOP
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file exists**
   ```bash
   # Check if model file is present
   ls models/traffic_detection_model.pt
   ```


## Running the Application

### Step 1: Start the FastAPI Backend

```bash
python api/app.py
```

**Expected output:**
```
✓ Model loaded successfully from models/traffic_detection_model.pt
INFO: Uvicorn running on http://127.0.0.1:8001
INFO: Application startup complete.
```

**Test the API:**
```bash

# Open in browser
http://localhost:8001/docs
```

### Step 2: Start the Streamlit UI

Open a **new terminal** and run:

```bash
streamlit run ui/streamlit_app.py
```

The UI will open automatically at `http://localhost:8501/`

### Step 3: Make Predictions

**Using the UI**
1. Go to the **Prediction** tab
2. Upload a traffic image (JPG/PNG)
3. Click **"Run Detection"**
4. View results with confidence scores


### Retraining Process

The system automatically:
1. **Backs up** the current model
2. **Preprocesses** images (resize to 640x640, normalize)
3. **Trains** with data augmentation (flips, rotations)
4. **Validates** on validation set
5. **Saves** improved model
6. **Reloads** into API without downtime


## Model Performance

### Training Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 94% |
| **Precision** | 91% |
| **Recall** | 89% |
| **F1-Score** | 90% |
| **mAP@50** | 87% |

### Preprocessing Steps

1. **Image Resizing**: All images resized to 640×640 pixels
2. **Normalization**: Pixel values scaled to [0, 1]
3. **Data Augmentation**: 
   - Horizontal flips
   - Random rotations (±10°)
   - Brightness adjustments
   - Mosaic augmentation (YOLO-specific)

### Evaluation

Model evaluation is performed in `notebook/traffic_detection_training.ipynb` with:
- Confusion matrix analysis
- Per-class precision/recall
- Confidence threshold analysis
- Validation on 20% holdout set


### Performance Results

| Metric | Value |
|--------|-------|
| **Requests/sec** | ~45 |
| **Average Response Time** | 220ms |
| **95th Percentile** | 450ms |
| **Failure Rate** | 0% |


## Visualizations

The Streamlit UI provides three key insights:

### 1. Object Class Distribution
Shows the distribution of detected objects in the dataset. Cars dominate at 57%, followed by people (21%) and bicycles (12%).

### 2. Confidence Distribution  
Displays prediction confidence scores. 75% of detections have >80% confidence, indicating strong model performance.

### 3. Image pixel values

## Dependencies

Main libraries used:
- **ultralytics** (YOLOv8): Object detection model
- **fastapi**: REST API framework
- **streamlit**: Web UI
- **plotly**: Interactive visualizations
- **locust**: Load testing
- **Pillow**: Image processing
- **requests**: HTTP client

Check `requirements.txt` for complete list.



## Results Summary

This project demonstrates:

 **End-to-end ML pipeline** from training to deployment  
 **RESTful API** serving predictions with <250ms latency  
 **User-friendly UI** with data visualizations  
 **Automated retraining** with new data  
 **Performance monitoring** and system health checks  
 **Load testing** showing system scalability  

### Future Improvements

- Add user authentication
- Real-time video stream detection
- Mobile app integration


## Demo link

Youtube ( https://youtu.be/lEKTKXgDme8 )
