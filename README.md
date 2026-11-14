
# FaceVision – Intelligent Face Detection & Recognition System

FaceVision is a Python-based face detection and recognition application developed using OpenCV and Streamlit. It provides an interactive dashboard where users can detect faces in images, register individuals, train a recognition model, and recognize previously registered faces.

The project demonstrates the practical implementation of computer vision and traditional machine learning techniques through a simple, user-friendly interface.

## ✨ Features

- **Face Detection:** Detects human faces in uploaded images using OpenCV Haar Cascade classifiers.
- **Face Registration:** Allows users to register individuals by uploading their face images.
- **Model Training:** Trains an LBPH face recognition model using registered face samples.
- **Face Recognition:** Attempts to identify registered individuals and labels unrecognized faces as Unknown.
- **Interactive Dashboard:** Provides a Streamlit-based interface for accessing different project functionalities.
- **Model Management:** Saves and loads trained recognition models for future use.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Face detection and recognition |
| OpenCV LBPH | Face recognition algorithm |
| NumPy | Image and numerical processing |
| Pillow (PIL) | Image handling |
| Streamlit | Interactive web interface |

## 📂 Project Structure

```text
FaceVision/
│
├── app.py
├── face_engine.py
├── recognition_engine.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── faces/
    └── models/
```

*Note: The `data/` directory stores registered face samples and trained model files locally. These files are excluded from the public repository for privacy.*

## ⚙️ Installation and Setup

Follow these steps to run FaceVision on your computer.

### 1. Clone the Repository

```bash
git clone https://github.com/bintemahmood71-prog/FaceVision.git
```

### 2. Navigate to the Project Directory

```bash
cd FaceVision
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL displayed in your terminal.

## 🚀 How to Use

1. **Open the Dashboard:** Launch FaceVision to access the main interface.
2. **Register Faces:** Upload face images and register individuals in the system.
3. **Train the Model:** Train the recognition model using the registered face samples.
4. **Detect Faces:** Upload an image to detect faces.
5. **Recognize Faces:** Upload an image to identify faces against the registered dataset.

## 🧠 Algorithms

### Haar Cascade Classifier
Used for detecting faces in images. It uses a cascade of classifiers to identify facial patterns.

### Local Binary Patterns Histograms (LBPH)
Used for face recognition. LBPH extracts local texture patterns from face images and compares them with the trained samples to identify possible matches.

## ⚠️ Limitations

- Detection accuracy can vary depending on lighting, image quality, face angles, and image conditions.
- Haar Cascade may produce false detections or miss faces in challenging images.
- LBPH recognition is sensitive to variations in facial expressions, lighting, and angles.
- Recognition results are not guaranteed to be accurate and should not be treated as identity verification.

## 🔒 Privacy

FaceVision processes face images locally within the application. Registered face images and trained recognition data are excluded from the public GitHub repository.

Use this project responsibly and only with appropriate consent when collecting or processing facial images.

## 🎓 Project Information

**Project Name:** FaceVision – Intelligent Face Detection & Recognition System

**Category:** Computer Vision / Artificial Intelligence

**Developed Using:** Python, OpenCV, NumPy, Pillow, and Streamlit

**Purpose:** Internship Project

---

*FaceVision is an educational project developed to explore practical applications of computer vision and face recognition using Python.*
