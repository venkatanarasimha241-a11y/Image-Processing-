# AI Camera Monitoring System

A state-of-the-art, web-based intelligent surveillance and facial recognition platform built for real-time security monitoring, academic research, and enterprise access control.

Developed as a **Final-Year B.Tech Computer Science & Engineering Project**.

---

## 🌟 Key Highlights & Features

- 🎯 **Real-Time Object Detection**: Integrated with **Ultralytics YOLO** (YOLOv8) to accurately identify persons, vehicles, and objects in live video streams.
- 👤 **Facial Recognition Pipeline**: Automated facial extraction and biometric verification matching against registered personnel (`known_faces/`).
- 📅 **Student Timetable & Academic Details**: Real-time extraction of student roll number, class section, academic year, CGPA, and day-wise class timetable upon image capture or face match.
- 👨‍🏫 **Faculty Schedule & Class Allocation**: Immediate lookup of faculty ID, department, designation, and assigned lecture schedule with timings and classroom locations.
- 🔐 **Dual Login Models (Admin & Departments)**:
  - **Admin Model**: Full platform control, cross-department access, camera feeds, and person registration.
  - **Departments Model**: Department-specific access (CSE, ECE, IT, AI&DS, MECH) with localized student and faculty rosters.
- 📸 **Snapshot Capture & Image Upload Identification**: Instant snapshot analysis directly from live feed or file upload returning rich person profiles and schedules.
- 🚨 **Automated Alert System**: Flags unknown or unauthorized visitors instantly with unique tracking IDs (e.g. `UNKNOWN-0001`) and visual alert badges.
- ☁️ **Cloud Telemetry with Firebase**:
  - **Cloud Firestore**: Real-time storage of detection metadata, visitor logs, and student/faculty profiles with timetables.
  - **Firebase Cloud Storage**: Secure persistence of high-resolution evidence snapshots.
- 🛡️ **Intelligent Demo Mode & Graceful Degradation**: Operates seamlessly out-of-the-box even before custom AI weights (`best.pt`), hardware cameras, or Firebase credentials are provided.
- 🎨 **High-Aesthetic Light Theme Dashboard**: Commercial-grade, responsive UI engineered with Vanilla CSS, glassmorphism accents, animated KPI counters, live canvas charts, and mobile drawer support.
- ⏱️ **Debounce Cooldown Engine**: Configurable deduplication timer (`DETECTION_COOLDOWN = 5s`) to prevent database flooding during continuous person presence.

---

## 🏗️ System Architecture

```
                 CAMERA STREAM / OPENCV
                           ↓
               YOLO OBJECT DETECTION (YOLOv8)
                           ↓
                    [Person Detected]
                           ↓
                FACE RECOGNITION MATCHING
                ┌──────────┴──────────┐
                ↓                     ↓
          KNOWN PERSON          UNKNOWN PERSON
        (Verified Student/Staff)  (Security Alert Created)
                └──────────┬──────────┘
                           ↓
                 CAPTURE EVIDENCE IMAGE
                ┌──────────┴──────────┐
                ↓                     ↓
         FIREBASE STORAGE      CLOUD FIRESTORE
      (or local fallback)     (Audit Log Record)
                └──────────┬──────────┘
                           ↓
                    FLASK REST API
                           ↓
             MODERN MONITORING DASHBOARD
```

---

## 📁 Project Folder Structure

```
AI-Camera-Monitoring/
│
├── app.py                     # Flask web server, page routes & REST endpoints
├── firebase_config.py         # Firebase Admin SDK initialization & fallback logic
├── firebase_service.py        # Firestore CRUD & Storage upload abstractions
├── yolo_detector.py           # Ultralytics YOLO inference wrapper
├── recognition.py             # Face detection & template matching pipeline
├── camera.py                  # OpenCV multi-threaded camera stream & evidence capture
├── requirements.txt           # Python dependency specifications
├── serviceAccountKey.json     # Firebase service account credentials (template)
├── .gitignore                 # Security exclusion for credentials & evidence
├── README.md                  # Comprehensive project documentation
│
├── models/
│   ├── best.pt                # Custom fine-tuned YOLO model weights
│   └── README.md              # Weight placement instructions
│
├── known_faces/               # Directory for registered facial reference photos
│   ├── STU001_Rahul_Kumar.jpg
│   ├── STU002_Priya_Sharma.jpg
│   └── FAC001_Dr_Rajesh_Iyer.jpg
│
├── captured_images/           # Local evidence image repository (auto-created)
│
├── templates/
│   ├── index.html             # Main surveillance dashboard with live stream & KPIs
│   ├── login.html             # Split-screen branded authentication interface
│   ├── persons.html           # Personnel management & registration directory
│   └── detections.html        # Comprehensive detection audit log & CSV exporter
│
└── static/
    ├── css/
    │   └── style.css          # Master light-themed responsive design system
    ├── js/
    │   ├── dashboard.js       # KPI counters, canvas charts, and real-time polling
    │   └── camera.js          # Stream controls, snapshot capture, and fullscreen
    └── images/                # Vector SVG avatars and evidence previews
```

---

## 💻 Tech Stack

- **Backend**: Python 3.10+, Flask, Threading, NumPy, Pillow
- **Computer Vision**: OpenCV (`cv2`), Ultralytics YOLOv8, Haar Cascade Face Classifiers
- **Cloud Database & Storage**: Firebase Admin SDK (Cloud Firestore, Firebase Cloud Storage)
- **Frontend**: HTML5 Semantic Architecture, Modern Vanilla CSS3 (Custom Properties & Keyframes), Vanilla JavaScript (ES6+, Fetch API, HTML5 Canvas)

---

## ⚙️ Installation & Setup

### 1. Clone & Navigate to the Repository
```bash
git clone https://github.com/venkatanarasimha241-a11y/Image-Processing-.git
cd Image-Processing-
```

### 2. Set Up Virtual Environment

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔥 Firebase Configuration

The application works seamlessly in **Demo Mode** without Firebase credentials. To connect to your live Firebase Cloud project:

1. Go to the [Firebase Console](https://console.firebase.google.com/).
2. Create a new Firebase project (e.g., `ai-camera-monitoring`).
3. Enable **Cloud Firestore** in Test or Production mode.
4. Enable **Firebase Storage**.
5. Navigate to **Project Settings** (gear icon) ➔ **Service accounts**.
6. Click **Generate new private key** and download the JSON file.
7. Rename the downloaded file to `serviceAccountKey.json` and place it in the root project folder (replacing the placeholder).
8. *(Optional)* Set environment variables in your `.env` file:
   ```env
   FIREBASE_PROJECT_ID=your-project-id
   FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
   ```

> [!SECURITY NOTE]
> `serviceAccountKey.json` is strictly added to `.gitignore`. Never commit private service keys to GitHub or expose them in client-side code.

---

## 🧠 YOLO Model Configuration

1. If you have custom trained YOLO weights from a dataset (e.g. trained on person detection or campus CCTV footage), place the file at:
   ```
   models/best.pt
   ```
2. If `models/best.pt` is not present:
   - The detector automatically attempts to load the standard lightweight `yolov8n.pt`.
   - If offline, it activates the **Demo Detection Engine** with synthetic tracking boxes and zero crashes.

---

## 👥 Registering Known Faces

To add authorized individuals for face recognition:

### Method A: Via Web UI
1. Navigate to `http://127.0.0.1:5000/persons`.
2. Click **"Add Person"**.
3. Fill in Full Name, ID, Department, Role, and upload a portrait photograph.
4. The system automatically registers the record and refreshes the recognition database.

### Method B: Via File System
1. Save front-facing portrait photos directly inside the `known_faces/` folder using the naming convention:
   ```
   [ID]_[First_Name]_[Last_Name].jpg
   ```
   *Examples*:
   - `STU001_Rahul_Kumar.jpg`
   - `STU002_Priya_Sharma.jpg`
   - `FAC001_Dr_Rajesh_Iyer.jpg`
2. Restart the app or click **Refresh** on the dashboard.

---

## 🚀 Running the Application

Start the Flask application server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Demo Login Credentials:
- **Email**: `admin@aicamera.com`
- **Password**: `admin123`
- *(Or simply click the **⚡ Quick Demo Login** button on the login screen)*

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main surveillance monitoring dashboard |
| `GET` | `/login` | Branded user authentication portal |
| `GET` | `/persons` | Registered personnel directory & management |
| `GET` | `/detections` | Detection audit history with image evidence |
| `GET` | `/video_feed` | Multipart MJPEG real-time AI video stream |
| `POST` | `/api/login` | Authenticates security administrators |
| `GET` | `/api/stats` | Returns real-time KPI metrics and chart arrays |
| `GET` | `/api/detections` | Returns filtered detection records (`?status=&camera=&search=`) |
| `GET` | `/api/persons` | Returns list of registered persons |
| `POST` | `/api/persons` | Registers a new person with optional photo |
| `PUT` | `/api/persons/<id>` | Updates person record |
| `DELETE` | `/api/persons/<id>` | Deletes person from system |
| `POST` | `/api/camera/start` | Starts the camera feed |
| `POST` | `/api/camera/stop` | Stops the camera feed |
| `POST` | `/api/capture` | Takes an instant manual snapshot of current frame |
| `GET` | `/api/alerts` | Fetches active security alerts for unauthorized visitors |

---

## 💡 Troubleshooting & FAQs

- **Q: What if I don't have a physical webcam connected?**
  - **A**: The system automatically detects missing hardware and starts the **Intelligent Synthetic Stream**, rendering a live simulation feed complete with animated surveillance targets and timestamps so the full pipeline can be demonstrated anywhere.
- **Q: Why does the dashboard show "DEMO MODE ACTIVE"?**
  - **A**: If `serviceAccountKey.json` contains placeholder values, the app activates in-memory local caching for demo and evaluation purposes. Once valid Firebase keys are supplied, it transitions automatically to cloud Firestore.
- **Q: How to adjust detection cooldown?**
  - **A**: Set the environment variable `DETECTION_COOLDOWN=10` (in seconds) in your `.env` file or export it in your shell.

---

## 🎓 Academic Viva & Presentation Notes

When presenting this project for your B.Tech viva:
1. **Pipeline Explanation**: Explain how OpenCV captures frames in a dedicated thread to eliminate I/O blocking; the frames are fed into YOLOv8 for spatial bounding-box regression; detected person patches are passed to the recognition engine for feature distance comparison; positive matches trigger a cooldown check before uploading evidence blobs to Firebase.
2. **Debounce Logic**: Highlight that continuous frame-by-frame database writes would cause 30 writes/sec; the cooldown mechanism caps alerts to 1 event per 5 seconds per subject, optimizing cloud bandwidth and Firestore costs.
3. **Resilience & Fault Tolerance**: Emphasize that every third-party integration (camera, model weights, cloud DB) includes safe fallbacks preventing application crashes during demonstrations.
