# Face Recognition Attendance Management System

A Python-based attendance management system that uses face recognition technology to mark attendance automatically.

## Features

- Real-time face detection and recognition
- Automatic attendance marking (IN/OUT)
- API integration for attendance synchronization
- User-friendly web interface
- Support for multiple users
- Location-based attendance tracking

## Requirements

- Python 3.7+
- OpenCV
- dlib
- Flask
- scikit-learn
- pandas
- numpy
- joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/trilokluniya27/Face_Recognition_Attendance_Management_System_Using-_Machine_Learning.git
cd Face_Recognition_Attendance_Management_System_Using-_Machine_Learning
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the required models:
- haarcascade_frontalface_default.xml
- shape_predictor_68_face_landmarks.dat
- dlib_face_recognition_resnet_model_v1.dat

4. Place the models in the `models` directory

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and go to `http://localhost:5000`

3. Add new users:
   - Click on "Add New User"
   - Enter name and employee ID
   - Follow the instructions to capture face images

4. Mark attendance:
   - Click on "Start Attendance"
   - The system will automatically detect and recognize faces
   - Attendance will be marked automatically

## Project Structure

```
.
├── app.py                  # Main application file
├── attendance_api.py       # API integration
├── models/                 # ML models
├── static/                 # Static files
│   ├── faces/             # User face images
│   └── face_recognition_model.pkl  # Trained model
├── templates/              # HTML templates
└── Attendance/             # Attendance records
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.