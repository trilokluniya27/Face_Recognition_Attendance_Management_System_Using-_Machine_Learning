import cv2
import os
import numpy as np
import pandas as pd
import joblib
import dlib
from flask import Flask, request, render_template
from datetime import date, datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from attendance_api import post_attendance_swipe
import time

app = Flask(__name__)
nimgs = 200
imgBackground = cv2.imread("background.png")
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier(
    'models/haarcascade_frontalface_default.xml')
shape_predictor = dlib.shape_predictor(
    'models/shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(
    'models/dlib_face_recognition_resnet_model_v1.dat')

os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
# Update the CSV header creation
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        # Matches format: Name,employee_id,In-Time,Out-Time
        f.write('Name,employee_id,In-Time,Out-Time')

confidence_threshold = 0.90  # Default threshold - balanced between security and usability


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    # Enhance image quality
    img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # More robust face detection parameters
    faces = face_detector.detectMultiScale(
        img,
        scaleFactor=1.05,  # Smaller scale factor for more accurate detection
        minNeighbors=6,  # Increased for more reliable detection
        minSize=(30, 30),  # Larger minimum face size
        maxSize=(300, 300)  # Maximum face size limit
    )
    return faces


def train_model():
    faces, labels = [], []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = cv2.resize(img, (100, 100))  # Changed back to 100x100
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            faces.append(img.ravel())
            labels.append(user)

    if not faces:
        print("No faces found for training.")
        return

    faces_train, faces_test, labels_train, labels_test = train_test_split(
        faces, labels, test_size=0.2, random_state=42)

    # Using SVM classifier for better unknown detection
    model = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
    model.fit(faces_train, labels_train)
    train_accuracy = model.score(faces_train, labels_train)
    test_accuracy = model.score(faces_test, labels_test)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    joblib.dump(model, 'static/face_recognition_model.pkl')
    print("Model saved to 'static/face_recognition_model.pkl'.")


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['employee_id'], df['In-Time'], df['Out-Time'], len(
        df)


def add_attendance(name):
    if '_' not in name:
        print(f"Invalid name format: {name}")
        return

    username, userid = name.split('_')
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        display_time = datetime.now().strftime("%H:%M:%S")
        iso_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Get location data
        print("\n=== Location Status ===")
        latitude, longitude = get_location()
        
        # First entry of the day
        if int(userid) not in df['employee_id'].values:
            print("\n=== API Connection Status ===")
            print("✓ Connecting to attendance server...")
            
            print("\n=== Attendance Status ===")
            print("✓ Swipe type determined: IN-TIME")
            print(f"✓ Employee: {username}")
            print(f"✓ Time: {display_time}")
            
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{display_time},-')

            try:
                response = post_attendance_swipe(
                    name=username,
                    employee_id=userid,
                    timestamp=iso_timestamp,
                    latitude=latitude,
                    longitude=longitude,
                    swipe_type="IN-TIME")
                
                print("\n=== API Response ===")
                if isinstance(response, dict) and response.get("status") == "success":
                    print("✓ Connection established")
                    print("✓ Data synchronized successfully")
                else:
                    print("! Connection failed")
                    print("! Local backup saved")
            except Exception as api_error:
                print("\n=== API Error ===")
                print(f"! Connection error: {str(api_error)}")
                print("! Attendance saved locally")

        else:
            # Handle subsequent entries
            user_entries = df[df['employee_id'] == int(userid)]
            last_entry = user_entries.iloc[-1]

            if last_entry['Out-Time'] == '-':
                last_time = datetime.strptime(last_entry['In-Time'], "%H:%M:%S")
                current = datetime.strptime(display_time, "%H:%M:%S")
                time_diff = (current - last_time).total_seconds()

                if time_diff > 30:
                    print(f"\n=== Processing OUT-TIME for {username} ===")
                    print(f"✓ Check-out recorded at {display_time}")
                    
                    df.loc[df['employee_id'] == int(userid), 'Out-Time'] = display_time
                    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

                    try:
                        response = post_attendance_swipe(
                            name=username,
                            employee_id=userid,
                            timestamp=iso_timestamp,
                            latitude=latitude,
                            longitude=longitude,
                            swipe_type="OUT-TIME")
                        
                        if isinstance(response, dict) and response.get("status") == "success":
                            print("✓ API: Check-out synced successfully")
                        else:
                            print("! API: Failed to sync check-out")
                    except Exception as api_error:
                        print(f"! API Error: {str(api_error)}")
                else:
                    print(f"\n! Please wait {30 - int(time_diff)} seconds before checking out")
            else:
                print(f"\n! Already checked out for today")

    except Exception as e:
        print(f"\n! Error in attendance processing: {e}")
        return None, f"Error: {str(e)}"


def getallusers():
    userlist, names, employee_ids = os.listdir('static/faces'), [], []
    for user in userlist:
        name, employee_id = user.split('_')
        names.append(name)
        employee_ids.append(employee_id)
    return userlist, names, employee_ids, len(userlist)


@app.route('/')
def home():
    names, employee_ids, in_times, out_times, l = extract_attendance()
    return render_template('home.html',
                           names=names,
                           rolls=employee_ids,
                           in_times=in_times,
                           out_times=out_times,
                           l=l,
                           totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/start')
def start():
    names, employee_ids, in_times, out_times, l = extract_attendance()
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',
                               names=names,
                               rolls=employee_ids,
                               in_times=in_times,
                               out_times=out_times,
                               l=l,
                               totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess='No trained model. Add a face first.')

    # Try different camera indices
    cap = None
    for i in range(3):  # Try first 3 camera indices
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow backend
            if cap.isOpened():
                print(f"Successfully opened camera {i}")
                break
        except Exception as e:
            print(f"Error opening camera {i}: {str(e)}")
            if cap:
                cap.release()
    
    if not cap or not cap.isOpened():
        return render_template('home.html',
                               names=names,
                               rolls=employee_ids,
                               in_times=in_times,
                               out_times=out_times,
                               l=l,
                               totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess='Error: Could not access camera. Please check if it is connected and not in use by another application.')

    last_attendance_time = None
    last_prediction = None
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                time.sleep(0.1)  # Add small delay before retrying
                continue
                
            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = frame[y:y + h, x:x + w]
                prediction, confidence = identify_face(face)

                if prediction == "Unknown" or confidence < confidence_threshold:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown Person", (x, y - 15),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, "Access Denied", (x, y + h + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    name = prediction[0] if isinstance(prediction, list) else prediction
                    current_time = datetime.now()
                    
                    if (last_attendance_time is None or 
                        last_prediction != name or 
                        (current_time - last_attendance_time).total_seconds() > 30):
                        
                        add_attendance(name)
                        last_attendance_time = current_time
                        last_prediction = name
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 15),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            imgBackground[162:642, 55:695] = frame
            cv2.imshow('Attendance', imgBackground)
            if cv2.waitKey(1) == 27: 
                break
                
        except Exception as e:
            print(f"Error in camera loop: {str(e)}")
            time.sleep(0.1)  # Add small delay before retrying
            
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    
    names, rolls, in_times, out_times, l = extract_attendance()
    return render_template('home.html',
                           names=names,
                           rolls=rolls,
                           in_times=in_times,
                           out_times=out_times,
                           l=l,
                           totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)
    
    # Initialize counter for captured images
    captured_images = 0
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not cap.isOpened():
        return render_template('home.html',
                             names=names,
                             rolls=employee_ids,
                             in_times=in_times,
                             out_times=out_times,
                             l=l,
                             totalreg=totalreg(),
                             datetoday2=datetoday2,
                             mess='Error: Could not access camera. Please check if it is connected.')
    
    while captured_images < nimgs:  # Stop when we reach the limit
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                continue
                
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {captured_images}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
                
                # Only capture every 5th frame to get different angles
                if captured_images < nimgs:  # Double check we haven't exceeded the limit
                    face_img = frame[y:y + h, x:x + w]
                    cv2.imwrite(f'{userimagefolder}/{newusername}_{captured_images}.jpg', face_img)
                    captured_images += 1
                    print(f"Captured image {captured_images}/{nimgs}")
            
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:  # ESC key to exit
                break
                
        except Exception as e:
            print(f"Error in image capture: {str(e)}")
            continue
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Verify the number of images captured
    actual_images = len([f for f in os.listdir(userimagefolder) if f.endswith('.jpg')])
    print(f"Total images captured: {actual_images}")
    
    if actual_images < nimgs:
        print(f"Warning: Only captured {actual_images} images instead of {nimgs}")
    
    # Automatically retrain the model after adding a new user
    train_model()
    
    names, rolls, in_times, out_times, l = extract_attendance()
    return render_template('home.html',
                           names=names,
                           rolls=rolls,
                           in_times=in_times,
                           out_times=out_times,
                           l=l,
                           totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/retrain', methods=['GET'])
def retrain():
    train_model()
    names, rolls, in_times, out_times, l = extract_attendance(
    )  # Updated this line
    return render_template('home.html',
                           names=names,
                           rolls=rolls,
                           in_times=in_times,
                           out_times=out_times,
                           l=l,
                           totalreg=totalreg(),
                           datetoday2=datetoday2,
                           mess='Model retrained successfully.')


@app.route('/test_model', methods=['POST'])
def test_model():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    faces = extract_faces(img)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = img[y:y + h, x:x + w]
        prediction = identify_face(face, threshold=confidence_threshold)
        return f"Prediction: {prediction[0]}"
    return "No face detected."


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global confidence_threshold
    confidence_threshold = float(request.form['threshold'])
    return f"Confidence threshold set to {confidence_threshold}"


def compute_face_descriptor(img, face_rect):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = shape_predictor(gray, face_rect)
    descriptor = face_recognition_model.compute_face_descriptor(img, shape)
    return descriptor


def identify_face(facearray):
    try:
        if not os.path.exists('static/face_recognition_model.pkl'):
            print("Model file not found.")
            return ["Unknown", 0]

        model = joblib.load('static/face_recognition_model.pkl')

        # Preprocessing steps remain the same...
        facearray = cv2.cvtColor(facearray, cv2.COLOR_BGR2GRAY)
        facearray = cv2.equalizeHist(facearray)
        facearray = cv2.resize(facearray, (100, 100))
        facearray = cv2.GaussianBlur(facearray, (3, 3), 0)
        facearray = cv2.normalize(facearray, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        facearray = clahe.apply(facearray)
        facearray = facearray.ravel().reshape(1, -1)

        predictions = model.predict(facearray)
        confidence = model.predict_proba(facearray).max()
        print(f"Prediction: {predictions[0]}, Confidence: {confidence:.2f}")

        return [predictions[0], confidence]
    except Exception as e:
        print(f"Error in face identification: {e}")
        return ["Unknown", 0]


# Add after imports at the top
from geopy.geocoders import Nominatim
import socket


def get_location():
    try:
        # Hardcoded coordinates for Jeddah
        latitude = 21.543333
        longitude = 39.172779
        print("✓ Using default location: Jeddah")
        return latitude, longitude
    except Exception as e:
        print(f"Error getting location: {e}")
        return None, None


if __name__ == '__main__':
    app.run(debug=True)
