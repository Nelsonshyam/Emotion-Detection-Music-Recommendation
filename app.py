import cv2
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
import os
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"  # Folder to save uploaded files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Keras emotion detection model
MODEL_PATH = "My_model.h5"
MUSIC_CSV_PATH = "data/data_moods.csv"

print("Loading emotion detection model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Function to map emotions to moods
def map_emotion_to_mood(emotion):
    label_to_mood = {
        'angry': 'Energetic',
        'disgust': 'Energetic',
        'fear': 'Energetic',
        'surprise': 'Energetic',
        'happy': 'Happy',
        'sad': 'Sad',
        'neutral': 'Calm'
    }
    return label_to_mood.get(emotion, 'Calm')

# Predict emotion from a face image
def predict_emotion(model, face):
    model_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Preprocess the face image for the model
    face_resized = cv2.resize(face, (48, 48))  # Resize to 48x48
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_input = np.expand_dims(face_gray, axis=0)  # Add batch dimension
    face_input = np.expand_dims(face_input, axis=-1)  # Add channel dimension
    face_input = face_input / 255.0  # Normalize pixel values
    
    # Predict the emotion
    predictions = model.predict(face_input)
    predicted_label_index = np.argmax(predictions)
    predicted_emotion = model_labels[predicted_label_index]
    return predicted_emotion

# Capture user's reaction for specified duration
def capture_reaction(duration=10):
    print("Starting camera... Get ready to register your face with a reaction!")
    cap = cv2.VideoCapture(0)  # Use default camera (0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detected_face = None

    start_time = time.time()
    while time.time() - start_time <= duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect face in the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            detected_face = frame[y:y+h, x:x+w]  # Crop the detected face

        remaining_time = int(duration - (time.time() - start_time))
        cv2.putText(frame, f"Time Left: {remaining_time} sec", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture Reaction - Press 'q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return detected_face

# Route for combined functionality
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        songs_count = int(request.form.get('songs_count', 5))  # Default to 5 songs

        # Process image upload
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load the image and predict emotion
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({"error": "Invalid image file."})
            
            predicted_emotion = predict_emotion(model, image)
            mood = map_emotion_to_mood(predicted_emotion)
        else:
            # Process video capture
            face = capture_reaction(duration=10)
            if face is None:
                return jsonify({"error": "No face detected in video. Try again!"})
            predicted_emotion = predict_emotion(model, face)
            mood = map_emotion_to_mood(predicted_emotion)

        # Recommend songs
        music_df = pd.read_csv(MUSIC_CSV_PATH)
        recommended_songs = music_df[music_df['mood'] == mood].head(songs_count)
        songs_list = recommended_songs[['name', 'artist']].values.tolist()
        
        return render_template('index.html', 
                               emotion=predicted_emotion, 
                               mood=mood, 
                               songs_list=songs_list, 
                               songs_count=songs_count)

    # GET method: display the upload page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
