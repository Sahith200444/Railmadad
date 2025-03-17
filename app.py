from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
from flask_mysqldb import MySQL
import google.generativeai as genai
from flask_session import Session
from datetime import datetime, date, timedelta
import MySQLdb.cursors
import pytesseract
from PIL import Image
import os
import base64
import cv2
import whisper
import numpy as np
import tensorflow as tf  
import keras

app = Flask(__name__)

app.secret_key = 'abc3445'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'rail'


genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyCzXELRj3yG8q78oB5BYsgejaoADRAWqbY'))

image_model = tf.keras.models.load_model('d:/dataset/my_model.keras')

model1 = whisper.load_model("base") 
with open('d:/dataset/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f]

with open('d:/data/class_names.txt', 'r') as f:
    class_names1 = [line.strip() for line in f]


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)




video_model = tf.keras.models.load_model('d:/data/video_classification_model.keras')



def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    frame_interval = max(total_frames // 30, 1)
    
 
    for i in range(30):
        frame_pos = i * frame_interval
        if frame_pos >= total_frames: 
            frame = np.zeros((224, 224, 3), dtype=np.uint8) 
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)  
            else:
                frame = cv2.resize(frame, (224, 224))  

        frames.append(frame)

    cap.release()
    
   
    frames_array = np.array(frames) / 255.0  
    frames_array = np.expand_dims(frames_array, axis=0)

    
    predictions = video_model.predict(frames_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    
    return class_names1[class_idx], confidence



def classify_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = image_model.predict(img_array)
    class_idx = tf.argmax(predictions[0]).numpy()
    confidence = predictions[0][class_idx]

    class_name = class_names[class_idx]
    return class_name, confidence


def get_chatbot_response(message,details):
    try:
        input=f"You are a chatbot for the train service in india user details are.Details:{details},message:{message}"
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(input)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"


def gpt(type, desp, img, date):
    try:
        user_input = f"You are a complaint register for the train department using the text, image register the complaint to the respective department. Type: {type}, Description: {desp}, Image: {img}, Date: {date}"
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(user_input)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

mysql = MySQL(app)

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        user = request.form.get('user')
        pas = request.form.get('pas')
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('select username, mobile from login where username=%s and password=%s', (user, pas))
        us = cursor.fetchone()
        if us:
            session['mobile'] = us['mobile']
            message = "Logged in"
        else:
            message = "Wrong username/password"
    return render_template('dash.html', message=message)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'mobile' in request.form:
        username = request.form['username']
        password = request.form['password']
        mobile = request.form['mobile']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO login VALUES (%s, %s, %s)', (username, password, mobile))
        mysql.connection.commit()
        msg = 'You have successfully registered!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('dash.html', msg=msg)

@app.route('/dash.html', methods=['GET', 'POST'])
def dash():
    return render_template('dash.html')

@app.route('/complaint.html', methods=['GET', 'POST'])
def complaint():
    message = ""
    if request.method == 'POST':
        complaint_type = request.form.get('complaint-type')
        if complaint_type == "train":
            desp = request.form.get('tdesp')
            date = request.form.get('tdate')
            img = request.files.get('tphoto')
            name = request.form.get('tname')
        else:
            desp = request.form.get('sdesp')
            date = request.form.get('sdate')
            img = request.files.get('sphoto')
            name = request.form.get('sname')

        if complaint_type and desp and date and img:
            filename = img.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(img_path)
           
            extracted_text = pytesseract.image_to_string(Image.open(img_path))

            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                'INSERT INTO complaints (com_desp, com_img, com_type, com_date, com_mobile, name, extracted_text) '
                'VALUES (%s, %s, %s, %s, %s, %s, %s)',
                (desp, filename, complaint_type, date, session['mobile'], name, extracted_text)
            )
            mysql.connection.commit()
            cursor.close()

            message = "Complaint registered successfully!"
        else:
            message = "Please fill out all fields."

    return render_template('complaint.html', message=message)


@app.route('/vide.html', methods=['GET', 'POST'])
def complaint_using_video():
    message = ''
    if request.method == 'POST':
        complaint_type = request.form.get('complaint-type')
        if complaint_type == "train":
            desp = request.form.get('tdesp')
            date = request.form.get('tdate')
            video = request.files.get('tvideo')
            name = request.form.get('tname')
        else:
            desp = request.form.get('sdesp')
            date = request.form.get('sdate')
            video = request.files.get('svideo')
            name = request.form.get('sname')
        
        
        if video:
            v=video.filename
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(video_path)
            
            
            
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                'INSERT INTO complaintsvideo (com_desp, com_video, com_type, com_date, com_mobile, name) '
                'VALUES (%s, %s, %s, %s, %s, %s)',
                (desp, v, complaint_type, date, session['mobile'], name)
            )
            mysql.connection.commit()
            cursor.close()
            # Display a success message
            message = f"Complaint Registered"
        else:
            message = "Invalid detailes"

    return render_template('vide.html', message=message)




@app.route('/audio.html', methods=['GET', 'POST'])
def complaint_using_audio():
    msg = ''
    if request.method == 'POST':
        # Get form data
        complaint_type = request.form.get('complaint-type')
        if complaint_type == 'train':
            description = request.form.get('tdesp')
            date = request.form.get('tdate')
            name = request.form.get('tname')
        else:
            description = request.form.get('sdesp')
            date = request.form.get('sdate')
            name = request.form.get('sname')

        try:
            # Database operation
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                'INSERT INTO complaintsaudio (com_date, com_desp, com_mobile, com_type, name) VALUES (%s, %s, %s, %s, %s)',
                (date, description, session['mobile'], complaint_type, name)
            )
            mysql.connection.commit()
            msg = 'Complaint submitted successfully!'
        except Exception as e:
            mysql.connection.rollback()
            msg = 'Failed to submit complaint. Please try again.'
            print(f"Error: {e}")

    return render_template('audio.html', msg=msg)


@app.route('/generate_description_2', methods=['POST'])
def generate_description_2():
    if request.method == 'POST':
        data = request.get_json()
        audio_data = data.get('audio')

        if audio_data:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recorded_audio.wav')

            # Write audio bytes to file
            with open(audio_path, 'wb') as audio_file:
                audio_file.write(audio_bytes)

            # Transcribe audio using your model
            try:
                result = model1.transcribe(audio_path, task="translate")
                transcription = result['text']
                return jsonify({'transcription': transcription})
            except Exception as e:
                print(f"Error in transcription: {e}")
                return jsonify({'transcription': 'Error in transcription.'})

    return jsonify({'transcription': ''})

@app.route('/generate_description_1', methods=['POST'])
def generate_description_1():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

   
    class_name1, confidence = classify_video(video_path)

    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description = gpt(f"The image is of category {class_name1} in train. Describe this problem. First line: The problem. Next line: Describe the problem from the context of a user.", desp="", img=video_path, date=date)

    return jsonify({'description': description})




@app.route('/generate_description', methods=['POST'])
def generate_description():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(img_path)
    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    class_name, confidence = classify_image(img_path)
    description = gpt(f"The image is of category {class_name} in train. Describe this problem. First line: The problem. Next line: Describe the problem from the context of a user.", desp="", img=img_path, date=date)

    return jsonify({'description': description})



@app.route('/tracking.html',methods=['GET','POST'])
def tracking():
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('select * from complaints where com_mobile=%s',(session['mobile'],))
        complaints=cursor.fetchall()
        return render_template('tracking.html',complaints=complaints)


@app.route('/chatbot')
def chatbot():
    return render_template('c.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('select * from complaints where com_mobile=%s',(session['mobile'],))
    user_details = cursor.fetchall()
    bot_response = get_chatbot_response(user_message, user_details)
    return jsonify({'response': bot_response})

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)