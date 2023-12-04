
# http://127.0.0.1:5000

from __future__ import division, print_function
import sys
import os
import cv2
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st
from mtcnn import MTCNN
import keyboard
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")


# ... (your existing code)

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    i = 0

    GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
    model = tf.keras.models.load_model('kagglemodel.h5')
    detector = MTCNN()
    
    output = []
    cap = cv2.VideoCapture(0)
    
    while (i <= 30):
        ret, img = cap.read()
        faces = detector.detect_faces(img)

        for face in faces:
            x, y, w, h = face['box']
            face_img = img[y:y + h, x:x + w]

            resized = cv2.resize(face_img, (48, 48))
            reshaped = resized.reshape(1, 48, 48, 3) / 255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)

            cv2.rectangle(img, (x, y), (x + w, y + h), GR_dict[1], 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), GR_dict[1], -1)
            cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        i = i + 1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        
        # Add this section to check for Ctrl + Q
        if keyboard.is_pressed('ctrl') and keyboard.is_pressed('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    
    
    
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("buttons.html", final_output=final_output1)

    # nickname = request.form['nickname']
    # chat_output = ask_music_questions(nickname)
    # return render_template("buttons.html", final_output=final_output1, chat_output=chat_output, nickname=nickname)


# import random
# from textblob import TextBlob

# # Define a function for asking questions about music
# def ask_music_questions(nickname):
#     # Greeting selection
#     greetings = [
#         f'How are you today {nickname}?',
#         f'Howdy {nickname}! How are you feeling today?',
#         f'What\'s up {nickname}?',
#         f'Greetings {nickname}, are you well?',
#         f'How are things going {nickname}?'
#     ]
#     print(random.choice(greetings))

#     ans = input()
#     blob = TextBlob(ans)

#     if blob.polarity > 0:
#         print('Glad you are doing well!😊')
#     else:
#         print('Sorry to hear that! 😔')

#     # Ask opinions about music-related topics
#     music_topics = [
#         'pop music',
#         'rock music',
#         'classical music',
#         'hip-hop music',
#         'jazz music',
#         'rap music'
#     ]

#     questions = [
#         'What is your take on ',
#         'What do you think about ',
#         'How do you feel about ',
#         'What do you reckon about ',
#         'I would like your opinion on '
#     ]

#     for i in range(0, random.randint(3, 4)):
#         question = random.choice(questions)
#         questions.remove(question)
#         topic = random.choice(music_topics)
#         music_topics.remove(topic)
#         print(question + topic+'?')
#         ans = input()
#         blob = TextBlob(ans)

#         if blob.polarity > 0.5:
#             print(f"OMG, you really love {topic}!")
#         elif blob.polarity > 0.1:
#             print(f"Well, you clearly like {topic}.")
#         elif blob.polarity < -0.5:
#             print(f"Uff, you totally hate {topic}.")
#         elif blob.polarity < -0.1:
#             print(f"So you don't like {topic}.")
#         else:
#             print(f"That is a very neutral view on {topic}.")

#         if blob.subjectivity > 0.6:
#             print('And you are so biased!')
#         elif blob.subjectivity > 0.3:
#             print('And you are a bit biased!')
#         else:
#             print('And you are quite objective, huh!')

#     # Random goodbye
#     goodbyes = [
#         f'It was good talking to you, {nickname}. I gotta go now!',
#         "OK, I'm bored, I'm going to watch Netflix.",
#         "Bye Bye American Pie, I'm out!",
#         f'Catch ya later, {nickname}!'
#     ]
#     print(random.choice(goodbyes))

# # Name and nickname conversation
# print("Hello human, what's your name?!🤔")
# name = input()
# print('Do you have a nickname?! [y/n] 🙃')
# ans = input()

# if 'y' in ans.lower():
#     print("What's your nickname?!😍")
#     nickname = input()
#     print(f'Good to meet you, {nickname}! 😁')
# else:
#     name_list = ['killua', 'gon', 'naruto', 'xoxo', 'kimchi', 'mother-coconuts', 'phineas',
#                  'ferb', 'tennison', 'gwen', 'prarthana', 'meow', 'tuple', 'silly goose', 'babe', 'rose', 'tupperware', 'dude']
#     nickname = random.choice(name_list)
#     print(f'I will call you {nickname}! 😜')


# Call the function to ask music-related questions
# Assuming you have the user's nickname available in the request
    
# ****************************************************************************************
# @app.route('/camera', methods = ['GET', 'POST'])
# def camera():
#     i=0

#     GR_dict={0:(0,255,0),1:(0,0,255)}
#     model = tf.keras.models.load_model('kagglemodel.h5')
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     output=[]
#     cap = cv2.VideoCapture(0)
#     while (i<=30):
#         ret, img = cap.read()
#         faces = face_cascade.detectMultiScale(img,1.05,5)

#         for x,y,w,h in faces:

#             face_img = img[y:y+h,x:x+w] 

#             resized = cv2.resize(face_img,(48,48))
#             reshaped=resized.reshape(1, 48,48,3)/255
#             predictions = model.predict(reshaped)

#             max_index = np.argmax(predictions[0])

#             # emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
#             emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad')
#             predicted_emotion = emotions[max_index]
#             output.append(predicted_emotion)
            
            
            
#             cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
#             cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
#             cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
#         i = i+1

#         cv2.imshow('LIVE', img)
#         key = cv2.waitKey(1)
#         if key == 27: 
#             cap.release()
#             cv2.destroyAllWindows()
#             break
#     print(output)
#     cap.release()
#     cv2.destroyAllWindows()
#     final_output1 = st.mode(output)
#     return render_template("buttons.html",final_output=final_output1)


@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/surprise', methods = ['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/angry', methods = ['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods = ['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods = ['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")

@app.route('/movies/happy', methods = ['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods = ['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods = ['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")

@app.route('/songs/surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songsSad.html")

@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")
    



# -----------------------------------------------------------------
# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Define some sample patterns and responses
# patterns_responses = {
#     "hello": "Hi there!",
#     "how are you": "I'm good, thank you!",
#     "bye": "Goodbye!",
#     "default": "I'm not sure how to respond to that."
# }

# def get_response(user_input):
#     for pattern, response in patterns_responses.items():
#         if pattern in user_input:
#             return response
#     return patterns_responses["default"]

# @app.route('/')
# def home():
#     return render_template('chat.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.form['user_message']
#     bot_response = get_response(user_message)
#     return bot_response



if __name__ == "__main__":
    app.run(debug=True)





