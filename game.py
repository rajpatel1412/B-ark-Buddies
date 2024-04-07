import cv2
import numpy as np
import time

attentive = "./videos/Attentive.mov"
bang = "./videos/bang.mov"
confusion = "./videos/confusion.mov"
excited = "./videos/excited.mov"
happy = "./videos/happy.mov"
humanSad = "./videos/humanSad.mov"
idle = "./videos/idle.mov"
lieDown = "./videos/lieDown.mov"
paw = "./videos/paw.mov"
scared = "./videos/scared.mov"
sitSleep = './videos/sitSleep.mov'
standing = "./videos/standin.mov"
turn = "./videos/turn.mov"

attentive_time = 6
bang_time = 10
confusion_time = 9
excited_time = 5
happy_time = 11
humanSad_time = 4
idle_time = 6
lieDown_time = 8
paw_time = 1
scared_time = 11
sitSleep_time = 6
standing_time = 6
turn_time = 3


def playVideo(video):
    
    cap = cv2.VideoCapture(video) 
    if (cap.isOpened()== False): 
        print("Error opening video file") 
    
    while(cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True: 
            cv2.imshow('Frame', frame) 
        
            if cv2.waitKey(20) & 0xFF == ord('q'): 
                break

        else: 
            break

    cap.release() 
    cv2.destroyAllWindows() 
    return

def valenceArousalToEmotion(valence, arousal):
    emotion = ""
    if(valence == 1 and arousal == 1):
        emotion = "sad"
    elif(valence == 1 and arousal == 2):
        emotion = "disgust"
    elif(valence == 1 and arousal == 3):
        emotion = "angry"
    elif(valence == 2 and arousal == 1):
        emotion = "low_neutral"
    elif(valence == 2 and arousal == 2):
        emotion = "neutral"
    elif(valence == 2 and arousal == 3):
        emotion = "attentive"
    elif(valence == 3 and arousal == 1):
        emotion = "happy"
    elif(valence == 3 and arousal == 2):
        emotion = "happy"
    elif(valence == 3 and arousal == 3):
        emotion = "surprise"
    return emotion

def playGame(arousal, valence, confidence, command):
    if (confidence < 0.3):
        playVideo(confusion)
        time.sleep(confusion_time)
    else:
        emotion = valenceArousalToEmotion(valence, arousal)
        if(emotion == "sad" or emotion == "disgust"):
            playVideo(humanSad)
            time.sleep(humanSad_time)
        elif(emotion == "angry"):
            playVideo(scared)
            time.sleep(scared_time)
        elif(emotion == "neutral"):
            playVideo(idle)
            time.sleep(idle_time)
        elif(emotion == "low_neutral"):
            playVideo(lieDown)
            time.sleep(lieDown_time)
        elif(emotion == "attentive"):
            playVideo(attentive)
            time.sleep(attentive_time)
        elif(emotion == "happy"):
            playVideo(happy)
            time.sleep(happy_time)
        elif(emotion == "surprise"):
            playVideo(excited)
            time.sleep(excited_time)
        # else:
        #     playVideo(idle)
        

        if(command.lower() == "stand"):
            playVideo(standing)
            time.sleep(standing_time)
        elif(command.lower() == "shake"):
            playVideo(paw)
            time.sleep(paw_time)
        elif(command.lower() == "turn"):
            playVideo(turn)
            time.sleep(turn_time)
        elif(command.lower() == "sit"):
            playVideo(sitSleep)
            time.sleep(sitSleep_time)
        elif(command.lower() == "bang"):
            playVideo(bang)
            time.sleep(bang_time)
