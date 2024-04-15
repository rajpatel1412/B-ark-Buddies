import cv2

# file to each video
attentive = "./videos/Attentive.mp4"
bang = "./videos/bang.mp4"
confusion = "./videos/confusion.mp4"
excited = "./videos/excited.mp4"
happy = "./videos/happy.mp4"
humanSad = "./videos/humanSad.mp4"
idle = "./videos/idle.mp4"
lieDown = "./videos/lieDown.mp4"
paw = "./videos/paw.mp4"
scared = "./videos/scared.mp4"
sitSleep = './videos/sitSleep.mp4'
standing = "./videos/standing.mp4"
turn = "./videos/turn.mp4"

# uses cv2 to play the video #https://www.geeksforgeeks.org/python-play-a-video-using-opencv/
def playVideo(video):
    
    cap = cv2.VideoCapture(video) 
    if (cap.isOpened()== False): 
        print("Error opening video file") 
    
    while(cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True: 
            cv2.imshow('Frame', frame)    
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else: 
            break

    cap.release() 
    ''' may need to uncomment this on different computers'''
    cv2.destroyAllWindows()
    return

# convers valence-arousal pairs to emotion/affect
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

# uses affect to queue videos/trigger dog reactions
def playGame(arousal, valence, confidence, command):
    if (confidence < 30):
        playVideo(confusion)
    else:
        if(command == ""):
            emotion = valenceArousalToEmotion(valence, arousal)
            if(emotion == "sad" or emotion == "disgust"):
                playVideo(humanSad)
            elif(emotion == "angry"):
                playVideo(scared)
            elif(emotion == "neutral"):
                playVideo(idle)
            elif(emotion == "low_neutral"):
                playVideo(lieDown)
            elif(emotion == "attentive"):
                playVideo(attentive)
            elif(emotion == "happy"):
                playVideo(happy)
            elif(emotion == "surprise"):
                playVideo(excited)
            else:
                playVideo(idle)
            
        else:
            if(command.lower() == "stand"):
                playVideo(standing)
            elif(command.lower() == "shake"):
                playVideo(paw)
            elif(command.lower() == "turn"):
                playVideo(turn)
            elif(command.lower() == "sit"):
                playVideo(sitSleep)
            elif(command.lower() == "bang"):
                playVideo(bang)

# runs the main game logic
def runGame(arousal, valence, confidence, command):
    print("(" + str(arousal) + ", " + str(valence) + ", " + str(confidence) + ", " + command + ")")
    playGame(arousal, valence, confidence, command)
