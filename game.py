# import cv2
# # import exit
# import threading
# import numpy as np
# import time
# import keyboard
# import sys, signal

# import random

# attentive = "./videos/Attentive.mov"
# bang = "./videos/bang.mov"
# confusion = "./videos/confusion.mov"
# excited = "./videos/excited.mov"
# happy = "./videos/happy.mov"
# humanSad = "./videos/humanSad.mov"
# idle = "./videos/idle.mov"
# lieDown = "./videos/lieDown.mov"
# paw = "./videos/paw.mov"
# scared = "./videos/scared.mov"
# sitSleep = './videos/sitSleep.mov'
# standing = "./videos/standing.mov"
# turn = "./videos/turn.mov"

# # vid_names = [attentive, bang, confusion, excited, 
# #              happy, humanSad, idle, lieDown, paw,
# #              scared, sitSleep, standing, turn]

# # attentive_video = []
# # bang_video = []
# # confusion_video = []
# # excited_video = []
# # happy_video = []
# # humanSad_video = []
# # idle_video = []
# # lieDown_video = []
# # paw_video = []
# # scared_video = []
# # sitSleep_video = []
# # standing_video = []
# # turn_video = []

# # videos = [attentive_video, bang_video, confusion_video,
# #           excited_video, happy_video, humanSad_video, 
# #           idle_video, lieDown_video, paw_video, scared_video,
# #           sitSleep_video, standing_video, turn_video]

# # def gameInit():
# #     for i, video in enumerate(videos):
# #         cap = cv2.VideoCapture(vid_names[i]) 
# #         if (cap.isOpened()== False): 
# #             print("Error opening video file") 
        
# #         while(cap.isOpened()): 
# #             ret, frame = cap.read() 
# #             if ret == True: 
# #                 videos[i].append(frame) 
# #             else: 
# #                 break

# #         cap.release() 

# # isPlaying = False



# def playVideo(video):
# # def playVideo(video, vid_name):
    
#     cap = cv2.VideoCapture(video) 
#     if (cap.isOpened()== False): 
#         print("Error opening video file") 
    
#     while(cap.isOpened()): 
#         ret, frame = cap.read() 
#         if ret == True: 
#             cv2.imshow('Frame', frame) 
        
#             if cv2.waitKey(1) & 0xFF == ord('q'): 
#                 break

#         else: 
#             break

#     cap.release() 
#     # setVideoPlaying(False)

#     ''' may need to uncomment this on different computers'''
#     # cv2.destroyAllWindows()

#     # # cap = cv2.VideoCapture(vid_name)
#     # for frame in video:
#     #     # cap = cv2.VideoCapture(video) 
#     #     # if (cap.isOpened()== False): 
#     #     #     print("Error opening video file") 
        
#     #     # while(cap.isOpened()): 
#     #     cv2.imshow('Playback', frame)
#     #     if cv2.waitKey(1) & 0xFF == ord('q'): 
#     #         break

#     # # cap.release()
     
#     return

# def valenceArousalToEmotion(valence, arousal):
#     emotion = ""
#     if(valence == 1 and arousal == 1):
#         emotion = "sad"
#     elif(valence == 1 and arousal == 2):
#         emotion = "disgust"
#     elif(valence == 1 and arousal == 3):
#         emotion = "angry"
#     elif(valence == 2 and arousal == 1):
#         emotion = "low_neutral"
#     elif(valence == 2 and arousal == 2):
#         emotion = "neutral"
#     elif(valence == 2 and arousal == 3):
#         emotion = "attentive"
#     elif(valence == 3 and arousal == 1):
#         emotion = "happy"
#     elif(valence == 3 and arousal == 2):
#         emotion = "happy"
#     elif(valence == 3 and arousal == 3):
#         emotion = "surprise"
#     return emotion

# def playGame(arousal, valence, confidence, command):
#     # if (confidence < 0.3):
#     if (confidence < 30):
#         playVideo(confusion)
#         # time.sleep(confusion_time)
#     else:
#         if(command == ""):
#             emotion = valenceArousalToEmotion(valence, arousal)
#             if(emotion == "sad" or emotion == "disgust"):
#                 playVideo(humanSad)
#                 # time.sleep(humanSad_time)
#             elif(emotion == "angry"):
#                 playVideo(scared)
#                 # time.sleep(scared_time)
#             elif(emotion == "neutral"):
#                 playVideo(idle)
#                 # time.sleep(idle_time)
#             elif(emotion == "low_neutral"):
#                 playVideo(lieDown)
#                 # time.sleep(lieDown_time)
#             elif(emotion == "attentive"):
#                 playVideo(attentive)
#                 # time.sleep(attentive_time)
#             elif(emotion == "happy"):
#                 playVideo(happy)
#                 # time.sleep(happy_time)
#             elif(emotion == "surprise"):
#                 playVideo(excited)
#                 # time.sleep(excited_time)
#             # else:
#             #     playVideo(idle)
            
#         else:
#             if(command.lower() == "stand"):
#                 playVideo(standing)
#                 # time.sleep(standing_time)
#             elif(command.lower() == "shake"):
#                 playVideo(paw)
#                 # time.sleep(paw_time)
#             elif(command.lower() == "turn"):
#                 playVideo(turn)
#                 # time.sleep(turn_time)
#             elif(command.lower() == "sit"):
#                 playVideo(sitSleep)
#                 # time.sleep(sitSleep_time)
#             elif(command.lower() == "bang"):
#                 playVideo(bang)
#                 # time.sleep(bang_time)

# def runGame(arousal, valence, confidence, command):
#     print("(" + str(arousal) + ", " + str(valence) + ", " + str(confidence) + ", " + command + ")")
#     playGame(arousal, valence, confidence, command)

# def handleInput():
#     global exitFlag
#     while(True):
#         # user_input = await asyncio.to_thread(input, "Enter 'Return' to stop: ")
#         userInput = input("Enter 'q' to stop: \n")
#         if userInput.lower() == 'q':
#             print("Exiting!\n")
#             exitFlag = True
#             break

# # def setVideoPlaying(playing):
# #     global isPlaying 
# #     isPlaying = playing
# #     return

# # def getVideoPlaying():
# #     return isPlaying




# ######################### TESTING ##########################

# import datetime
# def humanTime():
#     return (datetime.datetime.fromtimestamp(time.time())).strftime('%Y-%m-%d %H:%M:%S')


# import asyncio
# exitFlag = False

# # async def runGame():
# def runTestGame():
#     while(exitFlag == False):
#         time.sleep(5)
#         print("end of input: " + humanTime())
#         valence = random.randint(1, 3)
#         arousal = random.randint(1, 3)
#         confidence = random.uniform(0, 1) * 100
#         command = "stand"
#         print("(" + str(arousal) + ", " + str(valence) + ", " + str(confidence) + ", " + command + ")")
#         playGame(arousal, valence, confidence, command)
#         print("end of video: " + humanTime())
#         # await asyncio.sleep(1)

# # async def handleInput():
# def handle_Input():
#     global exitFlag
#     while(True):
#         # user_input = await asyncio.to_thread(input, "Enter 'Return' to stop: ")
#         userInput = input("Enter 'q' to stop: \n")
#         if userInput.lower() == 'q':
#             print("Exiting!\n")
#             exitFlag = True
#             break

# # async def tester():
# def tester():
#     # exitThread = threading.Thread(target=exit.checkExit(), args=None)
#     # exitThread.start()
#     # while(exit.getExit()):

#     # gameInit()

#     # try:
#         # while(True):
#         #     time.sleep(5)
#         #     print("end of input: " + humanTime())
#         #     valence = random.randint(1, 3)
#         #     arousal = random.randint(1, 3)
#         #     confidence = random.uniform(0, 1) * 100
#         #     command = "stand"
#         #     print("(" + str(arousal) + ", " + str(valence) + ", " + str(confidence) + ", " + command + ")")
#         #     playGame(arousal, valence, confidence, command)
#         #     print("end of video: " + humanTime())
#     # except input() == "n":
#     #     print('exit')

#     # exitThread.join()

#     # for frame in attentive_video:

#     # Create tasks for the loop and input handling
#     # loop_task = asyncio.create_task(runGame())
#     # input_task = asyncio.create_task(handleInput())

#     # # Wait for either task to finish
#     # await asyncio.wait([loop_task, input_task], return_when=asyncio.FIRST_COMPLETED)

#     # # Cancel the remaining task
#     # if not loop_task.done():
#     #     loop_task.cancel()
#     # if not input_task.done():
#     #     input_task.cancel()

#     inputThread = threading.Thread(target=handle_Input)
#     inputThread.start()

#     runTestGame()

#     inputThread.join()

# # asyncio.run(tester())

# # tester()

import cv2
# import exit
import threading
import numpy as np
import time
import keyboard
import sys, signal

import random

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
standing = "./videos/standing.mov"
turn = "./videos/turn.mov"

# vid_names = [attentive, bang, confusion, excited, 
#              happy, humanSad, idle, lieDown, paw,
#              scared, sitSleep, standing, turn]

# attentive_video = []
# bang_video = []
# confusion_video = []
# excited_video = []
# happy_video = []
# humanSad_video = []
# idle_video = []
# lieDown_video = []
# paw_video = []
# scared_video = []
# sitSleep_video = []
# standing_video = []
# turn_video = []

# videos = [attentive_video, bang_video, confusion_video,
#           excited_video, happy_video, humanSad_video, 
#           idle_video, lieDown_video, paw_video, scared_video,
#           sitSleep_video, standing_video, turn_video]

# def gameInit():
#     for i, video in enumerate(videos):
#         cap = cv2.VideoCapture(vid_names[i]) 
#         if (cap.isOpened()== False): 
#             print("Error opening video file") 
        
#         while(cap.isOpened()): 
#             ret, frame = cap.read() 
#             if ret == True: 
#                 videos[i].append(frame) 
#             else: 
#                 break

#         cap.release() 

# isPlaying = False



def playVideo(video):
# def playVideo(video, vid_name):
    
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
    # setVideoPlaying(False)

    ''' may need to uncomment this on different computers'''
    # cv2.destroyAllWindows()

    # # cap = cv2.VideoCapture(vid_name)
    # for frame in video:
    #     # cap = cv2.VideoCapture(video) 
    #     # if (cap.isOpened()== False): 
    #     #     print("Error opening video file") 
        
    #     # while(cap.isOpened()): 
    #     cv2.imshow('Playback', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'): 
    #         break

    # # cap.release()
     
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
    # if (confidence < 0.3):
    if (confidence < 30):
        playVideo(confusion)
        # time.sleep(confusion_time)
    else:
        if(command == ""):
            emotion = valenceArousalToEmotion(valence, arousal)
            if(emotion == "sad" or emotion == "disgust"):
                playVideo(humanSad)
                # time.sleep(humanSad_time)
            elif(emotion == "angry"):
                playVideo(scared)
                # time.sleep(scared_time)
            elif(emotion == "neutral"):
                playVideo(idle)
                # time.sleep(idle_time)
            elif(emotion == "low_neutral"):
                playVideo(lieDown)
                # time.sleep(lieDown_time)
            elif(emotion == "attentive"):
                playVideo(attentive)
                # time.sleep(attentive_time)
            elif(emotion == "happy"):
                playVideo(happy)
                # time.sleep(happy_time)
            elif(emotion == "surprise"):
                playVideo(excited)
                # time.sleep(excited_time)
            # else:
            #     playVideo(idle)
            
        else:
            if(command.lower() == "stand"):
                playVideo(standing)
                # time.sleep(standing_time)
            elif(command.lower() == "shake"):
                playVideo(paw)
                # time.sleep(paw_time)
            elif(command.lower() == "turn"):
                playVideo(turn)
                # time.sleep(turn_time)
            elif(command.lower() == "sit"):
                playVideo(sitSleep)
                # time.sleep(sitSleep_time)
            elif(command.lower() == "bang"):
                playVideo(bang)
                # time.sleep(bang_time)

def runGame(arousal, valence, confidence, command):
    print("(" + str(arousal) + ", " + str(valence) + ", " + str(confidence) + ", " + command + ")")
    playGame(arousal, valence, confidence, command)

def handleInput():
    global exitFlag
    while(True):
        # user_input = await asyncio.to_thread(input, "Enter 'Return' to stop: ")
        userInput = input("Enter 'q' to stop: \n")
        if userInput.lower() == 'q':
            print("Exiting!\n")
            exitFlag = True
            break

# def setVideoPlaying(playing):
#     global isPlaying 
#     isPlaying = playing
#     return

# def getVideoPlaying():
#     return isPlaying




######################### TESTING ##########################

import datetime
def humanTime():
    return (datetime.datetime.fromtimestamp(time.time())).strftime('%Y-%m-%d %H:%M:%S')


import asyncio
exitFlag = False

# async def runGame():
def runTestGame():
    while(exitFlag == False):
        time.sleep(5)
        print("end of input: " + humanTime())
        valence = random.randint(1, 3)
        arousal = random.randint(1, 3)
        confidence = random.uniform(0, 1) * 100
        command = "stand"
        print("(" + str(arousal) + ", " + str(valence) + ", " + str(confidence) + ", " + command + ")")
        playGame(arousal, valence, confidence, command)
        print("end of video: " + humanTime())
        # await asyncio.sleep(1)

# async def handleInput():
def handle_Input():
    global exitFlag
    while(True):
        # user_input = await asyncio.to_thread(input, "Enter 'Return' to stop: ")
        userInput = input("Enter 'q' to stop: \n")
        if userInput.lower() == 'q':
            print("Exiting!\n")
            exitFlag = True
            break

# async def tester():
def tester():
    # exitThread = threading.Thread(target=exit.checkExit(), args=None)
    # exitThread.start()
    # while(exit.getExit()):

    # gameInit()

    # try:
        # while(True):
        #     time.sleep(5)
        #     print("end of input: " + humanTime())
        #     valence = random.randint(1, 3)
        #     arousal = random.randint(1, 3)
        #     confidence = random.uniform(0, 1) * 100
        #     command = "stand"
        #     print("(" + str(arousal) + ", " + str(valence) + ", " + str(confidence) + ", " + command + ")")
        #     playGame(arousal, valence, confidence, command)
        #     print("end of video: " + humanTime())
    # except input() == "n":
    #     print('exit')

    # exitThread.join()

    # for frame in attentive_video:

    # Create tasks for the loop and input handling
    # loop_task = asyncio.create_task(runGame())
    # input_task = asyncio.create_task(handleInput())

    # # Wait for either task to finish
    # await asyncio.wait([loop_task, input_task], return_when=asyncio.FIRST_COMPLETED)

    # # Cancel the remaining task
    # if not loop_task.done():
    #     loop_task.cancel()
    # if not input_task.done():
    #     input_task.cancel()

    inputThread = threading.Thread(target=handle_Input)
    inputThread.start()

    runTestGame()

    inputThread.join()

# asyncio.run(tester())

# tester()