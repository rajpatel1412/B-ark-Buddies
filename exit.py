# import asyncio
# import keyboard

# dontExit = True

# def setExit(bool):
#     global dontExit 
#     dontExit = bool
    
# def getExit():
#     return dontExit

# # Function to get input asynchronously
# async def async_input(prompt):
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, input, prompt)


# def checkExit():
#     # dontExit = True
#     # key = input("Exit?")
#     # if(key == "\n"):
#     #     setExit(False)
#     # print(dontExit)
#     while(True):
#         # if(keyboard.is_pressed('esc')):
#         #     setExit(False)
#         #     break

#         if(asyncio.run(async_input("Exit?")) == "n"):
#             setExit(False)
#             break

#     return

# checkExit()


import game
import threading
import time
import random
import datetime

def humanTime():
    return (datetime.datetime.fromtimestamp(time.time())).strftime('%Y-%m-%d %H:%M:%S')

while(True):
    time.sleep(5)
    print("end of input: " + humanTime())
    valence = random.randint(1, 3)
    arousal = random.randint(1, 3)
    confidence = random.uniform(0, 1) * 100
    print("(" + str(arousal) + ", " + str(valence) + ", " + str(confidence) + ")")
    playGameThread = threading.Thread(target=game.playGame, args=(arousal, valence, confidence, ""))
    playGameThread.start()
    game.setVideoPlaying(True)
    while(game.getVideoPlaying() == True):
        time.sleep(1)
    print("end of video: " + humanTime())
    playGameThread.join()