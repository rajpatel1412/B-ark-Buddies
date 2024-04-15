# # ---- Sources/Documentation used:
# # https://pypi.org/project/opencv-python/
# # https://www.datacamp.com/tutorial/face-detection-python-opencv


# import cv2
# import time
# import shutil
# import os
# from tensorflow import keras
# from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
# from keras.utils import to_categorical
# from keras import preprocessing
# from PIL import Image

# import matplotlib.pyplot as plt
# from matplotlib.image import imread
# import numpy

# # ---- Helper Functions

# def loadFaceModel():
#     # model from: https://github.com/akash720/Facial-expression-recognition/blob/master/README.md
#     # Note: had to train it and then save it with a newer version of keras for some reason in google collab
#     pretrainedModel = load_model("my_model.h5")
#     return pretrainedModel


# def createFaceImages(videoFrame, facesFromFrame, facesFolder):
#     for face in facesFromFrame:
#         # Unpack data for each face
#         xframeCoordinate, yframeCoordinate, width, height = face

#         # Extract facial region from the frame
#         extractedFace = videoFrame[yframeCoordinate:yframeCoordinate+height, xframeCoordinate:xframeCoordinate+width]

#         # Save the extracted face image with a timestamp prefix
#         timestampForImageName = int(time.time())
#         fileName = facesFolder + str(timestampForImageName) + "_face.jpg"
#         cv2.imwrite(fileName, extractedFace)

# def extractFacesFromFrame(videoFrame, faceDetectionModel):
#     # Convert frame to grayscale for more efficent face detection
#     grayscaleVideoFrame = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)

#     # Scale down the size of the frame from the video by 20% to icrease performance when a face
#     # takes up a large portion of the frame
#     frameScale = 1.2

#     # Choose 5 as the window size that will slide over the frame to detect images
#     minNumNeighbours = 5

#     # Ignore any faces that are smaller than 30x30
#     minFaceSize = (30, 30)

#     # Use detectMultiScale to identify faces in frame
#     facesFromFrame = faceDetectionModel.detectMultiScale(grayscaleVideoFrame, scaleFactor=frameScale, minNeighbors=minNumNeighbours, minSize=minFaceSize)

#     return facesFromFrame

# def translateNumberToCategory(number):
#       # From the fer2013 data set:
#     # (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
#     if number == 0:
#         categoryString = 'angry'
#     elif number == 1:
#         categoryString = 'disgust'
#     elif number == 2:
#         categoryString = 'fear'
#     elif number == 3:
#         categoryString = 'happy'
#     elif number == 4:
#         categoryString = 'sad'
#     elif number == 5:
#         categoryString = 'surprise'
#     elif number == 6:
#         categoryString = 'neutral'
#     else:
#         categoryString = 'unknown'
#     return categoryString

# # This function replicates the process of processing images from the 
# # pre-trained model (https://github.com/akash720/Facial-expression-recognition/blob/master/README.md)
# # so that the image input is as similarly formated as possible to training data
# def processImage(imageFilePath):
#     imageToCheck = Image.open(imageFilePath)
    
#     # Resize and re-sample image, the model originally used the ANTIALIAS resampling method
#     # but this was deprecated in the current version of pillow
#     resizedImage = imageToCheck.resize((48,48), Image.LANCZOS)
#     resizedImage = numpy.array(resizedImage)
#     grayscaleImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

#     # Scale pixel values to be in the range [0,1]
#     scaledImage = grayscaleImage/255

#     # Reshaping image as done in the pre-trained model pre-processing
#     reshapeImage = scaledImage.reshape(-1, 48, 48, 1)
#     return reshapeImage

# # for fusion
# def packageInputsForFusion(expressionValue, confidence):
#     expressionLabel = translateNumberToCategory(expressionValue)
#     arousel = 0
#     valence = 0

#     # 1 is low/negative and 3 is high/positive
#     if expressionLabel == 'angry':
#         arousel = 3
#         valence = 1
#     elif expressionLabel == 'disgust':
#         arousel = 2
#         valence = 1
#     elif expressionLabel == 'fear':
#         arousel = 3
#         valence = 1
#     elif expressionLabel == 'happy':
#         arousel = 2
#         valence = 3
#     elif expressionLabel == 'sad':
#         arousel = 1
#         valence = 1
#     elif expressionLabel == 'surprise':
#         arousel = 3
#         valence = 3
#     elif expressionLabel == 'neutral':
#         arousel = 2
#         valence = 2

#     packagedValues = ('face', arousel, valence, confidence)
#     return packagedValues

# def selectPrediction(predictionList):
#     mostCommonValue = 0
#     mostCommonCount = 0

#     # find the most common value and store it's index
#     for value in predictionList:
#         count = predictionList.count(value)
#         if count > mostCommonCount:
#             mostCommonValue = value
#             mostCommonCount = count
    
#     return mostCommonValue


# def findLastIndexOfMostCommonPrediction(predictionList, mostCommonValue):
#     lastIndexOfCommonValue = 0
#     for index in range(len(predictionList)):
#         if predictionList[index] == mostCommonValue:
#             lastIndexOfCommonValue = index
#     return lastIndexOfCommonValue

# # ---- Main Code
# def runFacialExpressionRecognition(pretrainedModel, results):
#     # Load the face detection model from cv2
#     # used the haarcascade frontal face model since it is relatively fast and a common pre-trained model for
#     # facial detection
#     faceDetectionModel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Initialize video capture
#     defaultWebcamIndex = 0
#     videoCapture = cv2.VideoCapture(defaultWebcamIndex)

#     # Define a variable to keep track of the faces extracted from video frames
#     faceCount = 0

#     # Create a folder to store the images of the faces if it does not currently exist
#     facesFolder = "faces/"
#     if os.path.exists(facesFolder):
#         shutil.rmtree(facesFolder)
#     os.makedirs(facesFolder)

#     # take 5 seconds of video
#     start_time = time.time()
#     while time.time() - start_time < 5:
#         # Get frame to process from video camera
#         status, videoFrame = videoCapture.read()
        
#         # Ensure frame is read sucessfully
#         if status is False:
#             break

#         # extract a list of faces from each frame
#         facesFromFrame = extractFacesFromFrame(videoFrame, faceDetectionModel)

#         # Create jpgs from list of faces
#         createFaceImages(videoFrame, facesFromFrame, facesFolder)

#     # Release the capture
#     videoCapture.release()

#     #Extract pixel values for each image in the faces folder
#     facesFolder = "faces/"
#     imageList = os.listdir(facesFolder)

#     predictionLabels = []
#     confidence = []

#     for imageFile in imageList:
#         # process the image
#         imageFilePath = facesFolder + imageFile
#         processedImage = processImage(imageFilePath)

#         # use model to predict
#         probabilityDistribution = pretrainedModel.predict(processedImage)
#         category = numpy.argmax(probabilityDistribution)
#         predictionLabels.append(category)

#         # calculate confidence from probability
#         confidencePercentage = probabilityDistribution[0][category] * 100
#         confidence.append(confidencePercentage)
    
#     selectedPrediction = selectPrediction(predictionLabels)
#     # print(selectedPrediction)
#     selectedPredictionIndex = findLastIndexOfMostCommonPrediction(predictionLabels, selectedPrediction)
#     selectedPredictionConfidence = confidence[selectedPredictionIndex]

#     results.append(packageInputsForFusion(selectedPrediction, selectedPredictionConfidence))
#     # print(results)



# # def main():
# #     print("audio audio")
# #     videoModel = loadFaceModel()
# #     results = []
# #     runFacialExpressionRecognition(videoModel, results)
# #     print("done")


# # if __name__ == "__main__":
# #     main()

# ---- Sources/Documentation used:
# https://pypi.org/project/opencv-python/
# https://www.datacamp.com/tutorial/face-detection-python-opencv


import cv2
import time
import shutil
import os
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras import preprocessing
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy

# ---- Helper Functions

def loadFaceModel():
    # model from: https://github.com/akash720/Facial-expression-recognition/blob/master/README.md
    # Note: had to train it and then save it with a newer version of keras for some reason in google collab
    pretrainedModel = load_model("my_model.h5")
    return pretrainedModel


def createFaceImages(videoFrame, facesFromFrame, facesFolder):
    for face in facesFromFrame:
        # Unpack data for each face
        xframeCoordinate, yframeCoordinate, width, height = face

        # Extract facial region from the frame
        extractedFace = videoFrame[yframeCoordinate:yframeCoordinate+height, xframeCoordinate:xframeCoordinate+width]

        # Save the extracted face image with a timestamp prefix
        timestampForImageName = int(time.time())
        fileName = facesFolder + str(timestampForImageName) + "_face.jpg"
        cv2.imwrite(fileName, extractedFace)

def extractFacesFromFrame(videoFrame, faceDetectionModel):
    # Convert frame to grayscale for more efficent face detection
    grayscaleVideoFrame = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)

    # Scale down the size of the frame from the video by 20% to icrease performance when a face
    # takes up a large portion of the frame
    frameScale = 1.2

    # Choose 5 as the window size that will slide over the frame to detect images
    minNumNeighbours = 5

    # Ignore any faces that are smaller than 30x30
    minFaceSize = (30, 30)

    # Use detectMultiScale to identify faces in frame
    facesFromFrame = faceDetectionModel.detectMultiScale(grayscaleVideoFrame, scaleFactor=frameScale, minNeighbors=minNumNeighbours, minSize=minFaceSize)

    return facesFromFrame

def translateNumberToCategory(number):
      # From the fer2013 data set:
    # (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    if number == 0:
        categoryString = 'angry'
    elif number == 1:
        categoryString = 'disgust'
    elif number == 2:
        categoryString = 'fear'
    elif number == 3:
        categoryString = 'happy'
    elif number == 4:
        categoryString = 'sad'
    elif number == 5:
        categoryString = 'surprise'
    elif number == 6:
        categoryString = 'neutral'
    else:
        categoryString = 'unknown'
    return categoryString

# This function replicates the process of processing images from the 
# pre-trained model (https://github.com/akash720/Facial-expression-recognition/blob/master/README.md)
# so that the image input is as similarly formated as possible to training data
def processImage(imageFilePath):
    imageToCheck = Image.open(imageFilePath)
    
    # Resize and re-sample image, the model originally used the ANTIALIAS resampling method
    # but this was deprecated in the current version of pillow
    resizedImage = imageToCheck.resize((48,48), Image.LANCZOS)
    resizedImage = numpy.array(resizedImage)
    grayscaleImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

    # Scale pixel values to be in the range [0,1]
    scaledImage = grayscaleImage/255

    # Reshaping image as done in the pre-trained model pre-processing
    reshapeImage = scaledImage.reshape(-1, 48, 48, 1)
    return reshapeImage

# for fusion
def packageInputsForFusion(expressionValue, confidence):
    expressionLabel = translateNumberToCategory(expressionValue)
    arousel = 0
    valence = 0

    # 1 is low/negative and 3 is high/positive
    if expressionLabel == 'angry':
        arousel = 3
        valence = 1
    elif expressionLabel == 'disgust':
        arousel = 2
        valence = 1
    elif expressionLabel == 'fear':
        arousel = 3
        valence = 1
    elif expressionLabel == 'happy':
        arousel = 2
        valence = 3
    elif expressionLabel == 'sad':
        arousel = 1
        valence = 1
    elif expressionLabel == 'surprise':
        arousel = 3
        valence = 3
    elif expressionLabel == 'neutral':
        arousel = 2
        valence = 2

    packagedValues = ('face', arousel, valence, confidence)
    return packagedValues

def selectPrediction(predictionList):
    mostCommonValue = 0
    mostCommonCount = 0

    # find the most common value and store it's index
    for value in predictionList:
        count = predictionList.count(value)
        if count > mostCommonCount:
            mostCommonValue = value
            mostCommonCount = count
    
    return mostCommonValue


def findLastIndexOfMostCommonPrediction(predictionList, mostCommonValue):
    lastIndexOfCommonValue = 0
    for index in range(len(predictionList)):
        if predictionList[index] == mostCommonValue:
            lastIndexOfCommonValue = index
    return lastIndexOfCommonValue

# ---- Main Code
def runFacialExpressionRecognition(pretrainedModel, results):
    # Load the face detection model from cv2
    # used the haarcascade frontal face model since it is relatively fast and a common pre-trained model for
    # facial detection
    faceDetectionModel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize video capture
    defaultWebcamIndex = 0
    videoCapture = cv2.VideoCapture(defaultWebcamIndex)

    # Define a variable to keep track of the faces extracted from video frames
    faceCount = 0

    # Create a folder to store the images of the faces if it does not currently exist
    facesFolder = "faces/"
    if os.path.exists(facesFolder):
        shutil.rmtree(facesFolder)
    os.makedirs(facesFolder)

    # take 5 seconds of video
    start_time = time.time()
    while time.time() - start_time < 5:
        # Get frame to process from video camera
        status, videoFrame = videoCapture.read()
        
        # Ensure frame is read sucessfully
        if status is False:
            break

        # extract a list of faces from each frame
        facesFromFrame = extractFacesFromFrame(videoFrame, faceDetectionModel)

        # Create jpgs from list of faces
        createFaceImages(videoFrame, facesFromFrame, facesFolder)

    # Release the capture
    videoCapture.release()

    #Extract pixel values for each image in the faces folder
    facesFolder = "faces/"
    imageList = os.listdir(facesFolder)

    predictionLabels = []
    confidence = []

    for imageFile in imageList:
        # process the image
        imageFilePath = facesFolder + imageFile
        processedImage = processImage(imageFilePath)

        # use model to predict
        probabilityDistribution = pretrainedModel.predict(processedImage)
        category = numpy.argmax(probabilityDistribution)
        predictionLabels.append(category)

        # calculate confidence from probability
        confidencePercentage = probabilityDistribution[0][category] * 100
        confidence.append(confidencePercentage)
    
    selectedPrediction = selectPrediction(predictionLabels)
    # print(selectedPrediction)
    selectedPredictionIndex = findLastIndexOfMostCommonPrediction(predictionLabels, selectedPrediction)
    selectedPredictionConfidence = confidence[selectedPredictionIndex]

    results.append(packageInputsForFusion(selectedPrediction, selectedPredictionConfidence))
    # print("Face Analysis: ", results)



# def main():
#     print("audio audio")
#     videoModel = loadFaceModel()
#     results = []
#     while(1):
#         runFacialExpressionRecognition(videoModel, results)
#     print("done")


# if __name__ == "__main__":
#     main()