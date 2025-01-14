# ---- Sources/Documentation used:
# https://pypi.org/project/opencv-python/
# https://www.datacamp.com/tutorial/face-detection-python-opencv
# https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
# https://www.geeksforgeeks.org/face-detection-using-cascade-classifier-using-opencv-python/
# https://github.com/akash720/Facial-expression-recognition/blob/master/README.md
# https://stackoverflow.com/questions/76616042/attributeerror-module-pil-image-has-no-attribute-antialias

# ---- Libraries
import cv2
import time
import shutil
import os
import numpy

from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras import preprocessing
from PIL import Image

# ---- Helper Functions

# Loads the pre-trained facial recognition model
def loadFaceModel():
    # model from: https://github.com/akash720/Facial-expression-recognition/blob/master/README.md
    # Note: had to train it and then save it with a newer version of keras, the updated model is included in this directory
    pretrainedModel = load_model("my_model.h5")
    return pretrainedModel

# Saves each face in it's own image file, each file name is prefixed with the timestamp so that no image files have the same name
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

# Use the provided face detection model to to extract faces from the provided video frame
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

# Translate the number label to the string discription for increased readibility 
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

# Translate expression value into arousel and valence metrics, package data into a tuple in preparation for late fusion
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

# If multiple different expressions were detected during recording select the most common expression label
def selectPrediction(predictionList):
    mostCommonValue = 0
    mostCommonCount = 0

    # find the most common value
    for value in predictionList:
        count = predictionList.count(value)
        if count > mostCommonCount:
            mostCommonValue = value
            mostCommonCount = count
    
    return mostCommonValue

# If multiple different expressions were detected during recording return the last index of the most common expression
# label, this will be used to select the confidence value that will be passed to into the late fusion process
def findLastIndexOfMostCommonPrediction(predictionList, mostCommonValue):
    lastIndexOfCommonValue = 0
    
    # find and store the index of the most common value
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

    # Create a folder to store the images of the faces if it does not currently exist
    facesFolder = "faces/"
    if os.path.exists(facesFolder):
        shutil.rmtree(facesFolder)
    os.makedirs(facesFolder)

    # Take 5 seconds of video
    recordingTime = time.time()
    while time.time() - recordingTime < 5:

        # Get frame to process from video camera
        status, videoFrame = videoCapture.read()
        
        # Ensure frame is read sucessfully
        if status is False:
            break

        # Extract a list of faces from each frame
        facesFromFrame = extractFacesFromFrame(videoFrame, faceDetectionModel)

        # Create jpgs from list of faces and store them in faces folder
        createFaceImages(videoFrame, facesFromFrame, facesFolder)

    # Release the capture
    videoCapture.release()

    # For each image created, pre-process it, and use model to predict it's label
    imageList = os.listdir(facesFolder)
    predictionLabels = []
    confidence = []

    for imageFile in imageList:
        # Pre-process the image
        imageFilePath = facesFolder + imageFile
        processedImage = processImage(imageFilePath)

        # Use model to predict
        probabilityDistribution = pretrainedModel.predict(processedImage)
        category = numpy.argmax(probabilityDistribution)
        predictionLabels.append(category)

        # Calculate confidence from probability
        confidencePercentage = probabilityDistribution[0][category] * 100
        confidence.append(confidencePercentage)
    
    # Select prediction and confidence values to pass to late fusion process
    selectedPrediction = selectPrediction(predictionLabels)
    selectedPredictionIndex = findLastIndexOfMostCommonPrediction(predictionLabels, selectedPrediction)
    selectedPredictionConfidence = confidence[selectedPredictionIndex]

    # Add prediction and confidence values to results list for fusion
    results.append(packageInputsForFusion(selectedPrediction, selectedPredictionConfidence))