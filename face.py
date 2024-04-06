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

import sys # for debugging (add argument no-video to use images from previous run)

# ---- Helper Functions
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

def extractFacesFromFrame(videoFrame):
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

# ---- Main Code

arguments = sys.argv

# skip video capture and use existing images (for debugging purposes)
captureVideo = len(arguments) == 1
if captureVideo:

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

    iterationNumber = 0
    maxNumIterations = 400
    while iterationNumber < maxNumIterations:
        # Get frame to process from video camera
        status, videoFrame = videoCapture.read()
        
        # Ensure frame is read sucessfully
        if status is False:
            break

        # extract a list of faces from each frame
        facesFromFrame = extractFacesFromFrame(videoFrame)

        # Create jpgs from list of faces
        createFaceImages(videoFrame, facesFromFrame, facesFolder)

        iterationNumber = iterationNumber + 1

    # Release the capture
    videoCapture.release()

# model from: https://github.com/akash720/Facial-expression-recognition/blob/master/README.md
# Note: had to train it and then save it with a newer version of keras for some reason in google collab
pretrainedModel = load_model("my_model.h5")
pretrainedModel.summary()

#Extract pixel values for each image in the faces folder
facesFolder = "faces/"
imageList = os.listdir(facesFolder)

predictionLabels = []
confidence = []

for imageFile in imageList:

    imageFilePath = facesFolder + imageFile
    processedImage = processImage(imageFilePath)

    print("---")
    probabilityDistribution = pretrainedModel.predict(processedImage)
    print(probabilityDistribution)
    category = numpy.argmax(probabilityDistribution)
    print(category)

    predictionLabels.append(category)
    confidencePercentage = probabilityDistribution[0][category] * 100
    confidence.append(confidencePercentage)
    print(confidencePercentage)
    print("---")


## TODO: remove all code below, this is for testing purposes only ########

def testing_DELETE_ME(translateNumberToCategory, facesFolder, imageList, predictionLabels):
    num_cols = 5  # Number of columns in the grid
    num_rows = -(-len(imageList) // num_cols)  # Calculate the number of rows needed to accommodate all images


    predictionIndex = 0
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

    for i, faceImage_filename in enumerate(imageList):
        faceImage = imread(facesFolder + faceImage_filename)
        label = translateNumberToCategory(predictionLabels[predictionIndex]) + "\n " + str(confidence[i]) + "%"

        imageTitle = f"{label}\n({faceImage_filename})"
        row = i // num_cols  # Calculate the row index for this subplot
        col = i % num_cols   # Calculate the column index for this subplot

        axs[row, col].imshow(faceImage)
        axs[row, col].set_title(imageTitle)
        axs[row, col].axis('off')
        predictionIndex += 1

# Hide any remaining empty subplots
    for i in range(len(imageList), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].axis('off')
        axs[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()

testing_DELETE_ME(translateNumberToCategory, facesFolder, imageList, predictionLabels)  

####################################################################

def packageInputsForFusion(expressionLabel, confidence):
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

    packagedValues = (arousel, valence, confidence)
    return packagedValues

def performWeighting(valueA, confidenceA, valueB, confidenceB, valueC, confidenceC):
    # values of 1 are places at index 0, values of 2 are places at index 2, values of 3 are places at index 2

    # where the sorted values list holds [[confidence for values = 1], [confidence for values = 2], [confidence for values = 3]]
    sortedValues = [[], [], []]
    sortedValues[valueA - 1].append[confidenceA]
    sortedValues[valueB - 1].append[confidenceB]
    sortedValues[valueC -1].append[confidenceC]

    # calculate average confidence for each value rating
    averageConfidencePerValue = []
    for valueCategory in sortedValues:
        sumOfConfidences = sum(valueCategory)
        numberOfConfidences = len(valueCategory)

        averageConfidence = sumOfConfidences/numberOfConfidences

        averageConfidencePerValue.append(averageConfidence)

    predictedIndex = numpy.argmax(averageConfidencePerValue)
    predictedValue = predictedIndex + 1
    confidenceOfPrediction = averageConfidencePerValue[predictedIndex]

    return (predictedValue, confidenceOfPrediction)


def fuseModalities(facialInputs, sematicVoiceInput, toneVoiceInput):

    # extract inputs for fusion
    faceArousel, faceValence, faceConfidence = facialInputs
    semanticArousel, semanticValence, semanticConfidence, command = sematicVoiceInput
    toneArousel, toneValence, toneConfidence = toneVoiceInput

    # preform weighting for arousel
    predictedArousal, predictedArouselConfidence = performWeighting(faceArousel, faceConfidence, semanticArousel, semanticConfidence, toneArousel, toneConfidence)

    # preform weighting for valence
    predictedValence, predictedValenceConfidence = performWeighting(faceValence, faceConfidence, semanticValence, semanticConfidence, toneValence, toneConfidence)

    # generate overall confidence
    postFusionConfidence = (predictedArouselConfidence + predictedValenceConfidence)/2

    # pack fusion output values
    return (predictedArousal, predictedValence, postFusionConfidence, command)