# ---- Sources/Documentation used:
# https://pypi.org/project/opencv-python/
# https://www.datacamp.com/tutorial/face-detection-python-opencv


import cv2
import time
import shutil
import os

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


# ---- Main Code

print("STARTING VIDEO CAPTURE")

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
    print(iterationNumber)

# Release the capture
videoCapture.release()

print("ENDING VIDEO CAPTURE")