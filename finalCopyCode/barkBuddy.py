# ---- Sources/Documentation used:
# https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
# https://www.geeksforgeeks.org/multithreading-python-set-1/
# https://stackoverflow.com/questions/21759946/how-to-exit-program-using-the-enter-key

# ---- Libraries
import face
import audio
import game
import threading
import numpy
import time
import warnings

# ---- Program Flags
exitFlag = False

# ---- Helper Functions

# Weight each value by calculating it's average confidence then find the the value with the
# highest average confidence
def performWeighting(valueA, confidenceA, valueB, confidenceB, valueC, confidenceC):
    # values of 1 are places at index 0, values of 2 are places at index 2, values of 3 are places at index 3
    # where the sorted values list holds [[confidence for values = 1], [confidence for values = 2], [confidence for values = 3]]
    sortedValues = [[], [], []]
    sortedValues[valueA - 1].append(confidenceA)
    sortedValues[valueB - 1].append(confidenceB)
    sortedValues[valueC -1].append(confidenceC)

    # Calculate average confidence for each value rating
    averageConfidencePerValue = []
    for valueCategory in sortedValues:
        sumOfConfidences = sum(valueCategory)
        numberOfConfidences = len(valueCategory)
    
        # Handle case for divide by zero
        if numberOfConfidences == 0:
            averageConfidence = 0
        else:
            averageConfidence = sumOfConfidences/numberOfConfidences

        averageConfidencePerValue.append(averageConfidence)

    # Find value with highest average confidence 
    predictedIndex = numpy.argmax(averageConfidencePerValue)
    predictedValue = predictedIndex + 1
    confidenceOfPrediction = averageConfidencePerValue[predictedIndex]

    return (predictedValue, confidenceOfPrediction)

# Late fusion algorithm that accepts data for each modality and produces an output
def fuseModalities(facialInputs, sematicVoiceInput, toneVoiceInput):

    # Extract inputs for fusion
    faceIdentifer, faceArousel, faceValence, faceConfidence = facialInputs
    sentimentIdentifier, semanticArousel, semanticValence, semanticConfidence, command = sematicVoiceInput
    tineIdentifier, toneArousel, toneValence, toneConfidence = toneVoiceInput

    # Preform weighting for arousel
    predictedArousal, predictedArouselConfidence = performWeighting(faceArousel, faceConfidence, semanticArousel, semanticConfidence * 0.8, toneArousel, toneConfidence * 1.2)

    # Preform weighting for valence
    predictedValence, predictedValenceConfidence = performWeighting(faceValence, faceConfidence, semanticValence, semanticConfidence * 0.8, toneValence, toneConfidence * 1.2)

    # Generate overall confidence
    postFusionConfidence = (predictedArouselConfidence + predictedValenceConfidence)/2

    # Pack fusion output values
    return (predictedArousal, predictedValence, postFusionConfidence, command)

# Handles keyboard input which allows the user to quit
def handleInput():
    global exitFlag

    while(True):
        userKeyInput = input("Please press 'q' to stop the program at any time!: \n")
        if userKeyInput.lower() == 'q':
            print("Exiting!\n")
            exitFlag = True
            break

# ---- Main Code
def main():
    # Ignore any warnings that might occur
    warnings.filterwarnings("ignore", category=UserWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Pre load the speech and face models
    audioSpeechToTextModel = audio.loadSpeechToTextModel()
    audioSentimentModel = audio.loadAudioSentimentModel()
    audioToneModel = audio.loadAudioToneModel()
    faceModel = face.loadFaceModel()

    inputThread = threading.Thread(target=handleInput)
    inputThread.start()

    while(exitFlag == False):
        # Save processing results from each modality into a list
        results = []

        # Collect modality data
        videoProcessingThread = threading.Thread(target=face.runFacialExpressionRecognition, args=(faceModel, results))
        audioProcessingThread = threading.Thread(target=audio.runAudioRecognition, args=(audioSpeechToTextModel, audioSentimentModel, audioToneModel, results))

        videoProcessingThread.start()
        audioProcessingThread.start()

        videoProcessingThread.join()
        audioProcessingThread.join()

        # Organize collected data from threads
        faceInputs = None
        sentimentInputs = None
        toneInputs = None

        for result in results:
            if result[0] == 'face':
                faceInputs = result
            elif result[0] == 'sentiment':
                sentimentInputs = result
            elif result[0] == 'tone':
                toneInputs = result

        # Preform fusion
        fusedValues = fuseModalities(faceInputs, sentimentInputs, toneInputs)

        # Run game based on fusion results
        game.runGame(fusedValues[0], fusedValues[1], fusedValues[2], fusedValues[3])

    inputThread.join()

if __name__ == "__main__":
    main()