import face
import audio
import threading
import numpy

# Helper functions
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

# for fusion
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

def main():
    print("woof woof")

    # Pre load the speech and face models
    audioSentimentModelBase = audio.loadAudioSentimentModelBase()
    audioSentimentModel = audio.loadAudioSentimentModel()
    faceModel = face.loadFaceModel()

    while(True):
        # save processing results into a list
        results = []

        videoProcessingThread = threading.Thread(target=face.runFacialExpressionRecognition, args=(faceModel, results))
        audioProcessingThread = threading.Thread(target=audio.runAudioRecognition, args=(audioSentimentModelBase, audioSentimentModel, results))

        videoProcessingThread.start()
        audioProcessingThread.start()

        videoProcessingThread.join()
        audioProcessingThread.join()


if __name__ == "__main__":
    main()