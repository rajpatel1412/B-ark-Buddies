import face
import audio
import threading
import numpy

# Helper functions
def performWeighting(valueA, confidenceA, valueB, confidenceB, valueC, confidenceC):
    # values of 1 are places at index 0, values of 2 are places at index 2, values of 3 are places at index 2

    # where the sorted values list holds [[confidence for values = 1], [confidence for values = 2], [confidence for values = 3]]
    sortedValues = [[], [], []]
    sortedValues[valueA - 1].append(confidenceA)
    sortedValues[valueB - 1].append(confidenceB)
    sortedValues[valueC -1].append(confidenceC)

    # calculate average confidence for each value rating
    averageConfidencePerValue = []
    for valueCategory in sortedValues:
        sumOfConfidences = sum(valueCategory)
        numberOfConfidences = len(valueCategory)

        if numberOfConfidences == 0:
            averageConfidence = 0
        else:
            averageConfidence = sumOfConfidences/numberOfConfidences

        averageConfidencePerValue.append(averageConfidence)

    predictedIndex = numpy.argmax(averageConfidencePerValue)
    predictedValue = predictedIndex + 1
    confidenceOfPrediction = averageConfidencePerValue[predictedIndex]

    return (predictedValue, confidenceOfPrediction)

# for fusion
def fuseModalities(facialInputs, sematicVoiceInput, toneVoiceInput):

    # extract inputs for fusion
    faceIdentifer, faceArousel, faceValence, faceConfidence = facialInputs
    sentimentIdentifier, semanticArousel, semanticValence, semanticConfidence, command = sematicVoiceInput
    tineIdentifier, toneArousel, toneValence, toneConfidence = toneVoiceInput

    # preform weighting for arousel
    predictedArousal, predictedArouselConfidence = performWeighting(faceArousel, faceConfidence * 1.5, semanticArousel, semanticConfidence, toneArousel, toneConfidence * 1.5)

    # preform weighting for valence
    predictedValence, predictedValenceConfidence = performWeighting(faceValence, faceConfidence * 1.5, semanticValence, semanticConfidence, toneValence, toneConfidence * 1.5)

    # generate overall confidence
    postFusionConfidence = (predictedArouselConfidence + predictedValenceConfidence)/2

    # pack fusion output values
    return (predictedArousal, predictedValence, postFusionConfidence, command)

def main():
    print("woof woof")

    # Pre load the speech and face models
    audioSentimentModelBase = audio.loadAudioSentimentModelBase()
    audioSentimentModel = audio.loadAudioSentimentModel()
    audioToneModel = audio.loadAudioToneModel()
    faceModel = face.loadFaceModel()

    while(True):
        # save processing results into a list
        results = []

        videoProcessingThread = threading.Thread(target=face.runFacialExpressionRecognition, args=(faceModel, results))
        audioProcessingThread = threading.Thread(target=audio.runAudioRecognition, args=(audioSentimentModelBase, audioSentimentModel, audioToneModel, results))

        videoProcessingThread.start()
        audioProcessingThread.start()

        videoProcessingThread.join()
        audioProcessingThread.join()

        faceInputs = None
        semanticInputs = None
        toneInputs = None
        for result in results:
            if result[0] == 'face':
                faceInputs = result
            elif result[0] == 'semantic':
                semnaticInputs = result
            elif result[0] == 'tone':
                toneInputs = result

        fusedValues = fuseModalities(faceInputs, semnaticInputs, toneInputs)
        print(fusedValues)



if __name__ == "__main__":
    main()