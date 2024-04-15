# !pip install -U openai-whisper
# !pip install librosa
# !pip install torchaudio

### IMPORTS 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import whisper
from transformers import pipeline
from transformers import AutoModelForAudioClassification
import librosa, torch
import pyaudio
import wave

def loadSpeechToTextModel():
    ### LOADING MODELS 
    # STT Model
    sttModel = whisper.load_model("base.en")
    return sttModel

def loadAudioSentimentModel():
    # Sentiment Analysis model
    '''
        More emotions however not a great accuracy
        Possible outputs:
            joy (or happiness): Indicating content that expresses joy, happiness, or pleasure.
            anger: Reflecting content that expresses anger, irritation, or frustration.
            fear: Signifying text that expresses fear, anxiety, or apprehension.
            sadness: Identifying content that expresses sadness, gloominess, or disappointment.
            disgust: Highlighting content that expresses disgust, revulsion, or contempt.
            surprise: For content that expresses surprise, amazement, or shock.
            neutral: Indicating content that doesn't strongly express any of the above emotions.
        Wasn't able to successfully replicate all

    '''
    sentimentPipeline = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")
    return sentimentPipeline

def loadAudioToneModel():
    # Tone analysis model
    toneModel = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    return toneModel

# for fusion
def packageSentimentInputsForFusion(expressionLabel, confidence, command):
    arousel = 0
    valence = 0

    # 1 is low/negative and 3 is high/positive
    if expressionLabel == 'anger':
        arousel = 3
        valence = 1
    elif expressionLabel == 'disgust':
        arousel = 2
        valence = 1
    elif expressionLabel == 'fear':
        arousel = 3
        valence = 1
    elif expressionLabel == 'joy':
        arousel = 2
        valence = 3
    elif expressionLabel == 'sadness':
        arousel = 1
        valence = 1
    elif expressionLabel == 'surprise':
        arousel = 3
        valence = 3
    elif expressionLabel == 'neutral':
        arousel = 2
        valence = 2

    packagedValues = ('semantic', arousel, valence, confidence, command)
    return packagedValues

def packageToneInputsForFusion(expressionLabel, confidence):
    arousel = 0
    valence = 0

    # 1 is low/negative and 3 is high/positive
    if expressionLabel == 'angry':
        arousel = 3
        valence = 1
    elif expressionLabel == 'disgust':
        arousel = 2
        valence = 1
    elif expressionLabel == 'fearful':
        arousel = 3
        valence = 1
    elif expressionLabel == 'happy':
        arousel = 2
        valence = 3
    elif expressionLabel == 'sad':
        arousel = 1
        valence = 1
    elif expressionLabel == 'surprised':
        arousel = 3
        valence = 3
    elif expressionLabel == 'neutral' or expressionLabel == 'calm':
        arousel = 2
        valence = 2

    packagedValues = ('tone', arousel, valence, confidence)
    return packagedValues

def runAudioRecognition(sttModel, sentimentPipeline, audioToneModel, results):
    ### COLLECTING AUDIO
    # Settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    # Initialize pyaudio
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    audioFile = WAVE_OUTPUT_FILENAME

    ### Performing analysis 
    # STT
    transcription = sttModel.transcribe(audioFile)
    text = transcription["text"]
    # Print the Text
    print("Text: ", text)

    # Sentiment Analysis
    sentiment = sentimentPipeline(text)

    # Tonal Analysis
    tone = audioToneModel(audioFile)

    # Supported tricks
    tricks = ["Stand", "Shake", "Turn", "Sit", "Bang"]
    command = ""

    # Extract the last commandn
    for trick in tricks:
    # Check if the trick is in the input text
        if trick.lower() in text.lower():
            # Assign the trick to command
            command = trick

    results.append(packageSentimentInputsForFusion(sentiment[0]['label'], sentiment[0]['score'] * 100, command))
    results.append(packageToneInputsForFusion(tone[0]['label'], tone[0]['score'] * 100))
    print(results)

def main():
    audioSpeechToTextModel = loadSpeechToTextModel()
    audioSentimentModel = loadAudioSentimentModel()
    audioToneModel = loadAudioToneModel()
    results = []
    runAudioRecognition(audioSpeechToTextModel, audioSentimentModel, audioToneModel, results)

if __name__ == "__main__":
    main()