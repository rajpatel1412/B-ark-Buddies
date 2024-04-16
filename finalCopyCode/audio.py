# Documentation/Sources:
# https://github.com/openai/whisper
# https://huggingface.co/nateraw/bert-base-uncased-emotion
# https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
# https://stackoverflow.com/questions/40704026/voice-recording-using-pyaudio

# Suppress warning for a cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Needed imports
import whisper
from transformers import pipeline
import librosa, torch
import pyaudio
import wave

def loadSpeechToTextModel():
    # Speech To Text Model
    sttModel = whisper.load_model("base.en")
    return sttModel

def loadAudioSentimentModel():
    # Sentiment Analysis model
    sentimentPipeline = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")
    return sentimentPipeline

def loadAudioToneModel():
    # Tone analysis model
    toneModel = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    return toneModel

# Encode sentiment outputs to be used as fusion inputs
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

    packagedValues = ('sentiment', arousel, valence, confidence, command)
    return packagedValues

# Encode tone outputs to be used as fusion inputs 
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
    # Audio collection adapted from the stackoverflow post
    # https://stackoverflow.com/questions/40704026/voice-recording-using-pyaudio
    # Settings for audio collection
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    # Initialize pyaudio
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    # Iteratively reading chunks of data from the stream and appending them to frame
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
    # SpeechToText
    transcription = sttModel.transcribe(audioFile)
    text = transcription["text"]
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