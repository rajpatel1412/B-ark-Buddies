# !pip install -U openai-whisper
# !pip install librosa
# !pip install torchaudio
# !pip install git+https://github.com/speechbrain/speechbrain.git

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import whisper
from transformers import pipeline
from transformers import AutoModelForAudioClassification
import librosa, torch

### LOADING MODELS 
# STT Model
stt_model = whisper.load_model("base")

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
sentiment_pipeline = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

# Tone analysis model
tone_model = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Categorical-Attributes", trust_remote_code=True)
audio = "c.mp4"


### Performing analysis 
# STT
transcription = stt_model.transcribe(audio)
text = transcription["text"]

# Sentiment Analysis
print("performing analysis")
sentiment = sentiment_pipeline(text)

# Tone Analysis
#get mean/std
mean = tone_model.config.mean
std = tone_model.config.std

raw_wav, _ = librosa.load(audio, sr=tone_model.config.sampling_rate)

#normalize the audio by mean/std
norm_wav = (raw_wav - mean) / (std+0.000001)

#generate the mask
mask = torch.ones(1, len(norm_wav))

#batch it (add dim)
wavs = torch.tensor(norm_wav).unsqueeze(0)

#predict
with torch.no_grad():
    pred = tone_model(wavs, mask)

#convert logits to probability
probabilities = torch.nn.functional.softmax(pred, dim=1)

# Find the index of the maximum value in the tensor
max_score_index = torch.argmax(pred, dim=1)
# Retrieve the corresponding label from model.config.id2label
max_label = tone_model.config.id2label[max_score_index.item()]
max_prob = probabilities[0, max_score_index.item()]

# Print the results
print(text)
print(sentiment)
print(max_label, "  ", max_prob.item())
