mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

columnsToDrop = ['User_Input_Clip_Number', 'User_Input_Facial_Expression',
                    'User_Input_Speak', 'User_Input_Tone', 'Sentiment_Arousal',
                    'Face_Confidence', 'Face_Arousal', 'Face_Valence', 
                    'Sentiment_Valence', 'Tone_Arousal', 'Tone_Valence',
                    'Sentiment_Confidence', 'Tone_Confidence', 'Output_Confidence']
df = pd.read_csv('AnimationTagSheet-Datset.csv')
df = df.drop(columnsToDrop, axis=1)
# print(df.columns)

print('Valence Precision: ' + str(precision_score(df['Ground_Truth_Valence'], df['Output_Valence'], average='weighted')))
print('Valence Accuracy: ' + str(accuracy_score(df['Ground_Truth_Valence'], df['Output_Valence'])))
cm = confusion_matrix(df['Ground_Truth_Valence'], df['Output_Valence'], labels=[1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])
disp.plot()
# plt.show()
plt.title("Valence Confusion Matrix")
plt.savefig("Valence Confusion Matrix")

print('Arousal Precision: ' + str(precision_score(df['Ground_Truth_Arousal'], df['Output_Arousal'], average='weighted')))
print('Arousal Accuracy: ' + str(accuracy_score(df['Ground_Truth_Arousal'], df['Output_Arousal'])))
cm = confusion_matrix(df['Ground_Truth_Arousal'], df['Output_Arousal'], labels=[1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])
disp.plot()
# plt.show()
plt.title("Arousal Confusion Matrix")
plt.savefig("Arousal Confusion Matrix")