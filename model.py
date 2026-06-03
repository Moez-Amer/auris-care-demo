import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

class YamnetClassifier:
    def __init__(self):
        print("Loading YAMNet model from TensorFlow Hub... (This may take a moment on first run)")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        self.class_names = []
        with open(class_map_path) as csv_file:
            reader = csv.reader(csv_file)
            next(reader) # Skip header
            for row in reader:
                self.class_names.append(row[2])

    def classify_audio(self, audio_data: np.ndarray):
        scores, embeddings, spectrogram = self.model(audio_data)
        mean_scores = np.mean(scores, axis=0)
        top_class_index = np.argmax(mean_scores)
        inferred_class = self.class_names[top_class_index]
        confidence = mean_scores[top_class_index]
        
        return inferred_class, float(confidence)
        