import torch
import numpy as np
from transformers import AutoProcessor, ASTForAudioClassification
import warnings

warnings.filterwarnings("ignore")

class AudioClassifier:
    def __init__(self):
        self.model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = ASTForAudioClassification.from_pretrained(self.model_name)
        self.labels = self.model.config.id2label
        self.model.eval()

        # The care-home taxonomy: 9 event categories, each tiered by how urgently
        # staff must respond. CRITICAL = act now, URGENT = prompt check,
        # ADVISORY = log for clinical review (not a real-time alarm).
        #
        # Each category maps to an EXACT set of AudioSet label strings (matched
        # case-insensitively, exact equality — NOT substring). Substring matching
        # caused false positives like "Whimper (dog)" firing as human "Crying",
        # or "Church bell"/"Cowbell" firing as a medical "Call Bell". Exact labels
        # make those impossible by construction. Labels taken verbatim from AST's
        # 527-class AudioSet ontology.
        self.target_categories = {
            "Fall / Heavy Impact": {
                "labels": {"thump, thud", "thunk", "bang", "slam", "smash, crash", "clatter", "knock"},
                "tier": "CRITICAL"},
            "Choking / Gasping": {
                "labels": {"gasp", "wheeze", "pant"},
                "tier": "CRITICAL"},
            "Scream / Shout": {
                "labels": {"screaming", "shout", "yell", "bellow", "whoop", "children shouting"},
                "tier": "CRITICAL"},
            "Fire / Smoke Alarm": {
                "labels": {"smoke detector, smoke alarm", "fire alarm", "siren", "civil defense siren",
                           "emergency vehicle", "ambulance (siren)", "police car (siren)"},
                "tier": "CRITICAL"},
            "Crying / Pain": {
                # Human distress only — deliberately excludes "Whimper (dog)",
                # "Baby cry, infant cry", and "Battle cry".
                "labels": {"crying, sobbing", "wail, moan", "groan", "whimper"},
                "tier": "URGENT"},
            "Aggression / Breakage": {
                "labels": {"glass", "shatter", "breaking", "smash, crash"},
                "tier": "URGENT"},
            "Medical Alarm / Call Bell": {
                # Electronic alert tones only — excludes generic "Bell" (church/
                # cow/bicycle bells) and "Doorbell" (a separate wandering signal).
                "labels": {"alarm", "beep, bleep", "buzzer", "alarm clock", "ding-dong", "ding",
                           "telephone bell ringing"},
                "tier": "URGENT"},
            "Coughing": {
                "labels": {"cough"},
                "tier": "ADVISORY"},
            "Sneezing": {
                "labels": {"sneeze"},
                "tier": "ADVISORY"},
        }

    def classify_audio(self, audio_data):
        try:
            max_amp = np.max(np.abs(audio_data))
            if max_amp > 0:
                audio_data = audio_data / max_amp 

            inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            probabilities = torch.sigmoid(logits)[0]
            
            best_category = "Background Noise"
            best_specific_sound = "None"
            best_tier = "NONE"
            best_score = 0.0

            for i, prob in enumerate(probabilities):
                score = prob.item()
                label = self.labels[i].lower()

                for category, cfg in self.target_categories.items():
                    # Exact label match (case-insensitive) — no substring matching.
                    if label in cfg["labels"]:
                        if score > best_score:
                            best_score = score
                            best_category = category
                            best_specific_sound = self.labels[i]
                            best_tier = cfg["tier"]

            if best_score > 0.05 and best_category != "Background Noise":
                return best_tier, best_category, best_specific_sound, best_score
            else:
                return "NONE", "Background Noise", "None", 0.0
                
        except Exception as e:
            return "ERROR", "Error", "None", 0.0