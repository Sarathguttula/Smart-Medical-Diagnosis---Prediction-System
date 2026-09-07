from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer


DISEASE_PROFILES: dict[str, list[str]] = {
    "Common cold": ["cough", "sore_throat", "runny_nose", "fatigue", "headache"],
    "Influenza": ["fever", "cough", "body_ache", "fatigue", "headache", "chills"],
    "Migraine": ["headache", "nausea", "light_sensitivity", "dizziness"],
    "Allergic rhinitis": ["sneezing", "runny_nose", "itchy_eyes", "congestion"],
    "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "stomach_pain", "fever"],
    "Urinary tract infection": ["painful_urination", "frequent_urination", "fever", "fatigue"],
}

RECOMMENDATIONS = {
    "Common cold": ["Rest and drink fluids.", "Monitor symptoms for changes."],
    "Influenza": ["Rest, hydrate, and monitor your temperature.", "Contact a clinician promptly if symptoms are severe or worsening."],
    "Migraine": ["Rest in a quiet, dark room and hydrate.", "Discuss recurring or severe headaches with a clinician."],
    "Allergic rhinitis": ["Reduce exposure to suspected triggers.", "Ask a pharmacist or clinician about suitable symptom relief."],
    "Gastroenteritis": ["Take small, frequent sips of fluid.", "Seek care promptly for dehydration, blood, or persistent symptoms."],
    "Urinary tract infection": ["Arrange a clinical evaluation for testing.", "Seek urgent care for fever with flank pain or feeling very unwell."],
}


@dataclass
class DiagnosisModel:
    vectorizer: MultiLabelBinarizer
    classifier: LogisticRegression

    @classmethod
    def train(cls) -> "DiagnosisModel":
        rng = np.random.default_rng(42)
        labels = list(DISEASE_PROFILES)
        symptoms = sorted({symptom for profile in DISEASE_PROFILES.values() for symptom in profile})
        vectorizer = MultiLabelBinarizer(classes=symptoms)
        rows: list[list[str]] = []
        targets: list[str] = []
        for disease, profile in DISEASE_PROFILES.items():
            for _ in range(80):
                selected = [symptom for symptom in symptoms if symptom in profile and rng.random() < 0.7]
                selected += [symptom for symptom in symptoms if symptom not in profile and rng.random() < 0.03]
                rows.append(selected or [profile[0]])
                targets.append(disease)
        features = vectorizer.fit_transform(rows)
        classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        classifier.fit(features, targets)
        return cls(vectorizer, classifier)

    @property
    def symptoms(self) -> list[str]:
        return list(self.vectorizer.classes_)

    def predict(self, selected_symptoms: list[str], limit: int = 3) -> list[dict]:
        features = self.vectorizer.transform([selected_symptoms])
        probabilities = self.classifier.predict_proba(features)[0]
        ranked = np.argsort(probabilities)[::-1][:limit]
        results = []
        for index in ranked:
            probability = float(probabilities[index])
            disease = str(self.classifier.classes_[index])
            results.append({
                "disease": disease,
                "probability": round(probability, 4),
                "risk_level": self._risk_level(probability),
                "explanation": self.explain(features, index),
                "recommendations": RECOMMENDATIONS[disease],
            })
        return results

    def explain(self, features, class_index: int, limit: int = 4) -> list[dict[str, float | str]]:
        feature_values = features.toarray()[0] if hasattr(features, "toarray") else np.asarray(features)[0]
        contributions = feature_values * self.classifier.coef_[class_index]
        present = np.where(feature_values > 0)[0]
        ranked = sorted(present, key=lambda idx: abs(contributions[idx]), reverse=True)[:limit]
        return [{"symptom": str(self.vectorizer.classes_[idx]), "impact": round(float(contributions[idx]), 3)} for idx in ranked]

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.65:
            return "high"
        if probability >= 0.35:
            return "moderate"
        return "low"


model = DiagnosisModel.train()
