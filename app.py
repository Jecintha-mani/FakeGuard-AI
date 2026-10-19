"""
Fake News Detector - Flask Web Application
Serves the web dashboard and prediction API.
"""

import os
import json
import re
import joblib
import nltk
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Download NLTK resources silently
for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Global model/vectorizer ───────────────────────────────────────────────────
model = None
vectorizer = None
metadata = None

def load_models():
    """Load model, vectorizer, and metadata."""
    global model, vectorizer, metadata
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    vec_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
    meta_path = os.path.join(MODELS_DIR, 'metadata.json')

    if not os.path.exists(model_path):
        return False, "Model not found. Please run train_model.py first."

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        with open(meta_path) as f:
            metadata = json.load(f)
        return True, "Models loaded successfully"
    except Exception as e:
        return False, str(e)

def preprocess_text(text):
    """Clean and preprocess input text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
    lemmatizer = WordNetLemmatizer()
    try:
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens
                  if t not in stop_words and len(t) > 2]
        return ' '.join(tokens)
    except Exception:
        words = text.split()
        return ' '.join([w for w in words if w not in stop_words and len(w) > 2])

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    loaded, msg = load_models()
    model_loaded = loaded
    meta = metadata if loaded else None
    return render_template('index.html', model_loaded=model_loaded, metadata=meta, error=None if loaded else msg)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether news is real or fake."""
    if model is None or vectorizer is None:
        loaded, msg = load_models()
        if not loaded:
            return jsonify({'error': msg}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    title = data.get('title', '').strip()
    text = data.get('text', '').strip()

    if not title and not text:
        return jsonify({'error': 'Please provide news title or content'}), 400

    full_text = (title + ' ' + text).strip()
    processed = preprocess_text(full_text)

    if not processed:
        return jsonify({'error': 'Text could not be processed. Please try with more content.'}), 400

    try:
        features = vectorizer.transform([processed])
        prediction = int(model.predict(features)[0])

        # Confidence score
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba)) * 100
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(features)[0]
            confidence = float(min(abs(score) / 2 * 100 + 50, 99.0))
        else:
            confidence = 75.0

        label = 'FAKE' if prediction == 1 else 'REAL'

        # Feature importance analysis
        feature_names = vectorizer.get_feature_names_out()
        feature_values = features.toarray()[0]
        top_indices = feature_values.argsort()[-10:][::-1]
        top_words = [
            {'word': feature_names[i], 'score': round(float(feature_values[i]), 4)}
            for i in top_indices if feature_values[i] > 0
        ]

        # Linguistic red flags
        red_flags = []
        raw_text_lower = full_text.lower()
        upper_ratio = sum(1 for c in full_text if c.isupper()) / max(len(full_text), 1)
        exclaim_count = full_text.count('!')
        question_count = full_text.count('?')

        if upper_ratio > 0.3:
            red_flags.append({'flag': 'Excessive Capitalization', 'detail': f'{upper_ratio*100:.0f}% uppercase characters', 'severity': 'high'})
        if exclaim_count > 2:
            red_flags.append({'flag': 'Multiple Exclamation Marks', 'detail': f'{exclaim_count} exclamation marks detected', 'severity': 'medium'})
        alarm_phrases = ['breaking', 'bombshell', 'exposed', 'shocking', 'urgent', 'proof', 'wake up', 'do your research', 'they dont want', 'sheeple']
        found = [p for p in alarm_phrases if p in raw_text_lower]
        if found:
            red_flags.append({'flag': 'Sensationalist Language', 'detail': f'Detected: {", ".join(found[:3])}', 'severity': 'high'})
        anon_phrases = ['anonymous source', 'unnamed source', 'insider reveals', 'whistleblower', 'cannot be verified', 'trust me']
        found_anon = [p for p in anon_phrases if p in raw_text_lower]
        if found_anon:
            red_flags.append({'flag': 'Unverified Sources', 'detail': f'Detected: {", ".join(found_anon[:2])}', 'severity': 'medium'})
        share_phrases = ['share this', 'before it gets deleted', 'spread this', 'share before', 'forward this']
        found_share = [p for p in share_phrases if p in raw_text_lower]
        if found_share:
            red_flags.append({'flag': 'Viral Manipulation', 'detail': 'Article urges sharing before deletion', 'severity': 'high'})

        return jsonify({
            'label': label,
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'top_words': top_words,
            'red_flags': red_flags,
            'text_stats': {
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'exclamation_marks': exclaim_count,
                'question_marks': question_count,
                'uppercase_ratio': round(upper_ratio * 100, 1),
            },
            'model_used': metadata.get('best_model', 'Unknown') if metadata else 'Unknown'
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/metadata')
def get_metadata():
    """Return model metadata and performance statistics."""
    if metadata is None:
        load_models()
    if metadata:
        return jsonify(metadata)
    return jsonify({'error': 'Metadata not available'}), 404


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  Fake News Detector - Web App")
    print("=" * 50)
    loaded, msg = load_models()
    if loaded:
        print(f"Model loaded: {metadata.get('best_model')} (Accuracy: {metadata.get('best_accuracy', 0):.2%})")
    else:
        print(f"Warning: {msg}")
        print("Run train_model.py first to train the models.")
    print("\nStarting server at http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
