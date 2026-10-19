# FakeGuard AI — Fake News Detector

A complete Machine Learning project to detect **fake news vs real news** using Python, scikit-learn, NLTK, and Flask.

---

## 🚀 Quick Start (After Installing Python)

### Option 1: One-Click Run (Windows)
```
Double-click: run.bat
```

### Option 2: Manual Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train ML models
python train_model.py

# 3. Start web server
python app.py

# 4. Open in browser
# → http://localhost:5000
```

---

## 📁 Project Structure

```
fake_news_detector/
├── app.py              ← Flask web server & prediction API
├── train_model.py      ← ML training script (run this first!)
├── requirements.txt    ← Python dependencies
├── run.bat             ← One-click Windows launcher
├── templates/
│   └── index.html      ← Beautiful web dashboard
├── static/
│   ├── style.css       ← Premium dark theme
│   ├── app.js          ← Frontend JavaScript
│   └── *.png           ← Generated charts (after training)
├── models/             ← Saved ML models (after training)
│   ├── best_model.pkl
│   ├── vectorizer.pkl
│   └── metadata.json
└── data/
    └── dataset.csv     ← Generated training dataset
```

---

## 🤖 ML Models Trained

| Model | Type |
|-------|------|
| Logistic Regression | Linear classifier |
| Passive Aggressive | Online learning |
| Naive Bayes | Probabilistic |
| Linear SVC | Support Vector Machine |
| Random Forest | Ensemble |

The best-performing model is automatically selected and saved.

---

## 🔬 How It Works

1. **Text Input** → News headline + body
2. **NLP Preprocessing** → NLTK tokenization, stopword removal, lemmatization
3. **TF-IDF Vectorization** → Converts text to numerical features (10,000 features, bigrams)
4. **ML Classification** → Predicts REAL or FAKE with confidence score
5. **Red Flag Analysis** → Detects sensationalist language, caps abuse, unverified sources

---

## 📊 Features

- ✅ Real-time fake news prediction
- ✅ Confidence score with animated ring
- ✅ Red flag detection (ALL CAPS, exclamations, alarm phrases)
- ✅ Top influencing keyword visualization
- ✅ Model performance analytics charts
- ✅ Comparison of 5 ML algorithms
- ✅ Premium dark-themed web dashboard
- ✅ REST API endpoints

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/predict` | POST | Predict fake/real |
| `/api/metadata` | GET | Model statistics |
| `/api/health` | GET | Server health |

### Example API Call
```python
import requests

response = requests.post('http://localhost:5000/predict', json={
    "title": "Scientists discover cure for all diseases",
    "text": "Researchers announced today..."
})
print(response.json())
```

---

## ⚙️ Requirements

- Python 3.8+
- pip (comes with Python)
- Internet connection (for NLTK downloads)
