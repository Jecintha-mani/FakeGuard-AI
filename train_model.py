"""
Fake News Detector - Dataset Generator & Model Trainer
Generates synthetic training data and trains ML models for fake news detection.
"""

import pandas as pd
import numpy as np
import os
import re
import string
import nltk
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Synthetic Dataset ─────────────────────────────────────────────────────────

REAL_HEADLINES = [
    "Scientists discover new species of deep-sea fish in Pacific Ocean",
    "Stock markets rise as inflation data shows signs of cooling",
    "Government announces new infrastructure spending package",
    "Study links regular exercise to improved mental health outcomes",
    "Tech giant reports record quarterly earnings amid AI boom",
    "Climate summit reaches historic agreement on carbon emissions",
    "Central bank raises interest rates to combat rising prices",
    "New vaccine shows 95% efficacy in phase 3 clinical trials",
    "World leaders gather for annual United Nations General Assembly",
    "Research confirms benefits of Mediterranean diet for longevity",
    "NASA successfully launches new space telescope into orbit",
    "Olympic committee announces new host city for 2032 games",
    "Record rainfall causes flooding in multiple coastal regions",
    "International trade agreement signed between major economies",
    "University study reveals link between sleep and cognitive function",
    "Breakthrough cancer treatment receives FDA approval",
    "Global supply chain disruptions ease as shipping rates drop",
    "New electric vehicle model achieves record 500-mile range",
    "Scientists develop biodegradable plastic alternative from seaweed",
    "Major earthquake triggers tsunami warning across Pacific coast",
    "Tech company announces layoffs affecting thousands of employees",
    "Study shows children benefit from bilingual education",
    "Federal reserve signals potential rate cuts in coming months",
    "Archaeological discovery sheds light on ancient civilization",
    "Renewable energy surpasses fossil fuels in electricity generation",
    "New treatment shows promise for patients with Alzheimer's disease",
    "Global temperatures hit record highs for third consecutive year",
    "International space station crew returns safely after six-month mission",
    "Major pharmaceutical company recalls medication over safety concerns",
    "City council approves new affordable housing development plan",
    "Scientists detect gravitational waves from distant galaxy merger",
    "Country achieves milestone of 50% renewable electricity generation",
    "New study reveals impact of social media on teenage mental health",
    "Government launches digital ID program for citizens",
    "Artificial intelligence helps doctors diagnose rare diseases faster",
    "Record number of tourists visit national parks this summer",
    "New data privacy law takes effect across European Union",
    "Marine biologists document largest coral reef restoration project",
    "Stock exchange introduces new trading rules to prevent volatility",
    "University research develops faster biodegradable battery technology",
    "Health officials warn of rising cases of respiratory illness",
    "International court rules on territorial dispute between nations",
    "Wildfire destroys thousands of acres in western region",
    "New bridge construction project creates thousands of local jobs",
    "Scientists measure lowest Arctic ice extent on record",
    "Pharmaceutical trials begin for new diabetes treatment",
    "Local government implements new recycling program for residents",
    "Air quality improves in major cities after new emission regulations",
    "Tech startup raises 500 million dollars in series B funding",
    "Research shows benefits of green spaces in urban environments",
]

REAL_BODIES = [
    "Scientists from the National Oceanic and Atmospheric Administration confirmed the discovery following a six-month deep-sea expedition. The species, tentatively named Bathypelagus novus, was found at depths exceeding 3,000 meters and exhibits bioluminescent properties never previously observed.",
    "The Dow Jones Industrial Average closed up 1.2% following the release of consumer price index data showing inflation at 3.1%, below analyst expectations. Market participants interpreted the data as increasing the likelihood of interest rate cuts in the coming quarter.",
    "The federal government unveiled a 1.2 trillion dollar infrastructure package targeting roads, bridges, broadband expansion, and clean energy projects. The legislation passed with bipartisan support and is expected to create an estimated 2 million jobs over the next decade.",
    "A comprehensive meta-analysis published in the Journal of Psychiatric Research examined data from over 200 studies involving 1.2 million participants, finding that individuals who exercised at least 150 minutes per week had a 30% lower risk of depression.",
    "The technology corporation reported quarterly revenues of 94.3 billion dollars, exceeding Wall Street estimates by 6%. The company attributed the strong performance to growing demand for its artificial intelligence cloud services and enterprise software solutions.",
    "After two weeks of negotiations, delegates from 196 countries reached a landmark agreement to reduce carbon emissions by 45% by 2035. The accord includes binding commitments and a 300 billion dollar fund to help developing nations transition to clean energy.",
    "The central bank monetary policy committee voted unanimously to raise the benchmark interest rate by 25 basis points to 5.5%. Officials cited persistent core inflation and a robust labor market as justification for the decision.",
    "The pharmaceutical company announced that its mRNA-based vaccine demonstrated 95.2% efficacy in preventing severe illness, with a favorable safety profile across all demographic groups. Regulatory submissions are expected within the next 90 days.",
    "Heads of state and government convened at United Nations headquarters in New York for the annual high-level general debate. This year agenda focused on climate change, global food security, digital governance, and conflict resolution.",
    "A 30-year longitudinal study tracking 11,000 individuals found that adherence to Mediterranean dietary principles was associated with a 25% reduction in cardiovascular disease risk and a 20% lower incidence of cognitive decline in adults over 65.",
    "The space agency successfully deployed the next-generation infrared telescope into a stable orbit at the L2 Lagrange point. Scientists expect the instrument to revolutionize our understanding of galaxy formation and potentially detect biosignatures on exoplanets.",
    "The International Olympic Committee selected Brisbane as the host city for the 2032 Summer Olympic Games following an extensive evaluation of infrastructure, sustainability plans, and public support across candidate cities.",
    "Emergency services responded to widespread flooding affecting low-lying coastal areas following 48 hours of record precipitation. Hundreds of residents were evacuated, and authorities warned that floodwaters could rise further in the coming days.",
    "Trade ministers from 24 nations finalized a comprehensive free trade agreement eliminating tariffs on 90% of goods and establishing new frameworks for digital commerce, intellectual property protection, and environmental standards.",
    "Researchers at Stanford University published findings suggesting that adults who consistently get 7-9 hours of sleep per night perform 40% better on memory tests and demonstrate significantly improved executive function compared to those sleeping fewer hours.",
]

FAKE_HEADLINES = [
    "Scientists CONFIRM moon is actually hollow and filled with alien technology",
    "BREAKING: Government secretly putting mind control chips in COVID vaccines",
    "Doctors HATE this one weird trick that cures cancer overnight",
    "SHOCK: Obama born on Mars, new documents PROVE it",
    "Aliens land in Nevada and meet with secret government officials",
    "5G towers causing mass sterilization, leaked documents reveal",
    "Bill Gates admits plan to reduce world population through vaccines",
    "Secret underground city discovered beneath Antarctica with Nazi technology",
    "PROOF: The moon landing was staged in Hollywood studio by Stanley Kubrick",
    "Deep state agents caught putting fluoride in water supply to control minds",
    "Chemtrails from planes confirmed to contain behavior-modifying chemicals",
    "BOMBSHELL: Earth is flat and NASA has been lying for 50 years",
    "Miracle cure for diabetes discovered but BIG PHARMA is hiding it",
    "New World Order plan to enslave humanity EXPOSED in leaked memo",
    "URGENT: Doctors secretly given orders to euthanize COVID patients for profit",
    "Global elites caught performing satanic rituals in Washington DC",
    "Leaked documents prove climate change is a hoax invented by bankers",
    "SHOCKING: Hollywood celebrities are actually lizard people in disguise",
    "Secret military base found on the dark side of the moon",
    "EXPOSED: Major media companies paid to hide truth about vitamin C cancer cure",
    "Government insider reveals FEMA plans to put Americans in concentration camps",
    "PROOF that time travel exists and is being used to rig elections",
    "Scientists discover humans only use 10% of brain, elites using 100%",
    "BOMBSHELL: Princess Diana faked her death and is living in exile",
    "Doctors confirm drinking bleach kills coronavirus in 24 hours",
    "Pentagon whistleblower reveals United States has made contact with 5 alien races",
    "Secret society controlling all world governments EXPOSED by insider",
    "URGENT WARNING: New 5G phones upload your thoughts to government database",
    "SHOCKING PROOF: George Soros is paying protestors 500 dollars per day",
    "Ancient cure for all diseases suppressed by pharmaceutical industry for decades",
    "BREAKING: Elite pedophile ring operating out of pizza restaurant uncovered",
    "Scientists confirm that humans evolved from reptiles not primates COVER UP",
    "PROOF: Chemotherapy is a poison designed to kill rather than cure",
    "Secret tunnel network used by Illuminati discovered beneath major cities",
    "EXPOSED: Elections are completely rigged by global banking cartel",
    "Hollywood executive confirms all major movies contain secret Satanic messages",
    "BOMBSHELL: Tap water now contains DNA-altering nanobots released by elites",
    "Scientist who proved vaccines cause autism assassinated by government agents",
    "URGENT: New law will allow government to harvest organs from living citizens",
    "PROOF: The sun is a giant alien spacecraft providing energy to earth",
    "Deep state planning to trigger World War III to reduce population EXPOSED",
    "SHOCKING: Most celebrities were replaced by clones in the 1990s",
    "Secret service member CONFIRMS Obama is actually a CIA creation",
    "BREAKING NEWS: Microchip found in COVID vaccine that tracks your location",
    "Leaked NASA documents prove Mars was inhabited by humans 10,000 years ago",
    "EXPOSED: Major banks paying governments to keep cancer cures hidden",
    "URGENT: New social media AI designed to harvest and sell your soul",
    "Ancient Egyptian prophecy reveals identity of the Antichrist",
    "PROOF that the Titanic sinking was orchestrated by J.P. Morgan for insurance fraud",
    "Secret weather control machine used to create Hurricane Katrina EXPOSED",
]

FAKE_BODIES = [
    "According to an anonymous source who claims to have worked for a secret government agency, the moon is actually a massive hollow sphere constructed by an advanced alien civilization approximately 50,000 years ago. The source, whose identity cannot be verified, provided alleged photographs that experts say appear to be digitally manipulated.",
    "A leaked document allegedly from a government agency reveals that microscopic tracking chips have been inserted into coronavirus vaccines without public knowledge. The document, which has not been authenticated by any independent laboratory, claims officials are monitoring the movements and thoughts of vaccinated individuals.",
    "A naturopath in Arizona claims to have discovered a miracle compound derived from household spices that eliminates cancer cells in just 24 hours. The treatment, which has not undergone clinical trials or peer review, allegedly works by resetting the body natural frequency, a concept not recognized in mainstream medicine.",
    "A self-described genealogist claims to have uncovered birth records proving the former president was born on another planet and was placed on earth as part of an alien infiltration program. The documents, which appear on a website with no verifiable source information, have been dismissed by historians and legal experts.",
    "Multiple witnesses near Area 51 report seeing large spacecraft land in the Nevada desert last Tuesday, followed by what appeared to be a meeting between alien beings and men in black suits. The witnesses, who wish to remain anonymous, claim they were later visited by government agents who threatened them with imprisonment.",
    "A former telecommunications engineer, posting under a pseudonym on a fringe website, claims that hidden frequencies emitted by 5G towers are causing widespread reproductive harm. The post, which has been shared millions of times on social media, contains no peer-reviewed research and contradicts the scientific consensus on radio frequency safety.",
    "According to a doctored video circulating on social media, the technology billionaire made explicit statements about population reduction. Fact-checkers have confirmed the video was manipulated using artificial intelligence. The original footage shows no such statements being made.",
    "An anonymous blogger claims that explorers recently discovered a massive underground city in Antarctica containing advanced technology and architecture consistent with descriptions of ancient Germanic aesthetics. No verifiable evidence has been provided and no credible expedition has confirmed the discovery.",
    "A documentary filmmaker claims to have interviewed a now-deceased Hollywood lighting technician who allegedly worked on a secret government project to fake the Apollo missions. The filmmaker has refused to release the raw footage or identify any corroborating witnesses, and all physical evidence cited has been debunked by multiple independent experts.",
    "A conspiracy website claims that municipal water treatment facilities have been secretly instructed to add a compound that makes citizens more susceptible to government propaganda. The alleged compound does not exist according to chemists, and water treatment records are publicly available and do not support the claim.",
    "Share this before it gets deleted! The mainstream media is suppressing this story. Wake up and do your own research. This is what they do not want you to know. The truth is being hidden from the public. Sources: trust me bro. If this gets 10,000 shares maybe the truth will come out.",
    "An anonymous insider claiming to work deep within a shadow government agency has released classified documents proving that everything you know is a lie. The documents, which cannot be independently verified and have been dismissed as fabrications by multiple experts, allegedly reveal a decades-long conspiracy.",
    "A viral social media post, shared over 3 million times, claims to reveal the shocking truth that mainstream scientists have been hiding. The post, which contains no citations, peer-reviewed research, or verifiable sources, has been debunked by fact-checking organizations worldwide.",
    "According to a self-published book available only through a fringe website, whistleblowers from inside the deep state have confirmed that everything you thought you knew about reality is wrong. The author refuses to reveal their identity for safety reasons and provides no verifiable evidence.",
    "A video circulating on alternative social media platforms shows what appears to be a government official admitting to a massive cover-up. However, video forensic experts have confirmed the footage was created using deepfake technology and the statements attributed to the official are entirely fabricated.",
]

def generate_dataset(n_real=1500, n_fake=1500):
    """Generate a synthetic fake news dataset."""
    np.random.seed(42)

    def generate_article(headlines, bodies, label, n):
        articles = []
        for i in range(n):
            h_idx = i % len(headlines)
            b_idx = i % len(bodies)
            headline = headlines[h_idx]
            body = bodies[b_idx]
            if label == 1:
                if np.random.random() > 0.5:
                    headline = headline.upper()
                if np.random.random() > 0.7:
                    headline = headline + "!!!"
                body = body + " " + np.random.choice([
                    "Share this before it gets deleted!",
                    "The mainstream media is suppressing this story.",
                    "Wake up sheeple! Do your own research!",
                    "This is what THEY don't want you to know.",
                    "Spread this before the government censors it.",
                    "The truth is being hidden from you.",
                    "Sources: trust me bro. Do your own research.",
                    "If this gets 10,000 shares maybe the truth will come out.",
                ])
            text = headline + " " + body
            articles.append({
                'title': headline,
                'text': body,
                'full_text': text,
                'label': label,
                'label_name': 'FAKE' if label == 1 else 'REAL'
            })
        return articles

    real_articles = generate_article(REAL_HEADLINES, REAL_BODIES, 0, n_real)
    fake_articles = generate_article(FAKE_HEADLINES, FAKE_BODIES, 1, n_fake)
    all_articles = real_articles + fake_articles
    df = pd.DataFrame(all_articles)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ─── Text Preprocessing ────────────────────────────────────────────────────────

def preprocess_text(text):
    """Clean and preprocess text for ML models."""
    if not isinstance(text, str):
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
        tokens = [lemmatizer.lemmatize(token) for token in tokens
                  if token not in stop_words and len(token) > 2]
        return ' '.join(tokens)
    except Exception:
        words = text.split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(words)

# ─── Model Training ────────────────────────────────────────────────────────────

def train_models(df, models_dir='models', static_dir='static'):
    """Train multiple ML models and save the best one."""
    print("\n Preprocessing text data...")
    df['processed_text'] = df['full_text'].apply(preprocess_text)
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Passive Aggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Linear SVC': LinearSVC(max_iter=2000, random_state=42, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    print("\nTraining and evaluating models...\n" + "-" * 60)
    best_model_name = None
    best_accuracy = 0
    best_model = None

    for name, model in models.items():
        try:
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
                else:
                    y_prob = model.decision_function(X_test_tfidf)
                roc_auc = roc_auc_score(y_test, y_prob)
            except Exception:
                roc_auc = accuracy
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy, 'roc_auc': roc_auc,
                'report': report, 'confusion_matrix': cm.tolist(), 'model': model
            }
            print(f"OK  {name:<25} Accuracy: {accuracy:.4f}  ROC-AUC: {roc_auc:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_model = model
        except Exception as e:
            print(f"FAIL {name:<25} Failed: {e}")

    print("-" * 60)
    print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer.pkl'))
    for name, res in results.items():
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(res['model'], os.path.join(models_dir, f'{safe_name}.pkl'))

    metadata = {
        'best_model': best_model_name,
        'best_accuracy': float(best_accuracy),
        'models': {
            name: {
                'accuracy': float(res['accuracy']),
                'roc_auc': float(res['roc_auc']),
                'precision_real': float(res['report'].get('0', {}).get('precision', 0)),
                'recall_real': float(res['report'].get('0', {}).get('recall', 0)),
                'precision_fake': float(res['report'].get('1', {}).get('precision', 0)),
                'recall_fake': float(res['report'].get('1', {}).get('recall', 0)),
                'f1_real': float(res['report'].get('0', {}).get('f1-score', 0)),
                'f1_fake': float(res['report'].get('1', {}).get('f1-score', 0)),
                'confusion_matrix': res['confusion_matrix'],
            }
            for name, res in results.items()
        },
        'dataset': {
            'total_samples': len(df),
            'real_count': int((df['label'] == 0).sum()),
            'fake_count': int((df['label'] == 1).sum()),
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
    }
    with open(os.path.join(models_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    os.makedirs(static_dir, exist_ok=True)
    generate_charts(results, df, static_dir)

    print(f"\nModels saved to '{models_dir}/'")
    print(f"Charts saved to '{static_dir}/'")
    return results, best_model_name, vectorizer, best_model


def generate_charts(results, df, static_dir):
    """Generate visualization charts for the web dashboard."""
    plt.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.labelcolor': '#e6edf3', 'xtick.color': '#8b949e',
        'ytick.color': '#8b949e', 'text.color': '#e6edf3',
        'grid.color': '#30363d', 'axes.edgecolor': '#30363d',
    })

    # 1. Model Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] * 100 for n in names]
    roc_aucs = [results[n]['roc_auc'] * 100 for n in names]
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy %', color='#238636', alpha=0.9)
    bars2 = ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC %', color='#1f6feb', alpha=0.9)
    ax.set_xlabel('Models', fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', color='#f0f6fc')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.legend(facecolor='#21262d', edgecolor='#30363d')
    ax.set_ylim(50, 105)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8, color='#7ee787')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'model_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # 2. Confusion Matrix
    best_name = max(results, key=lambda n: results[n]['accuracy'])
    cm = np.array(results[best_name]['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
                ax=ax, linewidths=0.5, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(f'Confusion Matrix - {best_name}', fontsize=13, fontweight='bold', color='#f0f6fc')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # 3. Dataset Distribution Pie
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_pie = ['Real News', 'Fake News']
    sizes = [(df['label'] == 0).sum(), (df['label'] == 1).sum()]
    colors = ['#238636', '#da3633']
    wedges, texts, autotexts = ax.pie(
        sizes, explode=(0.05, 0.05), labels=labels_pie, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140,
        textprops={'color': '#e6edf3'}, wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')
    ax.set_title('Dataset Distribution', fontsize=14, fontweight='bold', color='#f0f6fc')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'dataset_distribution.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # 4. F1-Score Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    names_list = list(results.keys())
    f1_real = [results[n]['report'].get('0', {}).get('f1-score', 0) * 100 for n in names_list]
    f1_fake = [results[n]['report'].get('1', {}).get('f1-score', 0) * 100 for n in names_list]
    x = np.arange(len(names_list))
    ax.bar(x - 0.2, f1_real, 0.35, label='F1-Score (Real)', color='#238636', alpha=0.9)
    ax.bar(x + 0.2, f1_fake, 0.35, label='F1-Score (Fake)', color='#da3633', alpha=0.9)
    ax.set_xlabel('Models', fontsize=11)
    ax.set_ylabel('F1-Score (%)', fontsize=11)
    ax.set_title('F1-Score Comparison by Class', fontsize=14, fontweight='bold', color='#f0f6fc')
    ax.set_xticks(x)
    ax.set_xticklabels(names_list, rotation=15, ha='right', fontsize=9)
    ax.legend(facecolor='#21262d', edgecolor='#30363d')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'f1_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("Charts generated successfully!")


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    print("=" * 60)
    print("  FAKE NEWS DETECTOR - Model Training")
    print("=" * 60)

    print("\nGenerating synthetic training dataset...")
    df = generate_dataset(n_real=1500, n_fake=1500)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, 'dataset.csv'), index=False)
    print(f"Dataset generated: {len(df)} samples ({(df['label']==0).sum()} real, {(df['label']==1).sum()} fake)")

    results, best_model_name, vectorizer, best_model = train_models(df, MODELS_DIR, STATIC_DIR)

    print("\n" + "=" * 60)
    print("  Training Complete! Run app.py to start the web server.")
    print("=" * 60)
