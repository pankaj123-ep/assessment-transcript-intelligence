"""
Transcript Intelligence Platform
================================

A lightweight transcript analytics pipeline designed for restricted
execution environments.

Key capabilities:
1. Transcript ingestion
2. Cleaning & preprocessing
3. TF-IDF vectorization
4. Dynamic topic clustering
5. Rule-assisted tagging
6. Sentiment analysis
7. Trend analytics
8. Stakeholder insights
9. CSV + chart exports
10. Automated tests

This implementation intentionally avoids heavy ML dependencies.
"""

# =====================================================
# INSTALLATION (RUN IN TERMINAL)
# =====================================================

"""
pip install pandas numpy scikit-learn matplotlib
"""

# =========================
# IMPORTS
# =========================

import os
import re
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =========================
# CONFIGURATION
# =========================

DATA_PATH = "./data/transcripts.csv"
OUTPUT_DIR = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)


# =========================
# DATA LOADING
# =========================


def load_transcripts(path: str) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Transcript file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "transcript_id",
        "call_type",
        "transcript"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )

    return df


# =========================
# TEXT CLEANING
# =========================


FILLER_WORDS = [
    "um",
    "uh",
    "you know",
    "like",
    "sort of",
    "kind of"
]



def clean_text(text: str) -> str:

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"\d{1,2}:\d{2}:\d{2}", " ", text)

    text = re.sub(
        r"agent:|customer:|rep:|prospect:|pm:|engineer:",
        " ",
        text
    )

    for filler in FILLER_WORDS:
        text = re.sub(rf"\b{re.escape(filler)}\b", " ", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# VECTOR ENGINE
# =========================


class EmbeddingEngine:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=500
        )

    def generate(self, texts):

        vectors = self.vectorizer.fit_transform(texts)

        return vectors.toarray()


# =========================
# TOPIC MODELING
# =========================


class TopicModeler:

    def __init__(self, requested_topics=5):

        self.requested_topics = requested_topics
        self.model = None
        self.actual_topics = None

    def determine_topic_count(self, embeddings):

        sample_count = len(embeddings)

        # silhouette_score requires:
        # 2 <= num_clusters <= n_samples - 1

        if sample_count <= 2:
            return 1

        safe_max = sample_count - 1

        return min(self.requested_topics, safe_max)

    def fit(self, embeddings):

        topic_count = self.determine_topic_count(
            embeddings
        )

        self.actual_topics = topic_count

        if topic_count <= 1:
            return np.zeros(len(embeddings), dtype=int)

        self.model = KMeans(
            n_clusters=topic_count,
            random_state=42,
            n_init=10
        )

        topics = self.model.fit_predict(embeddings)

        return topics

    def evaluate(self, embeddings, topics):

        unique_topics = set(topics)

        sample_count = len(embeddings)

        if len(unique_topics) < 2:
            return 0.0

        if len(unique_topics) >= sample_count:
            return 0.0

        try:
            score = silhouette_score(
                embeddings,
                topics
            )

            return round(float(score), 3)

        except Exception:
            return 0.0


# =========================
# SENTIMENT ANALYSIS
# =========================


POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "resolved",
    "fixed",
    "successful",
    "happy"
}

NEGATIVE_WORDS = {
    "error",
    "issue",
    "problem",
    "crash",
    "failed",
    "billing",
    "frustrated",
    "locked",
    "slow",
    "bug"
}


class SentimentAnalyzer:

    @staticmethod
    def analyze(texts):

        results = []

        for text in texts:

            tokens = set(text.split())

            positive_score = len(
                tokens.intersection(POSITIVE_WORDS)
            )

            negative_score = len(
                tokens.intersection(NEGATIVE_WORDS)
            )

            if negative_score > positive_score:
                label = "NEGATIVE"
                score = negative_score

            elif positive_score > negative_score:
                label = "POSITIVE"
                score = positive_score

            else:
                label = "NEUTRAL"
                score = 0

            results.append({
                "label": label,
                "score": score
            })

        return results


# =========================
# RULE TAGGING
# =========================


TOPIC_RULES = {
    "authentication": [
        "login",
        "password",
        "authentication",
        "mfa",
        "reset"
    ],

    "billing": [
        "invoice",
        "payment",
        "billing",
        "charged",
        "renewal"
    ],

    "bug_issue": [
        "bug",
        "crash",
        "error",
        "issue",
        "failing"
    ],

    "feature_request": [
        "feature",
        "request",
        "wish",
        "would like"
    ],

    "pricing_objection": [
        "pricing",
        "budget",
        "expensive",
        "cost",
        "cheaper"
    ],

    "competitor": [
        "competitor",
        "vendor",
        "alternative",
        "migration"
    ]
}



def extract_rule_topics(text):

    matches = []

    for topic, keywords in TOPIC_RULES.items():

        for keyword in keywords:

            if keyword in text:
                matches.append(topic)
                break

    return matches


# =========================
# SUMMARIZER
# =========================


class TranscriptSummarizer:

    @staticmethod
    def summarize(text, max_words=20):

        words = text.split()

        return " ".join(words[:max_words])


# =========================
# VISUALIZATION
# =========================


class TrendAnalyzer:

    @staticmethod
    def topic_distribution(df):

        topic_counts = df[
            "primary_topic"
        ].value_counts()

        plt.figure(figsize=(10, 5))

        topic_counts.plot(kind="bar")

        plt.title("Topic Distribution")

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            "topic_distribution.png"
        )

        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def sentiment_distribution(df):

        sentiment_counts = df[
            "sentiment_label"
        ].value_counts()

        plt.figure(figsize=(8, 5))

        sentiment_counts.plot(kind="bar")

        plt.title("Sentiment Distribution")

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            "sentiment_distribution.png"
        )

        plt.savefig(output_path)
        plt.close()


# =========================
# INSIGHTS
# =========================


class InsightEngine:

    @staticmethod
    def generate_insights(df):

        insights = {}

        support_df = df[
            df["call_type"] == "support"
        ]

        if not support_df.empty:
            insights["top_support_topics"] = (
                support_df["primary_topic"]
                .value_counts()
                .head(3)
                .to_dict()
            )

        insights["feature_requests"] = int(
            df["rule_topics"]
            .astype(str)
            .str.contains("feature_request")
            .sum()
        )

        insights["pricing_objections"] = int(
            df["rule_topics"]
            .astype(str)
            .str.contains("pricing_objection")
            .sum()
        )

        return insights


# =========================
# PIPELINE
# =========================


class TranscriptIntelligencePipeline:

    def __init__(self):

        self.embedding_engine = EmbeddingEngine()

        self.topic_modeler = TopicModeler(
            requested_topics=5
        )

        self.sentiment_analyzer = SentimentAnalyzer()
        self.summarizer = TranscriptSummarizer()

    def run(self, df):

        print("\nSTEP 1 — Cleaning transcripts")

        df["cleaned_transcript"] = df[
            "transcript"
        ].apply(clean_text)

        print("\nSTEP 2 — TF-IDF vectorization")

        embeddings = self.embedding_engine.generate(
            df["cleaned_transcript"].tolist()
        )

        print("\nSTEP 3 — Topic clustering")

        topics = self.topic_modeler.fit(embeddings)

        df["cluster_topic"] = topics

        df["primary_topic"] = [
            f"Topic_{int(topic)}"
            for topic in topics
        ]

        print("\nSTEP 4 — Rule-based tagging")

        df["rule_topics"] = df[
            "cleaned_transcript"
        ].apply(extract_rule_topics)

        print("\nSTEP 5 — Sentiment analysis")

        sentiment_results = self.sentiment_analyzer.analyze(
            df["cleaned_transcript"].tolist()
        )

        df["sentiment_label"] = [
            item["label"]
            for item in sentiment_results
        ]

        df["sentiment_score"] = [
            item["score"]
            for item in sentiment_results
        ]

        print("\nSTEP 6 — Summarization")

        df["summary"] = df[
            "cleaned_transcript"
        ].apply(self.summarizer.summarize)

        print("\nSTEP 7 — Cluster evaluation")

        score = self.topic_modeler.evaluate(
            embeddings,
            topics
        )

        print(f"Silhouette Score: {score}")

        print("\nSTEP 8 — Generating insights")

        insights = InsightEngine.generate_insights(df)

        with open(
            os.path.join(
                OUTPUT_DIR,
                "insights.json"
            ),
            "w"
        ) as f:
            json.dump(insights, f, indent=2)

        print("\nSTEP 9 — Creating charts")

        TrendAnalyzer.topic_distribution(df)
        TrendAnalyzer.sentiment_distribution(df)

        print("\nSTEP 10 — Saving outputs")

        output_csv = os.path.join(
            OUTPUT_DIR,
            "processed_transcripts.csv"
        )

        df.to_csv(output_csv, index=False)

        print("\nPipeline complete")

        return df


# =========================
# SAMPLE DATA
# =========================



def generate_sample_transcripts(
    output_path="./data/transcripts.csv"
):

    sample_data = [

        {
            "transcript_id": "T001",
            "call_type": "support",
            "date": "2026-01-01",
            "transcript": (
                "Customer cannot log in after password reset."
            )
        },

        {
            "transcript_id": "T002",
            "call_type": "support",
            "date": "2026-01-02",
            "transcript": (
                "Customer reports billing issue and duplicate invoice."
            )
        },

        {
            "transcript_id": "T003",
            "call_type": "sales",
            "date": "2026-01-03",
            "transcript": (
                "Prospect says competitor pricing is cheaper."
            )
        },

        {
            "transcript_id": "T004",
            "call_type": "product",
            "date": "2026-01-04",
            "transcript": (
                "Customers requested new reporting feature."
            )
        },

        {
            "transcript_id": "T005",
            "call_type": "support",
            "date": "2026-01-05",
            "transcript": (
                "Application crash causing export issue."
            )
        }
    ]

    df = pd.DataFrame(sample_data)

    df.to_csv(output_path, index=False)

    return df


# =========================
# TESTS
# =========================



def run_tests():

    print("\nRunning tests...")

    # Clean text
    cleaned = clean_text(
        "Agent: Uh login failed at 12:01:22"
    )

    assert "uh" not in cleaned
    assert "12 01 22" not in cleaned

    # Rule extraction
    topics = extract_rule_topics(
        "Customer reports billing payment issue"
    )

    assert "billing" in topics

    # Sentiment tests
    sentiments = SentimentAnalyzer.analyze([
        "system crash issue",
        "great successful resolution"
    ])

    assert sentiments[0]["label"] == "NEGATIVE"
    assert sentiments[1]["label"] == "POSITIVE"

    # Small dataset clustering test
    small_embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1]
    ])

    modeler = TopicModeler(requested_topics=5)

    topics = modeler.fit(small_embeddings)

    score = modeler.evaluate(
        small_embeddings,
        topics
    )

    assert score == 0.0

    # Sample generator test
    sample_df = generate_sample_transcripts(
        "./data/test_transcripts.csv"
    )

    assert len(sample_df) == 5

    print("All tests passed")


# =========================
# EXECUTION
# =========================


if __name__ == "__main__":

    if not os.path.exists(DATA_PATH):
        generate_sample_transcripts(DATA_PATH)

    run_tests()

    df = load_transcripts(DATA_PATH)

    pipeline = TranscriptIntelligencePipeline()

    processed_df = pipeline.run(df)

    print("\nArtifacts generated:")
    print(f"- {OUTPUT_DIR}/processed_transcripts.csv")
    print(f"- {OUTPUT_DIR}/insights.json")
    print(f"- {OUTPUT_DIR}/topic_distribution.png")
    print(f"- {OUTPUT_DIR}/sentiment_distribution.png")
