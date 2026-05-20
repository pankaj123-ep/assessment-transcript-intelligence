Transcript Intelligence Platform

A lightweight, dependency-safe Transcript Intelligence pipeline for analyzing customer conversations, support tickets, sales calls, and product discussions.

This project processes raw transcript data and generates:

Topic clustering
Sentiment analysis
Rule-based tagging
Stakeholder insights
Trend visualizations
Summaries and analytics exports

The implementation is intentionally lightweight and designed to run in restricted Python environments without requiring GPU-heavy NLP libraries.

Features
1. Transcript Ingestion

Loads transcript datasets from CSV files.

Supported columns:

Column	Description
transcript_id	Unique transcript identifier
call_type	support / sales / product
transcript	Raw transcript text
date	Call date
2. Text Cleaning & Preprocessing

The pipeline automatically:

Removes timestamps
Removes speaker labels
Removes filler words
Removes punctuation
Normalizes whitespace
Converts text to lowercase

Example:

Raw
Agent: Uh customer login failed at 12:01:22
Cleaned
customer login failed
3. TF-IDF Embedding Generation

Instead of heavy transformer models, this project uses:

TfidfVectorizer
Sparse vector embeddings
Lightweight sklearn pipeline

Benefits:

Fast
Sandbox-safe
CPU-friendly
No GPU required
4. Dynamic Topic Clustering

Topic clustering is performed using:

KMeans
Dynamic cluster sizing
Safe silhouette evaluation

The system automatically prevents invalid clustering configurations for small datasets.

Example topic outputs:

Authentication Issues
Billing Problems
Feature Requests
Pricing Objections
Product Bugs
5. Rule-Based Topic Tagging

The pipeline supplements clustering with deterministic tagging.

Supported Tags
Topic	Keywords
authentication	login, password, mfa
billing	invoice, payment, charged
bug_issue	crash, error, issue
feature_request	feature, request
pricing_objection	pricing, budget
competitor	competitor, vendor

This hybrid approach improves interpretability for business stakeholders.

6. Sentiment Analysis

A lightweight sentiment engine classifies transcripts as:

POSITIVE
NEGATIVE
NEUTRAL

The system uses keyword scoring instead of transformer inference for maximum compatibility.

Example:

"Application crash causing export issue"
→ NEGATIVE
7. Transcript Summarization

The platform generates concise transcript summaries using extractive truncation.

Example:

Customer cannot log in after password reset
8. Trend Analytics

The pipeline automatically generates:

Topic distribution charts
Sentiment distribution charts

Generated outputs:

topic_distribution.png
sentiment_distribution.png
9. Stakeholder Insights

The Insight Engine generates business-oriented analytics.

Support Leaders
Most common support problems
Escalation trends
Negative sentiment tracking
Sales Teams
Pricing objections
Competitor mentions
Budget concerns
Product Managers
Feature requests
Product pain points
Bug frequency
Engineering Teams
Crash patterns
Reliability indicators
Deployment-related issues
Project Structure
project/
│
├── data/
│   ├── transcripts.csv
│   └── test_transcripts.csv
│
├── outputs/
│   ├── processed_transcripts.csv
│   ├── insights.json
│   ├── topic_distribution.png
│   └── sentiment_distribution.png
│
├── transcript_intelligence.py
├── requirements.txt
└── README.md
Installation
1. Clone the Repository
git clone <repository-url>
cd transcript-intelligence
2. Install Dependencies
pip install -r requirements.txt
Requirements
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
Running the Pipeline
Execute the Project
python transcript_intelligence.py
Automatic Workflow

When executed, the pipeline automatically:

Generates sample transcripts if missing
Runs validation tests
Cleans transcript data
Generates TF-IDF embeddings
Performs topic clustering
Applies rule-based tags
Runs sentiment analysis
Creates summaries
Generates insights
Exports charts and CSV outputs
Sample Dataset Generation

The system includes a built-in synthetic dataset generator.

Generated examples include:

Login failures
Billing disputes
Product crashes
Pricing objections
Feature requests

Example call:

generate_sample_transcripts()
Output Files
processed_transcripts.csv

Contains:

cleaned transcripts
cluster topics
sentiment labels
summaries
extracted rule tags
insights.json

Business insight summaries such as:

{
  "top_support_topics": {
    "Topic_0": 3,
    "Topic_1": 2
  },
  "feature_requests": 5,
  "pricing_objections": 2
}
Visualization Outputs
Topic Distribution

Shows frequency of clustered topics.

Sentiment Distribution

Shows overall customer sentiment trends.

Built-In Tests

The project includes automated tests for:

Text cleaning
Rule extraction
Sentiment analysis
Small dataset clustering
Sample dataset generation

Run automatically during execution.

Design Decisions
Why TF-IDF Instead of Transformers?

Reason:

Lightweight
Fast
No GPU dependency
Works in restricted environments
Avoids runtime failures from unavailable NLP packages
Why KMeans Instead of BERTopic?

Reason:

Simpler deployment
No HDBSCAN dependency
Stable in sandboxed execution
Easier reproducibility
Why Hybrid Rule + ML Approach?

Pure clustering can be difficult for business users to interpret.

Adding rule-based tags:

Improves explainability
Enables deterministic categorization
Makes dashboards easier to understand
Future Improvements

Potential upgrades:

OpenAI embeddings
LLM-based summarization
Real-time transcript streaming
Churn prediction
Agent quality scoring
RAG search over transcripts
Temporal trend analysis
Conversation anomaly detection
Dashboard UI
Vector database integration
Example Use Cases
Customer Support

Detect recurring login failures or billing complaints.

Sales Intelligence

Track competitor mentions and pricing objections.

Product Analytics

Identify feature requests and usability pain points.

Engineering Operations

Detect crash-related incidents and reliability problems.
