# Transcript Intelligence Platform

A lightweight, dependency-safe Transcript Intelligence pipeline for analyzing customer conversations, support tickets, sales calls, and product discussions.

This project processes raw transcript data and generates insights without requiring GPU-heavy NLP libraries—perfect for restricted Python environments and sandboxed execution.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Design Decisions](#design-decisions)
- [Use Cases](#use-cases)
- [Future Improvements](#future-improvements)
- [Requirements](#requirements)

## Quick Start

Get the pipeline running in 3 steps:

```bash
# 1. Clone the repository
git clone <repository-url>
cd assessment-transcript-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python transcript_intelligence.py
```

The pipeline automatically generates sample transcripts, runs validation tests, and exports results to the `outputs/` directory.

## Features

### 1. 📥 Transcript Ingestion

Loads transcript datasets from CSV files with support for:

| Column | Description |
|--------|-------------|
| `transcript_id` | Unique transcript identifier |
| `call_type` | support / sales / product |
| `transcript` | Raw transcript text |
| `date` | Call date |

### 2. 🧹 Text Cleaning & Preprocessing

Automatically normalizes transcripts by:

- ✓ Removing timestamps
- ✓ Removing speaker labels
- ✓ Removing filler words
- ✓ Removing punctuation
- ✓ Normalizing whitespace
- ✓ Converting to lowercase

**Example:**

```
Raw:      Agent: Uh customer login failed at 12:01:22
Cleaned:  customer login failed
```

### 3. 🔢 TF-IDF Embedding Generation

Uses lightweight sklearn embeddings instead of heavy transformer models:

- **TfidfVectorizer** for efficient text vectorization
- Sparse vector embeddings for memory efficiency
- Fast processing on CPU

**Why TF-IDF?**
- ⚡ Fast and lightweight
- 🔒 Sandbox-safe execution
- 💻 CPU-friendly (no GPU required)
- 📦 Minimal dependencies

### 4. 🎯 Dynamic Topic Clustering

Intelligent topic clustering using KMeans with:

- Dynamic cluster sizing
- Safe silhouette evaluation
- Automatic validation for small datasets

**Example Topic Outputs:**
- Authentication Issues
- Billing Problems
- Feature Requests
- Pricing Objections
- Product Bugs

### 5. 🏷️ Rule-Based Topic Tagging

Deterministic tagging supplements ML clustering for better business interpretability:

| Topic | Keywords |
|-------|----------|
| `authentication` | login, password, mfa |
| `billing` | invoice, payment, charged |
| `bug_issue` | crash, error, issue |
| `feature_request` | feature, request |
| `pricing_objection` | pricing, budget |
| `competitor` | competitor, vendor |

### 6. 😊 Sentiment Analysis

Classifies transcripts into three sentiment categories:

- **POSITIVE** - Satisfied customers
- **NEGATIVE** - Issues or complaints
- **NEUTRAL** - Informational content

Uses keyword scoring instead of transformer inference for maximum compatibility.

**Example:**
```
Input:  "Application crash causing export issue"
Output: NEGATIVE
```

### 7. 📝 Transcript Summarization

Generates concise summaries using extractive truncation:

```
"Customer unable to log in after password reset, tried multiple browsers"
→ "Customer cannot log in after password reset"
```

### 8. 📊 Trend Analytics

Automatically generates visualization charts:

- **topic_distribution.png** - Topic frequency distribution
- **sentiment_distribution.png** - Overall sentiment trends

### 9. 🎯 Stakeholder Insights

Business-oriented analytics tailored for different teams:

**Support Leaders**
- Most common support problems
- Escalation trends
- Negative sentiment tracking

**Sales Teams**
- Pricing objections
- Competitor mentions
- Budget concerns

**Product Managers**
- Feature requests
- Product pain points
- Bug frequency

**Engineering Teams**
- Crash patterns
- Reliability indicators
- Deployment-related issues

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd assessment-transcript-intelligence
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

## Usage

### Basic Execution

Run the complete pipeline:

```bash
python transcript_intelligence.py
```

### Code Examples

#### Generate Sample Transcripts

```python
from transcript_intelligence import generate_sample_transcripts

# Generate 50 sample transcripts
sample_data = generate_sample_transcripts(num_samples=50)
```

#### Load and Process Transcripts

```python
import pandas as pd
from transcript_intelligence import TranscriptProcessor

# Load your CSV
df = pd.read_csv('data/transcripts.csv')

# Initialize processor
processor = TranscriptProcessor()

# Process transcripts
processed = processor.clean_and_analyze(df)
```

#### Access Insights

```python
from transcript_intelligence import InsightEngine

# Generate stakeholder insights
insights = InsightEngine.generate_insights(processed_data)

# Access specific metrics
print(insights['top_support_topics'])
print(insights['feature_requests'])
print(insights['pricing_objections'])
```

### Automatic Workflow

When executed, the pipeline automatically:

1. ✓ Generates sample transcripts if missing
2. ✓ Runs validation tests
3. ✓ Cleans transcript data
4. ✓ Generates TF-IDF embeddings
5. ✓ Performs topic clustering
6. ✓ Applies rule-based tags
7. ✓ Runs sentiment analysis
8. ✓ Creates summaries
9. ✓ Generates insights
10. ✓ Exports charts and CSV outputs

## Project Structure

```
assessment-transcript-intelligence/
│
├── data/
│   ├── transcripts.csv              # Your transcript data
│   └── test_transcripts.csv         # Sample test data
│
├── outputs/
│   ├── processed_transcripts.csv    # Cleaned & analyzed transcripts
│   ├── insights.json                # Business insights export
│   ├── topic_distribution.png       # Topic frequency chart
│   └── sentiment_distribution.png   # Sentiment trends chart
│
├── transcript_intelligence.py       # Main pipeline
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Output Files

### processed_transcripts.csv

Enhanced transcript data containing:

- Cleaned transcript text
- Cluster topic assignments
- Sentiment labels (POSITIVE/NEGATIVE/NEUTRAL)
- Generated summaries
- Extracted rule-based tags

**Example:**
```csv
transcript_id,cleaned_text,topic,sentiment,summary,tags
1,"customer login failed",Topic_0,NEGATIVE,"Customer cannot log in after password reset","['authentication']"
2,"billing issue resolved",Topic_1,POSITIVE,"Billing dispute resolved quickly","['billing']"
```

### insights.json

Business intelligence summary:

```json
{
  "top_support_topics": {
    "Topic_0": 3,
    "Topic_1": 2
  },
  "feature_requests": 5,
  "pricing_objections": 2,
  "bug_reports": 8,
  "sentiment_summary": {
    "positive": 15,
    "negative": 8,
    "neutral": 12
  }
}
```

### Visualization Outputs

- **topic_distribution.png** - Bar chart of topic frequencies
- **sentiment_distribution.png** - Pie chart of sentiment distribution

## Design Decisions

### Why TF-IDF Instead of Transformers?

| Factor | Benefit |
|--------|---------|
| **Performance** | Fast processing without GPU |
| **Deployment** | Works in restricted/sandboxed environments |
| **Dependencies** | Minimal external requirements |
| **Reliability** | No runtime failures from missing NLP packages |
| **Memory** | Sparse vectors use less RAM |

### Why KMeans Instead of BERTopic?

- ✅ Simpler deployment (no HDBSCAN dependency)
- ✅ Stable execution in sandboxed environments
- ✅ Easier reproducibility across systems
- ✅ Automatic safe handling of small datasets

### Why Hybrid Rule + ML Approach?

Pure clustering can be difficult for business users to interpret. The hybrid approach:

- 📊 Improves explainability with deterministic rules
- 🎯 Enables business context (e.g., always tag "login" as authentication)
- 📈 Makes dashboard integration straightforward
- 🔍 Balances automation with human interpretability

## Built-In Tests

The project includes automated test coverage for:

- ✓ Text cleaning functions
- ✓ Rule extraction accuracy
- ✓ Sentiment analysis
- ✓ Small dataset clustering validation
- ✓ Sample dataset generation

Tests run automatically during pipeline execution.

## Use Cases

### 📞 Customer Support

Detect recurring login failures or billing complaints to identify systemic issues:
```
Common Issues: "customer login failed", "password reset not working"
Action: Escalate to engineering, create FAQ article
```

### 💰 Sales Intelligence

Track competitor mentions and pricing objections to inform strategy:
```
Objections: "Your pricing is 30% higher than Competitor X"
Action: Train sales team on value proposition
```

### 🛠️ Product Analytics

Identify feature requests and usability pain points:
```
Requests: "Need bulk export feature", "Dark mode support"
Action: Add to product roadmap
```

### 🔧 Engineering Operations

Detect crash-related incidents and reliability problems:
```
Patterns: "Application crashes on Chrome", "Export fails with large files"
Action: Debug, test, deploy fixes
```

## Future Improvements

Potential enhancements planned for future versions:

- 🔗 **OpenAI Embeddings** - Better semantic understanding
- 🤖 **LLM-based Summarization** - More natural summaries
- ⚡ **Real-time Streaming** - Process transcripts as they arrive
- 📉 **Churn Prediction** - Identify at-risk customers
- ⭐ **Agent Quality Scoring** - Evaluate support agent performance
- 🔍 **RAG Search** - Semantic search over transcripts
- 📈 **Temporal Analysis** - Track trends over time
- 🚨 **Anomaly Detection** - Identify unusual conversation patterns
- 🎨 **Dashboard UI** - Visual analytics interface
- 🗄️ **Vector Database** - Scale to millions of transcripts

## Troubleshooting

### Common Issues

**Issue: "No module named 'pandas'"**
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt --upgrade
```

**Issue: "Small dataset warning" appears**
- This is expected for datasets with fewer than 10 transcripts
- Clustering is automatically adjusted to safe configurations

**Issue: Output files not generated**
- Check that `outputs/` directory exists or is writable
- Verify CSV file format matches expected columns

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Last Updated:** 2026-05-20  
**Maintained by:** pankaj123-ep
