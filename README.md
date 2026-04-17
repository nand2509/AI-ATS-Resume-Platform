# 🎯 ResumeIQ – Advanced ATS Resume Analyzer

A production-grade Streamlit app for AI-powered resume analysis.

## Features
- **Semantic Similarity** – TF-IDF cosine similarity between resume & JD
- **Skill Gap Analysis** – 10 skill categories, matched/missing/bonus skills
- **Multi-Dimensional Scoring** – Adjustable weights for 3 scoring dimensions
- **Visualizations** – Gauge, radar, bar charts, category heatmaps, keyword frequency
- **Actionable Insights** – Personalized tips based on your score
- **Keyword Analysis** – Top keywords from resume vs JD with overlap view

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Project Structure
```
resume-analyzer/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── data/
│   └── skills.json         # Categorized skills database (10 categories)
└── utils/
    ├── __init__.py
    ├── parser.py           # PDF text extraction
    ├── cleaner.py          # Text cleaning + keyword extraction
    ├── extractor.py        # Skill extraction by category
    ├── scorer.py           # Multi-dimensional scoring engine
    └── insights.py         # Personalized insight generation
```

## Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app.py` → Deploy