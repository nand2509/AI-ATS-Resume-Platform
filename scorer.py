import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


def compute_detailed_score(
    resume_text: str,
    jd_text: str,
    matched_skills: list,
    jd_skills: list,
    w_similarity: float = 0.50,
    w_skills: float = 0.35,
    w_keywords: float = 0.15,
) -> dict:
    """
    Returns a dict with:
      - overall       : weighted composite score (0–100)
      - semantic      : TF-IDF cosine similarity (0–100)
      - skill_coverage: matched skills / JD skills (0–100)
      - keyword_overlap: common keyword ratio (0–100)
    """

    # 1. Semantic similarity via TF-IDF cosine
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        semantic_raw = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        semantic = float(np.clip(semantic_raw * 100, 0, 100))
    except Exception:
        semantic = 0.0

    # 2. Skill coverage
    if jd_skills:
        skill_coverage = len(matched_skills) / len(jd_skills) * 100
    else:
        skill_coverage = 100.0

    # 3. Keyword overlap (bag-of-words intersection ratio)
    r_words = set(resume_text.split())
    j_words = set(jd_text.split())
    if j_words:
        keyword_overlap = len(r_words & j_words) / len(j_words) * 100
        keyword_overlap = float(np.clip(keyword_overlap, 0, 100))
    else:
        keyword_overlap = 0.0

    # Normalize weights
    total_w = w_similarity + w_skills + w_keywords
    if total_w == 0:
        total_w = 1.0
    ws = w_similarity / total_w
    wsk = w_skills / total_w
    wkw = w_keywords / total_w

    overall = ws * semantic + wsk * skill_coverage + wkw * keyword_overlap
    overall = float(np.clip(overall, 0, 100))

    return {
        "overall":         round(overall, 1),
        "semantic":        round(semantic, 1),
        "skill_coverage":  round(skill_coverage, 1),
        "keyword_overlap": round(keyword_overlap, 1),
    }