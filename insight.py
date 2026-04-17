from utils.insights import generate_insights

def generate_insights(
    overall: float,
    matched: list,
    missing: list,
    bonus: list,
    scores: dict,
) -> list[dict]:
    """
    Generate a list of insight dicts: { type, icon, text }
    type ∈ { success, warning, danger, info }
    """
    insights = []

    # ── Overall score ──────────────────────────────────────────────────────
    if overall >= 80:
        insights.append({
            "type": "success", "icon": "🚀",
            "text": f"Excellent! Your resume scores <strong>{overall:.0f}%</strong> — this is above the typical ATS cutoff of 70%. You're highly competitive for this role."
        })
    elif overall >= 65:
        insights.append({
            "type": "warning", "icon": "👍",
            "text": f"Good match at <strong>{overall:.0f}%</strong>. You'll likely pass automated screening, but targeted improvements could push you above 80%."
        })
    elif overall >= 45:
        insights.append({
            "type": "warning", "icon": "⚠️",
            "text": f"Moderate match at <strong>{overall:.0f}%</strong>. You may be filtered out by strict ATS systems. Focus on adding missing skills and aligning your language with the JD."
        })
    else:
        insights.append({
            "type": "danger", "icon": "🔴",
            "text": f"Low match at <strong>{overall:.0f}%</strong>. Your resume is not well aligned with this job description. Significant keyword and skill additions are needed."
        })

    # ── Semantic similarity ────────────────────────────────────────────────
    sem = scores["semantic"]
    if sem < 30:
        insights.append({
            "type": "danger", "icon": "📝",
            "text": "Your resume language is very different from the job description. Try using the same terminology as the JD (e.g., if they say 'data pipeline' — use that exact phrase)."
        })
    elif sem >= 65:
        insights.append({
            "type": "success", "icon": "✍️",
            "text": "Strong language alignment with the JD. Your resume vocabulary closely mirrors the role requirements."
        })

    # ── Skill coverage ─────────────────────────────────────────────────────
    sk = scores["skill_coverage"]
    if sk >= 85:
        insights.append({
            "type": "success", "icon": "🎯",
            "text": f"Excellent skill coverage: <strong>{sk:.0f}%</strong> of required skills are present in your resume."
        })
    elif sk >= 55:
        insights.append({
            "type": "warning", "icon": "🛠️",
            "text": f"You have <strong>{sk:.0f}%</strong> skill coverage. Add the {len(missing)} missing skills to your resume if you have them."
        })
    else:
        insights.append({
            "type": "danger", "icon": "📚",
            "text": f"Only <strong>{sk:.0f}%</strong> skill coverage. You're missing <strong>{len(missing)}</strong> skills the job requires. Consider upskilling or rephrasing existing experience."
        })

    # ── Missing skills ─────────────────────────────────────────────────────
    if missing:
        top_missing = ", ".join(f"<code>{s}</code>" for s in missing[:5])
        insights.append({
            "type": "warning", "icon": "❌",
            "text": f"Top missing skills: {top_missing}. Even if you have experience with these, make sure they appear in your resume text."
        })

    # ── Bonus skills ───────────────────────────────────────────────────────
    if len(bonus) >= 3:
        insights.append({
            "type": "info", "icon": "⭐",
            "text": f"You have <strong>{len(bonus)}</strong> extra skills beyond what's required. These show range and can be a differentiator with human recruiters."
        })

    # ── Keyword overlap ────────────────────────────────────────────────────
    kw = scores["keyword_overlap"]
    if kw < 25:
        insights.append({
            "type": "danger", "icon": "🔤",
            "text": "Very low keyword overlap with the JD. Review the job description and incorporate more of its specific terminology into your resume."
        })
    elif kw >= 50:
        insights.append({
            "type": "success", "icon": "🔤",
            "text": "Good keyword overlap. Your resume shares significant vocabulary with the job description."
        })

    # ── Positive reinforcement ────────────────────────────────────────────
    if len(matched) >= 5:
        insights.append({
            "type": "success", "icon": "✅",
            "text": f"You matched <strong>{len(matched)}</strong> key skills with the role. That's a solid foundation."
        })

    return insights