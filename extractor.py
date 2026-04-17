import json
import os


def load_skills() -> dict:
    """Load the categorized skills database."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "skills.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_skills_categorized(text: str, skills_db: dict) -> dict[str, list[str]]:
    """
    Given cleaned text and a categorized skills dict, return
    { category: [found_skills] } for each category.
    """
    found: dict[str, list[str]] = {}
    for category, skill_list in skills_db.items():
        hits = []
        for skill in skill_list:
            # Match whole-word to avoid "r" matching "power bi" etc.
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, text):
                hits.append(skill)
        if hits:
            found[category] = hits
    return found


import re  # ensure re is available at module level