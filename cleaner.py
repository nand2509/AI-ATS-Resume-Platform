import re
from collections import Counter

# Common English stopwords (no NLTK required)
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","as","at",
    "be","because","been","before","being","below","between","both","but","by","can","did","do",
    "does","doing","down","during","each","few","for","from","further","get","got","had","has",
    "have","having","he","her","here","him","his","how","i","if","in","into","is","it","its",
    "just","ll","me","more","most","my","no","nor","not","of","off","on","once","only","or",
    "other","our","out","over","own","re","s","same","she","should","so","some","such","t",
    "than","that","the","their","them","then","there","these","they","this","those","through",
    "to","too","under","until","up","us","ve","very","was","we","were","what","when","where",
    "which","while","who","whom","why","will","with","would","you","your","yourself",
    "also","may","must","shall","use","used","using","work","working","well","new","make",
    "including","strong","good","high","based","within","across","ability","experience",
    "years","year","role","team","skills","skill","knowledge","understanding","responsible",
    "responsibilities","requirements","preferred","required","qualification","qualifications",
    "position","job","company","candidate","excellent","great","plus","looking",
    "resume","cv","applicant","apply","application",
}


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer that removes stopwords and short words."""
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]


def extract_keywords(text: str, top_n: int = 30) -> list[tuple[str, int]]:
    """Return top_n (word, frequency) pairs after removing stopwords."""
    tokens = tokenize(text)
    counts = Counter(tokens)
    return counts.most_common(top_n)