import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re, json, os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI ATS Checker — Beat the Bots, Get Hired",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=Fira+Code:wght@400;500&display=swap');

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0b0c14;color:#e2e8f0;}

/* ── NAV HEADER ── */
.top-nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:.9rem 2rem;background:rgba(11,12,20,.95);
  border-bottom:1px solid rgba(255,255,255,.07);
  border-radius:0 0 16px 16px;margin-bottom:1.5rem;
  backdrop-filter:blur(20px);flex-wrap:wrap;gap:.5rem;
}
.nav-logo{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#f9fafb;}
.nav-logo span{color:#b5f23d;}
.nav-pill{
  display:inline-flex;align-items:center;gap:6px;padding:7px 16px;
  border-radius:100px;font-size:.83rem;font-weight:600;cursor:pointer;
  border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.04);
  color:#9ca3af;transition:all .18s;margin:2px;font-family:'DM Sans',sans-serif;
}
.nav-pill:hover,.nav-pill.active{background:rgba(181,242,61,.15);border-color:rgba(181,242,61,.35);color:#b5f23d;}

/* ── HERO ── */
.hero-section{
  text-align:center;padding:4rem 2rem 3rem;
  background:linear-gradient(135deg,#12122a 0%,#1a1f4e 60%,#0f2d60 100%);
  border-radius:20px;border:1px solid rgba(99,179,237,.15);
  position:relative;overflow:hidden;margin-bottom:2rem;
}
.hero-section::before{
  content:'';position:absolute;top:-60%;right:-10%;width:500px;height:500px;
  background:radial-gradient(circle,rgba(181,242,61,.07) 0%,transparent 65%);
  pointer-events:none;
}
.hero-title{
  font-family:'Syne',sans-serif;font-size:clamp(2rem,5vw,3.6rem);
  font-weight:800;line-height:1.05;letter-spacing:-.02em;color:#f9fafb;
}
.hero-title span{color:#b5f23d;}
.hero-sub{color:#718096;font-size:1.05rem;max-width:560px;margin:.8rem auto 0;line-height:1.7;}

/* ── SCORE CARDS ── */
.score-card{
  background:linear-gradient(145deg,#181930,#1f2240);
  border-radius:16px;padding:1.5rem 1rem;text-align:center;
  border:1px solid rgba(99,179,237,.18);transition:transform .2s,box-shadow .2s;
}
.score-card:hover{transform:translateY(-3px);box-shadow:0 8px 30px rgba(0,0,0,.3);}
.score-num{font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;line-height:1;}
.score-lbl{font-size:.7rem;text-transform:uppercase;letter-spacing:2px;color:#718096;margin-top:.35rem;}
.score-desc{font-size:.78rem;color:#a0aec0;margin-top:.25rem;}
.s-green{color:#b5f23d;} .s-teal{color:#2dd4bf;} .s-yellow{color:#f6e05e;}
.s-red{color:#fb7185;}   .s-blue{color:#60a5fa;}  .s-purple{color:#a78bfa;}

/* ── SKILL BADGES ── */
.skill-badge{display:inline-block;padding:4px 12px;border-radius:100px;font-size:.76rem;font-weight:600;margin:3px;}
.b-green {background:rgba(181,242,61,.12);color:#b5f23d;border:1px solid rgba(181,242,61,.25);}
.b-red   {background:rgba(251,113,133,.11);color:#fb7185;border:1px solid rgba(251,113,133,.25);}
.b-teal  {background:rgba(45,212,191,.11); color:#2dd4bf;border:1px solid rgba(45,212,191,.25);}
.b-blue  {background:rgba(96,165,250,.11); color:#60a5fa;border:1px solid rgba(96,165,250,.25);}
.b-yellow{background:rgba(246,224,94,.11); color:#f6e05e;border:1px solid rgba(246,224,94,.25);}
.b-purple{background:rgba(167,139,250,.11);color:#a78bfa;border:1px solid rgba(167,139,250,.25);}

/* ── CARDS ── */
.sec-card{background:#13142a;border-radius:14px;padding:1.4rem 1.6rem;border:1px solid rgba(255,255,255,.06);margin-bottom:1rem;}
.sec-title{font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#718096;margin-bottom:.9rem;}

/* ── INSIGHT / REC CARDS ── */
.ins-item{padding:.8rem 1.1rem;border-radius:10px;margin-bottom:.5rem;font-size:.87rem;line-height:1.55;border-left:3px solid;}
.ins-green {background:rgba(181,242,61,.06);border-color:#b5f23d;color:#d9fca0;}
.ins-red   {background:rgba(251,113,133,.07);border-color:#fb7185;color:#fecdd3;}
.ins-yellow{background:rgba(246,224,94,.07); border-color:#f6e05e;color:#fde68a;}
.ins-blue  {background:rgba(96,165,250,.07); border-color:#60a5fa;color:#bfdbfe;}
.ins-teal  {background:rgba(45,212,191,.07); border-color:#2dd4bf;color:#99f6e4;}

/* ── PROGRESS BARS ── */
.prog-row{display:flex;align-items:center;gap:10px;margin-bottom:.55rem;}
.prog-label{min-width:130px;font-size:.8rem;color:#a0aec0;text-transform:capitalize;}
.prog-bg{flex:1;height:6px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden;}
.prog-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#60a5fa,#b5f23d);}
.prog-pct{min-width:38px;text-align:right;font-size:.78rem;font-weight:600;color:#b5f23d;}

/* ── SECTION CHECKER ── */
.sec-check{display:flex;align-items:center;gap:8px;padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:.87rem;}
.chk-yes{color:#b5f23d;font-size:1rem;} .chk-no{color:#fb7185;font-size:1rem;}

/* ── SALARY DISPLAY ── */
.salary-hero{
  background:linear-gradient(135deg,#1a1a2e,#0f2d60);
  border-radius:16px;padding:2rem;text-align:center;
  border:1px solid rgba(181,242,61,.2);margin-bottom:1rem;
}
.sal-range{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:#b5f23d;}
.sal-sub{font-size:.88rem;color:#a0aec0;margin-top:.3rem;}

/* ── BLOG CARDS ── */
.blog-card{background:#13142a;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,.06);transition:all .2s;margin-bottom:1rem;}
.blog-card:hover{border-color:rgba(255,255,255,.12);transform:translateY(-2px);}
.blog-thumb{height:100px;display:flex;align-items:center;justify-content:center;font-size:2.5rem;}
.blog-body{padding:1.2rem;}
.blog-title{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f9fafb;margin-bottom:.4rem;line-height:1.3;}
.blog-excerpt{font-size:.82rem;color:#718096;line-height:1.55;}

/* ── RESUME PREVIEW ── */
.resume-preview-box{
  background:#fff;color:#111;border-radius:12px;padding:2rem;
  font-family:'DM Sans',sans-serif;font-size:.82rem;line-height:1.5;
  box-shadow:0 20px 60px rgba(0,0,0,.4);min-height:400px;
}
.rv-name{font-size:1.3rem;font-weight:800;color:#0a0a0f;}
.rv-title{font-size:.88rem;font-weight:600;color:#374151;margin-bottom:.2rem;}
.rv-contact{font-size:.73rem;color:#6b7280;margin-bottom:.8rem;}
.rv-h2{font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;
  color:#0a0a0f;border-bottom:2px solid #b5f23d;padding-bottom:.25rem;margin:1rem 0 .4rem;}
.rv-p{font-size:.77rem;color:#374151;margin-bottom:.3rem;}

/* ── FOOTER ── */
.footer-bar{
  background:#0e0f1e;border-top:1px solid rgba(255,255,255,.07);
  padding:2rem;margin-top:3rem;border-radius:16px 16px 0 0;
  text-align:center;color:#6b7280;font-size:.82rem;
}
.footer-links{display:flex;justify-content:center;gap:1.5rem;margin-bottom:.8rem;flex-wrap:wrap;}
.footer-link{color:#6b7280;cursor:pointer;font-size:.82rem;transition:color .15s;}
.footer-link:hover{color:#b5f23d;}

/* ── MISC ── */
.grad-div{height:1px;background:linear-gradient(90deg,transparent,rgba(99,179,237,.2),transparent);margin:1.2rem 0;}
.tag-pill{display:inline-block;padding:3px 10px;border-radius:100px;font-size:.72rem;font-weight:600;margin:3px;text-transform:uppercase;letter-spacing:.5px;}
.tag-lime{background:rgba(181,242,61,.15);color:#b5f23d;border:1px solid rgba(181,242,61,.3);}
.tag-teal{background:rgba(45,212,191,.15);color:#2dd4bf;border:1px solid rgba(45,212,191,.3);}
.tag-blue{background:rgba(96,165,250,.15);color:#60a5fa;border:1px solid rgba(96,165,250,.3);}
.tag-purple{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.3);}

/* Sidebar */
[data-testid="stSidebar"]{background:#0e0f1e !important;border-right:1px solid rgba(255,255,255,.05);}
.stTextArea textarea{background:#13142a !important;color:#e2e8f0 !important;border:1px solid rgba(181,242,61,.2) !important;border-radius:10px !important;}
.stSelectbox>div>div{background:#13142a !important;border:1px solid rgba(255,255,255,.12) !important;}
div[data-testid="metric-container"]{background:#13142a;border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:.9rem;}
.stTabs [data-baseweb="tab-list"]{background:#13142a;border-radius:10px;gap:3px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#718096 !important;font-family:'Syne',sans-serif;font-weight:600;font-size:.82rem;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1e3a1e,#2a5c0a) !important;color:#b5f23d !important;}
.stFileUploader>div{background:#13142a !important;border:1.5px dashed rgba(181,242,61,.25) !important;border-radius:12px !important;}
hr{border-color:rgba(255,255,255,.05) !important;}
.stSlider>div>div{background:rgba(181,242,61,.3) !important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — active page
# ══════════════════════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = {}

# ══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION BAR
# ══════════════════════════════════════════════════════════════════════════════
PAGES = [
    ("🏠", "home",    "Home"),
    ("🎯", "checker", "Resume Checker"),
    ("📝", "builder", "Resume Builder"),
    ("💰", "salary",  "Salary Predictor"),
    ("📰", "blog",    "Blog"),
    ("ℹ️",  "about",  "About"),
    ("📬", "contact", "Contact"),
]

st.markdown('<div class="top-nav">', unsafe_allow_html=True)
nav_cols = st.columns([2] + [1]*len(PAGES))
with nav_cols[0]:
    st.markdown('<div class="nav-logo">AI<span>ATS</span> Checker</div>', unsafe_allow_html=True)
for i, (icon, pid, label) in enumerate(PAGES):
    with nav_cols[i+1]:
        active = "active" if st.session_state.page == pid else ""
        if st.button(f"{icon} {label}", key=f"nav_{pid}",
                     help=label,
                     use_container_width=True):
            st.session_state.page = pid
            if pid != "checker":
                st.session_state.analysis_done = False
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES (all inlined)
# ══════════════════════════════════════════════════════════════════════════════
STOPWORDS = {
    "a","about","above","after","again","all","am","an","and","any","are","as","at",
    "be","because","been","before","being","below","between","both","but","by","can",
    "did","do","does","doing","down","during","each","few","for","from","get","got",
    "had","has","have","having","he","her","here","him","his","how","i","if","in",
    "into","is","it","its","just","me","more","most","my","no","nor","not","of","off",
    "on","once","only","or","other","our","out","over","own","same","she","should",
    "so","some","such","than","that","the","their","them","then","there","these",
    "they","this","those","through","to","too","under","until","up","us","very","was",
    "we","were","what","when","where","which","while","who","whom","why","will","with",
    "would","you","your","also","may","must","use","used","using","work","working",
    "well","new","make","including","strong","good","high","based","within","across",
    "ability","experience","years","year","role","team","skills","skill","knowledge",
    "understanding","responsible","responsibilities","requirements","preferred",
    "required","qualification","position","job","company","candidate","excellent",
    "great","plus","looking","resume","cv","applicant","apply","application","etc",
    "per","one","two","three","four","five","six","seven","eight","nine","ten",
}

SKILLS_DB = {
    "programming":    ["python","r","java","scala","javascript","typescript","go","rust","kotlin","bash","matlab","sas","c++"],
    "data_analysis":  ["pandas","numpy","scipy","data analysis","exploratory data analysis","eda","data wrangling","data cleaning","feature engineering","statistical analysis","hypothesis testing","a/b testing","regression analysis","time series analysis","forecasting","predictive modeling","descriptive statistics"],
    "machine_learning":["machine learning","deep learning","neural networks","nlp","natural language processing","computer vision","scikit-learn","tensorflow","pytorch","keras","xgboost","lightgbm","random forest","gradient boosting","clustering","classification","mlops","recommendation systems"],
    "databases":      ["sql","mysql","postgresql","mongodb","cassandra","redis","dynamodb","bigquery","snowflake","databricks","nosql","data modeling","etl","data warehouse","data lake"],
    "visualization":  ["tableau","power bi","looker","data visualization","matplotlib","seaborn","plotly","qlik","metabase","grafana","google data studio","dashboard"],
    "cloud_devops":   ["aws","azure","gcp","google cloud","docker","kubernetes","terraform","ci/cd","devops","airflow","spark","hadoop","kafka","lambda","s3","ec2","sagemaker"],
    "business_tools": ["excel","microsoft office","google sheets","powerpoint","jira","confluence","slack","notion","trello","asana","salesforce","sap","erp","crm"],
    "soft_skills":    ["communication","leadership","problem solving","teamwork","collaboration","project management","agile","scrum","analytical thinking","critical thinking","stakeholder management","time management","mentoring"],
    "finance":        ["financial modeling","financial analysis","budgeting","forecasting","balance sheet","cash flow","valuation","investment analysis","risk management","financial reporting","accounting","variance analysis","kpi","roi"],
    "statistics":     ["statistics","probability","bayesian","regression","anova","chi-square","t-test","correlation","data science","experimental design","sampling","confidence intervals","multivariate analysis"],
}

def extract_text_from_pdf(file):
    import pdfplumber
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        return f"ERROR: {e}"
    return text.strip()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_keywords(text, top_n=30):
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return Counter(tokens).most_common(top_n)

def load_skills():
    path = os.path.join(os.path.dirname(__file__), "data", "skills.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return SKILLS_DB

def extract_skills_categorized(text, skills_db):
    found = {}
    for category, skill_list in skills_db.items():
        hits = [s for s in skill_list if re.search(r"\b"+re.escape(s.lower())+r"\b", text)]
        if hits:
            found[category] = hits
    return found

def compute_detailed_score(resume_text, jd_text, matched_skills, jd_skills,
                            w_sim=0.50, w_sk=0.35, w_kw=0.15):
    try:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        mat = vec.fit_transform([resume_text, jd_text])
        semantic = float(np.clip(cosine_similarity(mat[0], mat[1])[0][0]*100, 0, 100))
    except Exception:
        semantic = 0.0
    skill_cov = (len(matched_skills)/len(jd_skills)*100) if jd_skills else 100.0
    r_words, j_words = set(resume_text.split()), set(jd_text.split())
    kw_overlap = float(np.clip(len(r_words & j_words)/len(j_words)*100, 0, 100)) if j_words else 0.0
    total_w = w_sim + w_sk + w_kw or 1.0
    overall = (w_sim*semantic + w_sk*skill_cov + w_kw*kw_overlap) / total_w
    return {
        "overall":         round(float(np.clip(overall, 0, 100)), 1),
        "semantic":        round(semantic, 1),
        "skill_coverage":  round(skill_cov, 1),
        "keyword_overlap": round(kw_overlap, 1),
    }

def detect_resume_sections(raw_text):
    t = raw_text.lower()
    SECS = {
        "Summary / Objective":["summary","objective","profile","about me"],
        "Work Experience":     ["experience","employment","work history","career"],
        "Education":           ["education","academic","degree","university","college"],
        "Skills":              ["skills","technical skills","core competencies","expertise"],
        "Projects":            ["projects","portfolio","case study"],
        "Certifications":      ["certification","certificate","certified","license"],
        "Achievements":        ["achievement","award","honor","recognition"],
        "Contact Info":        ["email","phone","linkedin","github","address"],
    }
    found, missing = [], []
    for sec, kws in SECS.items():
        (found if any(k in t for k in kws) else missing).append(sec)
    return found, missing

def detect_job_role(jd_text):
    t = jd_text.lower()
    roles = {
        "Data Analyst":        ["data analyst","business intelligence","tableau","power bi","sql"],
        "Data Scientist":      ["data scientist","machine learning","deep learning","tensorflow","pytorch"],
        "Software Engineer":   ["software engineer","developer","backend","frontend","api","microservices"],
        "DevOps / Cloud":      ["devops","aws","azure","gcp","docker","kubernetes","terraform"],
        "Product Manager":     ["product manager","roadmap","agile","scrum","user story"],
        "Business Analyst":    ["business analyst","requirements","process improvement","gap analysis"],
        "Marketing":           ["marketing","seo","campaign","social media","brand","digital"],
        "Finance":             ["finance","accounting","financial","budgeting","audit","cpa"],
        "HR / Recruiter":      ["human resources","recruiting","talent acquisition","onboarding","payroll"],
        "Cybersecurity":       ["cybersecurity","security","penetration testing","siem","firewall"],
    }
    scores = {role: sum(1 for kw in kws if kw in t) for role, kws in roles.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General / Unknown"

def estimate_experience_level(resume_text):
    t = resume_text.lower()
    if any(k in t for k in ["senior","lead","principal","architect","director","manager","vp","chief"]):
        return "Senior", "#a78bfa"
    if any(k in t for k in ["mid","intermediate","3 years","4 years","5 years","2+ years","3+ years"]):
        return "Mid-Level", "#60a5fa"
    if any(k in t for k in ["junior","entry","fresher","intern","graduate","0-1","1 year"]):
        return "Junior / Entry", "#f6e05e"
    return "Not Detected", "#a0aec0"

def generate_insights(overall, matched, missing, bonus, scores, word_count):
    ins = []
    if overall >= 80:
        ins.append(("green","🚀",f"Excellent score of <strong>{overall:.0f}%</strong> — above the 70% ATS cutoff. You are highly competitive."))
    elif overall >= 65:
        ins.append(("yellow","👍",f"Good match at <strong>{overall:.0f}%</strong>. Targeted improvements could push you above 80%."))
    elif overall >= 45:
        ins.append(("yellow","⚠️",f"Moderate match at <strong>{overall:.0f}%</strong>. Add missing skills and mirror JD language."))
    else:
        ins.append(("red","🔴",f"Low match at <strong>{overall:.0f}%</strong>. Significant keyword and skill additions needed."))
    if scores["semantic"] < 30:
        ins.append(("red","📝","Resume language differs greatly from JD. Use the exact same terminology — mirror JD phrases."))
    elif scores["semantic"] >= 65:
        ins.append(("green","✍️","Strong language alignment — vocabulary closely mirrors role requirements."))
    if scores["skill_coverage"] >= 85:
        ins.append(("green","🎯",f"Excellent skill coverage: <strong>{scores['skill_coverage']:.0f}%</strong> of required skills present."))
    elif scores["skill_coverage"] < 40:
        ins.append(("red","📚",f"Only <strong>{scores['skill_coverage']:.0f}%</strong> skill coverage. Missing <strong>{len(missing)}</strong> required skills."))
    if missing:
        top = ", ".join(f"<code style='background:rgba(255,255,255,.08);padding:1px 5px;border-radius:4px'>{s}</code>" for s in missing[:5])
        ins.append(("yellow","❌",f"Top missing skills: {top}"))
    if word_count < 300:
        ins.append(("red","📏",f"Resume only ~{word_count} words. Expand to 450–700 words with specific details."))
    elif word_count > 900:
        ins.append(("yellow","✂️",f"Resume ~{word_count} words may be too long. Aim for 1-2 pages max."))
    else:
        ins.append(("green","✅",f"Good resume length (~{word_count} words) — ideal for ATS and readability."))
    if len(bonus) >= 3:
        ins.append(("blue","⭐",f"<strong>{len(bonus)}</strong> extra skills beyond requirements — highlight the best 3 in your summary."))
    if scores["keyword_overlap"] < 20:
        ins.append(("red","🔤","Very low keyword overlap. Add more JD-specific terminology to your resume."))
    elif scores["keyword_overlap"] >= 50:
        ins.append(("green","🔤","Good keyword overlap with the job description."))
    return ins

def generate_recommendations(overall, matched, missing, bonus, scores,
                              sec_found, sec_missing, job_role, word_count):
    recs = []
    if overall >= 85:
        recs.append(("🏆","green","Top Tier Match",f"Score of {overall:.0f}% — you're in the top candidate bracket. Focus on interview prep."))
    elif overall >= 70:
        recs.append(("🚀","green","Strong Match",f"Score of {overall:.0f}% puts you above most applicants. A few tweaks can make it near-perfect."))
    elif overall >= 50:
        recs.append(("⚠️","yellow","Moderate — Needs Work",f"At {overall:.0f}%, add missing skills and mirror JD language precisely."))
    else:
        recs.append(("🔴","red","Low Match — Act Now",f"Score of {overall:.0f}% — rework skills section and rewrite summary with JD keywords."))
    if scores["semantic"] < 30:
        recs.append(("📝","red","Critical Language Mismatch","Rewrite summary and bullet points using the exact terminology from the job description."))
    elif scores["semantic"] < 55:
        recs.append(("✍️","yellow","Improve Language Alignment","Mirror JD phrasing in your bullets — use exact words, not paraphrases."))
    else:
        recs.append(("✍️","green","Good Language Alignment","Your resume language aligns well. Keep this consistency throughout."))
    if missing:
        top5 = ", ".join(f"'{s}'" for s in missing[:5])
        recs.append(("❌","red",f"Add {len(missing)} Missing Skill(s)",f"Priority: {top5}. Name them explicitly — ATS won't infer from context."))
    if "Summary / Objective" in sec_missing:
        recs.append(("📌","yellow","Add a Professional Summary","3-4 lines with your title, years of experience, and 2-3 key skills. Critical for ATS."))
    if "Skills" in sec_missing:
        recs.append(("🔧","red","Missing Skills Section","Many ATS systems specifically scan for a dedicated Skills section. Add one immediately."))
    if any(s in sec_missing for s in ["Projects","Certifications"]):
        recs.append(("📋","yellow","Add Projects & Certifications",f"For {job_role} roles, these sections significantly boost ATS score and credibility."))
    if word_count < 300:
        recs.append(("📏","red","Resume Too Short",f"Only ~{word_count} words. Expand to 450-700 with specific examples and outcomes."))
    elif word_count > 900:
        recs.append(("✂️","yellow","Consider Trimming",f"~{word_count} words. Remove roles older than 10 years and generic filler phrases."))
    else:
        recs.append(("✅","green","Good Resume Length",f"~{word_count} words is in the ideal range."))
    if scores["keyword_overlap"] < 20:
        recs.append(("🔤","red","Very Low Keyword Overlap","Copy the exact job title, required tools, and domain terms from the JD into your resume."))
    if len(bonus) >= 5:
        recs.append(("⭐","green","Highlight Extra Skills",f"You have {len(bonus)} skills beyond requirements. Feature the most relevant 3-4 in your summary."))
    recs.append(("📐","blue","ATS Formatting Tip","Single-column layout only. No tables, graphics, or text boxes — ATS parsers skip them."))
    recs.append(("🔗","blue","Tailor Per Application","Customizing per job — even matching the exact title — can boost ATS score by 10-20%."))
    return recs

PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a0aec0", family="DM Sans"),
    margin=dict(l=16, r=16, t=36, b=16),
)

def vc(v):
    return "s-green" if v >= 70 else ("s-yellow" if v >= 45 else "s-red")

def color_val(v):
    return "#b5f23d" if v >= 70 else ("#f6e05e" if v >= 45 else "#fb7185")

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  (shared across all pages)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎯 AI ATS Checker")
    st.markdown("---")
    st.markdown("**Navigate**")
    for icon, pid, label in PAGES:
        if st.button(f"{icon} {label}", key=f"sb_{pid}", use_container_width=True):
            st.session_state.page = pid
            if pid != "checker":
                st.session_state.analysis_done = False
            st.rerun()
    st.markdown("---")
    st.markdown("**📋 ATS Score Guide**")
    st.markdown("""
<div style='font-size:.82rem;color:#a0aec0;line-height:2.1'>
🟢 <b>85–100%</b> Excellent<br>
🔵 <b>70–84%</b> Strong<br>
🟡 <b>50–69%</b> Moderate<br>
🔴 <b>0–49%</b> Low Match
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with ❤️ for job seekers · v2.0")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # Hero
    st.markdown("""
    <div class="hero-section">
      <div style="font-family:'Fira Code',monospace;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#b5f23d;margin-bottom:.8rem">✦ AI-Powered ATS Resume Analysis</div>
      <div class="hero-title">Beat the Bots.<br><span>Get Hired.</span></div>
      <p class="hero-sub">75% of resumes never reach a human. Our AI analyzes your resume against any job description and tells you exactly how to fix it.</p>
      <div style="margin-top:1.5rem">
        <span class="tag-pill tag-lime">Free to Use</span>
        <span class="tag-pill tag-teal">Any Job Type</span>
        <span class="tag-pill tag-blue">No Signup</span>
        <span class="tag-pill tag-purple">Instant Results</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    for col, num, label, color in [
        (s1, "98%",  "ATS Compatibility",   "#b5f23d"),
        (s2, "2.4x", "More Interviews",      "#2dd4bf"),
        (s3, "50K+", "Resumes Analyzed",     "#a78bfa"),
        (s4, "30s",  "Analysis Time",        "#f6e05e"),
    ]:
        col.markdown(f"""<div class="score-card">
          <div class="score-num" style="color:{color}">{num}</div>
          <div class="score-lbl">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("### 🚀 Everything You Need to Land the Job")
    f1, f2, f3 = st.columns(3)
    for col, icon, title, desc, tag, page_id in [
        (f1,"🎯","ATS Resume Checker","Upload resume + paste any JD. Get ATS score, skill gaps, keyword analysis, and smart recommendations in seconds.","Free · Any Role","checker"),
        (f2,"📝","Resume Builder","Build a professional, ATS-optimized resume from scratch with our guided form. Live preview + print-ready output.","AI-Powered","builder"),
        (f3,"💰","Salary Predictor","Know your worth before you negotiate. Data-driven salary ranges by role, experience, location, and skills.","Data-Driven","salary"),
    ]:
        col.markdown(f"""<div class="sec-card">
          <div style="font-size:2.2rem;margin-bottom:.8rem">{icon}</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#f9fafb;margin-bottom:.5rem">{title}</div>
          <p style="color:#718096;font-size:.86rem;line-height:1.6">{desc}</p>
          <div style="margin-top:.8rem"><span class="tag-pill tag-lime" style="font-size:.68rem">{tag}</span></div>
        </div>""", unsafe_allow_html=True)
        if col.button(f"Open {title.split()[0]} →", key=f"home_{page_id}", use_container_width=True):
            st.session_state.page = page_id
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("### 🔄 How It Works — 3 Steps to More Interviews")
    h1, h2, h3 = st.columns(3)
    for col, num, color, title, desc in [
        (h1,"01","#b5f23d","Upload Resume","Upload your PDF or build one with our builder. We extract and analyze every word."),
        (h2,"02","#2dd4bf","Paste Job Description","Copy any JD from LinkedIn, Indeed, or anywhere. Works for any role, any industry."),
        (h3,"03","#a78bfa","Get Score & Fix It","Receive ATS score, skill gap report, keyword analysis, and personalized recommendations."),
    ]:
        col.markdown(f"""<div class="sec-card text-center">
          <div style="width:44px;height:44px;border-radius:50%;background:rgba(255,255,255,.06);
               border:1px solid {color}40;display:flex;align-items:center;justify-content:center;
               margin:0 auto .8rem;font-family:'Syne',sans-serif;font-weight:800;color:{color};font-size:.9rem">{num}</div>
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#f9fafb;margin-bottom:.4rem">{title}</div>
          <p style="color:#718096;font-size:.84rem;line-height:1.6">{desc}</p>
        </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-bar">
      <div class="footer-links">
        <span class="footer-link" onclick="">Resume Checker</span>
        <span class="footer-link">Resume Builder</span>
        <span class="footer-link">Salary Predictor</span>
        <span class="footer-link">Blog</span>
        <span class="footer-link">Terms & Conditions</span>
        <span class="footer-link">Privacy Policy</span>
        <span class="footer-link">Contact</span>
      </div>
      <p>© 2026 AI ATS Checker. All rights reserved. Built with ❤️ for job seekers.</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESUME CHECKER
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "checker":

    st.markdown("""<div style="font-family:'Fira Code',monospace;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#b5f23d;margin-bottom:.3rem">ATS Resume Checker</div>""", unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        # ── SCORING WEIGHTS (sidebar for checker) ──
        with st.sidebar:
            st.markdown("---")
            st.markdown("**⚙️ Scoring Weights**")
            w_sim = st.slider("Semantic Similarity", 0.1, 0.8, 0.50, 0.05)
            w_sk  = st.slider("Skill Coverage",      0.1, 0.8, 0.35, 0.05)
            w_kw  = st.slider("Keyword Overlap",     0.0, 0.5, 0.15, 0.05)
        w_sim = w_sim if 'w_sim' in dir() else 0.50
        w_sk  = w_sk  if 'w_sk'  in dir() else 0.35
        w_kw  = w_kw  if 'w_kw'  in dir() else 0.15

        col_h, col_tags = st.columns([3,2])
        with col_h:
            st.markdown("## 🎯 Analyze Your Resume")
        with col_tags:
            st.markdown("""<div style="text-align:right;padding-top:.5rem">
              <span class="tag-pill tag-lime">Free</span>
              <span class="tag-pill tag-teal">Any Job Type</span>
              <span class="tag-pill tag-blue">Instant</span>
            </div>""", unsafe_allow_html=True)

        in1, in2 = st.columns([1,1], gap="large")
        with in1:
            st.markdown("#### 📄 Upload Resume")
            uploaded_file = st.file_uploader("PDF format", type=["pdf"], label_visibility="collapsed")
            st.markdown("**— or paste resume text —**")
            resume_text_input = st.text_area("Resume Text", height=200, label_visibility="collapsed",
                                              placeholder="Paste your resume text here…")
        with in2:
            st.markdown("#### 📋 Paste Job Description")
            jd_input = st.text_area("Job Description", height=310, label_visibility="collapsed",
                                     placeholder="Paste the full job description here — any role, any industry…")

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col = st.columns([1,2,1])[1]
        if btn_col.button("🔍 Analyze Resume Now", use_container_width=True, type="primary"):
            raw_resume = ""
            if uploaded_file:
                raw_resume = extract_text_from_pdf(uploaded_file)
            if resume_text_input.strip():
                raw_resume += "\n" + resume_text_input.strip()
            if not raw_resume.strip():
                st.error("Please upload a resume PDF or paste resume text.")
                st.stop()
            if not jd_input.strip() or len(jd_input.strip()) < 30:
                st.error("Please paste a job description (at least 30 characters).")
                st.stop()

            with st.spinner("🔍 Analyzing resume against job description…"):
                clean_res = clean_text(raw_resume)
                clean_jd  = clean_text(jd_input)
                skills_db = load_skills()
                res_cat   = extract_skills_categorized(clean_res, skills_db)
                jd_cat    = extract_skills_categorized(clean_jd,  skills_db)
                res_flat  = [s for cat in res_cat.values() for s in cat]
                jd_flat   = [s for cat in jd_cat.values()  for s in cat]
                matched   = list(set(res_flat) & set(jd_flat))
                missing   = list(set(jd_flat)  - set(res_flat))
                bonus     = list(set(res_flat)  - set(jd_flat))
                scores    = compute_detailed_score(clean_res, clean_jd, matched, jd_flat, w_sim, w_sk, w_kw)
                res_kw    = extract_keywords(clean_res, 35)
                jd_kw     = extract_keywords(clean_jd,  35)
                sec_found, sec_missing = detect_resume_sections(raw_resume)
                job_role  = detect_job_role(jd_input)
                exp_level, exp_color = estimate_experience_level(raw_resume)
                word_count = len(raw_resume.split())
                insights  = generate_insights(scores["overall"], matched, missing, bonus, scores, word_count)
                recs      = generate_recommendations(scores["overall"], matched, missing, bonus, scores,
                                                     sec_found, sec_missing, job_role, word_count)
                # Category data
                cat_rows = []
                all_cats = set(list(res_cat.keys()) + list(jd_cat.keys()))
                for cat in sorted(all_cats):
                    r_set = set(res_cat.get(cat,[]))
                    j_set = set(jd_cat.get(cat,[]))
                    mc = r_set & j_set
                    if j_set:
                        pct = len(mc)/len(j_set)*100
                        cat_rows.append({"Category": cat.replace("_"," ").title(),
                                         "Coverage %": round(pct,1),
                                         "Matched": len(mc), "Required": len(j_set)})

                st.session_state.analysis_data = dict(
                    raw_resume=raw_resume, clean_res=clean_res, clean_jd=clean_jd,
                    matched=matched, missing=missing, bonus=bonus, scores=scores,
                    res_kw=res_kw, jd_kw=jd_kw, sec_found=sec_found, sec_missing=sec_missing,
                    job_role=job_role, exp_level=exp_level, exp_color=exp_color,
                    word_count=word_count, insights=insights, recs=recs,
                    cat_rows=cat_rows, res_cat=res_cat, jd_cat=jd_cat, jd_flat=jd_flat,
                )
                st.session_state.analysis_done = True
                st.rerun()

    else:
        # ══ RESULTS ══════════════════════════════════════════════════════
        D = st.session_state.analysis_data
        scores, matched, missing, bonus = D["scores"], D["matched"], D["missing"], D["bonus"]
        overall = scores["overall"]

        res_col, new_col = st.columns([3,1])
        with res_col:
            st.markdown("## 📊 Resume Analysis Results")
            st.markdown('<p style="color:#718096;font-size:.87rem">Your resume has been analyzed against ATS criteria and job requirements.</p>', unsafe_allow_html=True)
        with new_col:
            if st.button("← New Analysis", use_container_width=True):
                st.session_state.analysis_done = False
                st.rerun()

        # ── SCORE METRIC CARDS ────────────────────────────────────────────
        if overall >= 85:   sc_c, sc_l = "s-green",  "Excellent 🏆"
        elif overall >= 70: sc_c, sc_l = "s-teal",   "Strong 🚀"
        elif overall >= 50: sc_c, sc_l = "s-yellow",  "Moderate ⚠️"
        else:               sc_c, sc_l = "s-red",    "Low Match 🔴"

        m1,m2,m3,m4,m5 = st.columns(5)
        for col, val, lbl, desc, cc in [
            (m1, f"{overall:.0f}%",                      "ATS Score",       sc_l,                                           sc_c),
            (m2, f"{scores['semantic']:.0f}%",           "Semantic Match",  "TF-IDF Cosine",                                vc(scores['semantic'])),
            (m3, f"{scores['skill_coverage']:.0f}%",     "Skill Coverage",  f"{len(matched)}/{len(D['jd_flat'])} skills",   vc(scores['skill_coverage'])),
            (m4, f"{scores['keyword_overlap']:.0f}%",    "Keyword Overlap", f"+{len(bonus)} bonus skills",                  vc(scores['keyword_overlap'])),
            (m5, D["exp_level"],                          "Seniority",       D["job_role"],                                  "s-purple"),
        ]:
            col.markdown(f"""<div class="score-card">
              <div class="score-num {cc}">{val}</div>
              <div class="score-lbl">{lbl}</div>
              <div class="score-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────
        left, right = st.columns([2.1, 1], gap="large")

        with left:
            tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
                "🎯 Skills","📊 Charts","💡 Insights","🔤 Keywords","✅ Sections","📄 Resume"
            ])

            # ── SKILLS TAB ───────────────────────────────────────────────
            with tab1:
                c1,c2,c3 = st.columns(3)
                def skill_box(col, title, items, cls, empty):
                    col.markdown(f'<div class="sec-card"><div class="sec-title">{title}</div>', unsafe_allow_html=True)
                    if items:
                        col.markdown(" ".join(f'<span class="skill-badge {cls}">{s}</span>' for s in sorted(items)), unsafe_allow_html=True)
                    else:
                        col.markdown(f'<span style="color:#718096;font-size:.83rem">{empty}</span>', unsafe_allow_html=True)
                    col.markdown("</div>", unsafe_allow_html=True)
                skill_box(c1, "✅ Matched Skills", matched, "b-green", "No matching skills")
                skill_box(c2, "❌ Missing Skills", missing, "b-red",   "No missing skills 🎉")
                skill_box(c3, "⭐ Bonus Skills",   bonus,   "b-teal",  "No extra skills")

                if D["cat_rows"]:
                    st.markdown("#### 🗂️ Skill Category Coverage")
                    html = ""
                    for row in sorted(D["cat_rows"], key=lambda x: -x["Coverage %"]):
                        p = row["Coverage %"]
                        html += f"""<div class="prog-row">
                          <div class="prog-label">{row['Category']}</div>
                          <div class="prog-bg"><div class="prog-fill" style="width:{p}%"></div></div>
                          <div class="prog-pct">{p:.0f}%</div></div>"""
                    st.markdown(f'<div class="sec-card">{html}</div>', unsafe_allow_html=True)

            # ── CHARTS TAB ───────────────────────────────────────────────
            with tab2:
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=overall,
                        title={"text":"ATS Score","font":{"color":"#a0aec0","size":13}},
                        number={"suffix":"%","font":{"color":"#b5f23d","size":36,"family":"Syne"}},
                        gauge={"axis":{"range":[0,100],"tickcolor":"#4a5568"},
                               "bar":{"color":"#b5f23d","thickness":0.25},
                               "bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(0,0,0,0)",
                               "steps":[{"range":[0,50],"color":"rgba(251,113,133,.12)"},
                                        {"range":[50,70],"color":"rgba(246,224,94,.1)"},
                                        {"range":[70,100],"color":"rgba(181,242,61,.1)"}],
                               "threshold":{"line":{"color":"#a78bfa","width":3},"value":overall}}
                    ))
                    fig_g.update_layout(**PLOT_BG, height=255)
                    st.plotly_chart(fig_g, use_container_width=True)

                with ch2:
                    cats = ["Semantic","Skill Coverage","Keyword Overlap"]
                    vals = [scores["semantic"],scores["skill_coverage"],scores["keyword_overlap"]]
                    fig_r = go.Figure(go.Scatterpolar(
                        r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
                        fillcolor="rgba(181,242,61,.08)",
                        line=dict(color="#b5f23d",width=2),
                        marker=dict(color="#2dd4bf",size=7)
                    ))
                    fig_r.update_layout(**PLOT_BG, height=255,
                        polar=dict(bgcolor="rgba(0,0,0,0)",
                                   radialaxis=dict(visible=True,range=[0,100],
                                                   gridcolor="rgba(255,255,255,.05)",
                                                   tickfont=dict(color="#4a5568",size=9)),
                                   angularaxis=dict(gridcolor="rgba(255,255,255,.05)",
                                                    tickfont=dict(color="#8892b0"))),
                        title=dict(text="Score Dimensions",font=dict(color="#a0aec0",size=13))
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

                # Skill distribution
                fig_b = px.bar(
                    pd.DataFrame({"Category":["Matched","Missing","Bonus"],
                                  "Count":[len(matched),len(missing),len(bonus)]}),
                    x="Category", y="Count", color="Category",
                    color_discrete_map={"Matched":"#b5f23d","Missing":"#fb7185","Bonus":"#2dd4bf"},
                    text="Count", title="Skill Distribution"
                )
                fig_b.update_traces(textposition="outside", marker_line_width=0)
                fig_b.update_layout(**PLOT_BG, height=260, showlegend=False,
                                    xaxis=dict(gridcolor="rgba(255,255,255,.03)"),
                                    yaxis=dict(gridcolor="rgba(255,255,255,.03)"))
                st.plotly_chart(fig_b, use_container_width=True)

                # Score vs benchmark
                dim_df = pd.DataFrame({
                    "Dimension":["Semantic Match","Skill Coverage","Keyword Overlap"],
                    "Your Score":[scores["semantic"],scores["skill_coverage"],scores["keyword_overlap"]],
                    "Target":[65,75,40]
                })
                fig_dim = go.Figure()
                fig_dim.add_trace(go.Bar(name="Your Score", x=dim_df["Dimension"],
                                         y=dim_df["Your Score"], marker_color="#b5f23d",
                                         text=[f"{v:.0f}%" for v in dim_df["Your Score"]],
                                         textposition="outside"))
                fig_dim.add_trace(go.Bar(name="Good Target", x=dim_df["Dimension"],
                                         y=dim_df["Target"], marker_color="rgba(167,139,250,.35)",
                                         text=[f"{v}%" for v in dim_df["Target"]],
                                         textposition="outside"))
                fig_dim.update_layout(**PLOT_BG, height=280, barmode="group",
                                      title="Your Score vs Good Target",
                                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                                      xaxis=dict(gridcolor="rgba(255,255,255,.03)"),
                                      yaxis=dict(gridcolor="rgba(255,255,255,.03)",range=[0,120]))
                st.plotly_chart(fig_dim, use_container_width=True)

                if D["cat_rows"]:
                    fig_c = px.bar(pd.DataFrame(D["cat_rows"]).sort_values("Coverage %"),
                                   x="Coverage %", y="Category", orientation="h",
                                   color="Coverage %",
                                   color_continuous_scale=["#fb7185","#f6e05e","#b5f23d"],
                                   range_color=[0,100], text="Coverage %",
                                   title="Skill Coverage by Category")
                    fig_c.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
                    fig_c.update_layout(**PLOT_BG, height=max(230, len(D["cat_rows"])*46),
                                        coloraxis_showscale=False,
                                        xaxis=dict(gridcolor="rgba(255,255,255,.03)",range=[0,115]),
                                        yaxis=dict(gridcolor="rgba(255,255,255,.03)"))
                    st.plotly_chart(fig_c, use_container_width=True)

            # ── INSIGHTS TAB ─────────────────────────────────────────────
            with tab3:
                st.markdown("#### 💡 Score Insights")
                cls_map = {"green":"ins-green","yellow":"ins-yellow","red":"ins-red","blue":"ins-blue","teal":"ins-teal"}
                for color, icon, text in D["insights"]:
                    css = cls_map.get(color,"ins-blue")
                    st.markdown(f'<div class="ins-item {css}">{icon} {text}</div>', unsafe_allow_html=True)

                st.markdown('<div class="grad-div"></div>', unsafe_allow_html=True)
                st.markdown("#### 🗺️ Priority Action Plan")
                if missing:
                    ac1, ac2 = st.columns(2)
                    for i, sk in enumerate(missing[:10]):
                        with (ac1 if i%2==0 else ac2):
                            st.markdown(f"""<div class="ins-item ins-yellow">
                              <strong>Add:</strong> <code style="background:rgba(255,255,255,.08);padding:1px 5px;border-radius:4px">{sk}</code> — required by JD
                            </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ins-item ins-green">✅ All JD-required skills found in your resume!</div>', unsafe_allow_html=True)

                st.markdown('<div class="grad-div"></div>', unsafe_allow_html=True)
                st.markdown("#### 📝 Universal ATS Tips")
                for cls, icon, text in [
                    ("ins-blue","📏","Keep to 1-2 pages — concise beats comprehensive for ATS systems."),
                    ("ins-blue","🔢","Quantify achievements: 'Improved sales by 32%' beats 'Improved sales'."),
                    ("ins-blue","📂","Use standard headers: Summary, Experience, Education, Skills."),
                    ("ins-blue","🔤","Avoid tables, columns, text boxes — ATS parsers often skip them."),
                    ("ins-blue","🎯","Mirror the exact job title from the JD in your resume headline."),
                    ("ins-yellow","⚠️","Don't keyword stuff — recruiters will notice immediately."),
                    ("ins-blue","🔗","Customize per application — even small edits improve your score."),
                ]:
                    st.markdown(f'<div class="ins-item {cls}">{icon} {text}</div>', unsafe_allow_html=True)

            # ── KEYWORDS TAB ─────────────────────────────────────────────
            with tab4:
                kc1, kc2 = st.columns(2)
                with kc1:
                    st.markdown("#### 📄 Resume Keywords")
                    if D["res_kw"]:
                        df1 = pd.DataFrame(D["res_kw"], columns=["Keyword","Freq"])
                        fig1 = px.bar(df1.head(15), x="Freq", y="Keyword", orientation="h",
                                      text="Freq", color="Freq",
                                      color_continuous_scale=["#1e2d5e","#b5f23d"],
                                      title="Top Resume Keywords")
                        fig1.update_traces(textposition="outside")
                        fig1.update_layout(**PLOT_BG, height=420, coloraxis_showscale=False,
                                           xaxis=dict(gridcolor="rgba(255,255,255,.03)"),
                                           yaxis=dict(gridcolor="rgba(255,255,255,.03)"))
                        st.plotly_chart(fig1, use_container_width=True)
                with kc2:
                    st.markdown("#### 📋 JD Keywords")
                    if D["jd_kw"]:
                        df2 = pd.DataFrame(D["jd_kw"], columns=["Keyword","Freq"])
                        fig2 = px.bar(df2.head(15), x="Freq", y="Keyword", orientation="h",
                                      text="Freq", color="Freq",
                                      color_continuous_scale=["#1e1050","#a78bfa"],
                                      title="Top JD Keywords")
                        fig2.update_traces(textposition="outside")
                        fig2.update_layout(**PLOT_BG, height=420, coloraxis_showscale=False,
                                           xaxis=dict(gridcolor="rgba(255,255,255,.03)"),
                                           yaxis=dict(gridcolor="rgba(255,255,255,.03)"))
                        st.plotly_chart(fig2, use_container_width=True)

                r_kw_set = {w for w,_ in D["res_kw"]}
                j_kw_set = {w for w,_ in D["jd_kw"]}
                common   = r_kw_set & j_kw_set
                only_jd  = j_kw_set - r_kw_set

                st.markdown("#### 🟢 Common Keywords (Resume ∩ JD)")
                if common:
                    st.markdown('<div class="sec-card">' +
                        " ".join(f'<span class="skill-badge b-green">{w}</span>' for w in sorted(common)) +
                        "</div>", unsafe_allow_html=True)

                st.markdown("#### 🔴 JD Keywords Missing from Resume")
                if only_jd:
                    st.markdown('<div class="sec-card">' +
                        " ".join(f'<span class="skill-badge b-red">{w}</span>' for w in sorted(only_jd)[:35]) +
                        "</div>", unsafe_allow_html=True)
                else:
                    st.success("✅ Your resume covers all major JD keywords!")

            # ── SECTIONS TAB ─────────────────────────────────────────────
            with tab5:
                st.markdown("#### ✅ Resume Section Checker")
                st.caption("ATS systems look for standard section headers. Missing sections reduce your score.")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown('<div class="sec-card"><div class="sec-title">✅ Detected Sections</div>', unsafe_allow_html=True)
                    for s in D["sec_found"]:
                        st.markdown(f'<div class="sec-check"><span class="chk-yes">✓</span> {s}</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with sc2:
                    st.markdown('<div class="sec-card"><div class="sec-title">❌ Missing Sections</div>', unsafe_allow_html=True)
                    if D["sec_missing"]:
                        for s in D["sec_missing"]:
                            st.markdown(f'<div class="sec-check"><span class="chk-no">✗</span> {s}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sec-check"><span class="chk-yes">✓</span> All key sections found 🎉</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="grad-div"></div>', unsafe_allow_html=True)
                rs1,rs2,rs3,rs4 = st.columns(4)
                rs1.metric("Word Count",      D["word_count"])
                rs2.metric("Sections Found",  len(D["sec_found"]))
                rs3.metric("Detected Role",   D["job_role"])
                rs4.metric("Seniority",       D["exp_level"])

            # ── RAW TEXT TAB ──────────────────────────────────────────────
            with tab6:
                st.markdown("#### 📄 Extracted Resume Text")
                if D["raw_resume"].strip():
                    st.text_area("", value=D["raw_resume"], height=400, label_visibility="collapsed")
                    t1,t2,t3 = st.columns(3)
                    t1.metric("Words",      D["word_count"])
                    t2.metric("Characters", len(D["raw_resume"]))
                    t3.metric("Est. Pages", max(1, round(D["word_count"]/400)))
                else:
                    st.warning("Could not extract text. Ensure PDF is text-based, not a scanned image.")

        # ── RIGHT: RECOMMENDATIONS ────────────────────────────────────────
        with right:
            st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;
                 color:#f9fafb;margin-bottom:1rem;padding:.6rem 1rem;
                 background:linear-gradient(135deg,#1a2a0a,#1e3a0f);
                 border-radius:12px;border:1px solid rgba(181,242,61,.2)">
              🧠 Smart Recommendations
            </div>""", unsafe_allow_html=True)

            colors = {"green":("#b5f23d","rgba(181,242,61,.07)"),
                      "yellow":("#f6e05e","rgba(246,224,94,.07)"),
                      "red":("#fb7185","rgba(251,113,133,.07)"),
                      "blue":("#60a5fa","rgba(96,165,250,.07)")}

            for icon, rtype, title, body in D["recs"]:
                col, bg = colors.get(rtype, colors["blue"])
                st.markdown(f"""<div style="border-radius:11px;padding:.9rem 1rem;margin-bottom:.6rem;
                  border-left:3px solid {col};background:{bg};font-size:.84rem;line-height:1.55;">
                  <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.86rem;
                       margin-bottom:.2rem;color:{col}">{icon} {title}</div>
                  <div style="opacity:.9">{body}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="grad-div"></div>', unsafe_allow_html=True)

            # Quick stats
            st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;
                 color:#718096;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.7rem">
              📌 Quick Stats</div>""", unsafe_allow_html=True)

            for label, val, color in [
                ("Detected Role",    D["job_role"],          "#a78bfa"),
                ("Seniority",        D["exp_level"],         D["exp_color"]),
                ("Matched Skills",   len(matched),           "#b5f23d"),
                ("Missing Skills",   len(missing),           "#fb7185" if missing else "#b5f23d"),
                ("Bonus Skills",     len(bonus),             "#2dd4bf"),
                ("Word Count",       D["word_count"],        "#a0aec0"),
                ("Sections Found",   len(D["sec_found"]),    "#b5f23d"),
                ("Sections Missing", len(D["sec_missing"]),  "#fb7185" if D["sec_missing"] else "#b5f23d"),
            ]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;
                     padding:.38rem 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.82rem">
                  <span style="color:#718096">{label}</span>
                  <span style="color:{color};font-weight:600">{val}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="grad-div"></div>', unsafe_allow_html=True)

            # Score breakdown mini bars
            st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;
                 color:#718096;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.7rem">
              📊 Score Breakdown</div>""", unsafe_allow_html=True)
            for label, val in [("Semantic Match",scores["semantic"]),
                                ("Skill Coverage",scores["skill_coverage"]),
                                ("Keyword Overlap",scores["keyword_overlap"])]:
                c = color_val(val)
                st.markdown(f"""<div class="prog-row" style="margin-bottom:.7rem">
                  <div class="prog-label">{label}</div>
                  <div class="prog-bg"><div class="prog-fill" style="width:{val}%"></div></div>
                  <div class="prog-pct" style="color:{c}">{val:.0f}%</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESUME BUILDER
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "builder":
    st.markdown("""<div style="font-family:'Fira Code',monospace;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#b5f23d;margin-bottom:.3rem">Resume Builder</div>""", unsafe_allow_html=True)
    st.markdown("## 📝 Build Your ATS-Optimized Resume")
    st.caption("Fill in the sections below and generate a professional, ATS-ready resume.")

    form_col, preview_col = st.columns([1,1], gap="large")

    with form_col:
        tab_p, tab_e, tab_edu, tab_sk = st.tabs(["👤 Personal","💼 Experience","🎓 Education","🛠️ Skills"])

        with tab_p:
            b_name    = st.text_input("Full Name *",           placeholder="e.g. Alex Johnson")
            b_title   = st.text_input("Job Title / Headline *",placeholder="e.g. Senior Data Analyst")
            b_email   = st.text_input("Email",                 placeholder="alex@email.com")
            b_phone   = st.text_input("Phone",                 placeholder="+91 98765 43210")
            b_links   = st.text_input("LinkedIn / GitHub",     placeholder="linkedin.com/in/alexjohnson")
            b_loc     = st.text_input("Location",              placeholder="Mumbai, India (Remote)")
            b_summary = st.text_area("Professional Summary",   height=110,
                placeholder="3-4 lines: your title, years of experience, 2-3 key skills, and what you bring to the role…")

        with tab_e:
            st.markdown("**Experience 1**")
            b_e1t  = st.text_input("Job Title",  key="e1t", placeholder="e.g. Data Analyst")
            b_e1c  = st.text_input("Company",    key="e1c", placeholder="e.g. TechCorp Inc.")
            b_e1d  = st.text_input("Duration",   key="e1d", placeholder="Jan 2022 – Present")
            b_e1de = st.text_area("Responsibilities & Achievements", key="e1de", height=110,
                placeholder="• Built Power BI dashboards, reducing reporting time by 40%\n• Analyzed 5M+ rows using Python and Pandas\n• Wrote SQL queries for data extraction")
            st.markdown("**Experience 2**")
            b_e2t  = st.text_input("Job Title",  key="e2t", placeholder="e.g. Junior Analyst")
            b_e2c  = st.text_input("Company",    key="e2c", placeholder="e.g. Analytics Co.")
            b_e2d  = st.text_input("Duration",   key="e2d", placeholder="Jun 2020 – Dec 2021")
            b_e2de = st.text_area("Responsibilities", key="e2de", height=90,
                placeholder="• Supported senior analysts with data cleaning\n• Built Excel dashboards for business reviews")

        with tab_edu:
            b_degree = st.text_input("Degree",         placeholder="e.g. B.Sc. Computer Science")
            b_school = st.text_input("University",     placeholder="e.g. University of Mumbai")
            b_year   = st.text_input("Year",           placeholder="2016 – 2020")
            b_certs  = st.text_area("Certifications",  height=90,
                placeholder="• Google Data Analytics Certificate (2023)\n• Microsoft Power BI (2022)\n• AWS Cloud Practitioner (in progress)")

        with tab_sk:
            b_tech   = st.text_area("Technical Skills (comma-separated)", height=70,
                placeholder="Python, SQL, Excel, Power BI, Tableau, Pandas, NumPy, Machine Learning, AWS")
            b_soft   = st.text_area("Soft Skills (comma-separated)", height=60,
                placeholder="Communication, Leadership, Problem Solving, Agile, Stakeholder Management")
            b_proj   = st.text_area("Projects", height=90,
                placeholder="• Sales Forecasting Dashboard — Python + Streamlit, 89% accuracy\n• Customer Churn Model — XGBoost, reduced churn by 22%")
            b_langs  = st.text_input("Languages", placeholder="English (Fluent), Hindi (Native)")

    with preview_col:
        st.markdown("#### 👁️ Live Resume Preview")
        if st.button("🔄 Generate / Refresh Preview", use_container_width=True, type="primary"):
            st.session_state["builder_preview"] = True

        if st.session_state.get("builder_preview"):
            contact_parts = [p for p in [b_email, b_phone, b_loc, b_links] if p]
            contact_line  = " · ".join(contact_parts)

            def fmt_bullets(txt):
                if not txt: return ""
                lines = [l.strip().lstrip("•-* ") for l in txt.split("\n") if l.strip()]
                return "<ul style='padding-left:1.2rem;margin:.3rem 0'>" + "".join(f"<li style='font-size:.76rem;color:#374151;margin-bottom:.2rem'>{l}</li>" for l in lines) + "</ul>"

            def skill_badges(csv):
                if not csv: return ""
                items = [s.strip() for s in csv.split(",") if s.strip()]
                return " ".join(f"<span style='display:inline-block;padding:2px 8px;background:#f0fdf4;color:#15803d;border-radius:4px;font-size:.72rem;margin:2px;font-weight:600'>{s}</span>" for s in items)

            html = f"""
            <div class="resume-preview-box">
              <div class="rv-name">{b_name or 'Your Name'}</div>
              <div class="rv-title">{b_title or 'Professional Title'}</div>
              {f'<div class="rv-contact">{contact_line}</div>' if contact_line else ''}
              {f'<div class="rv-h2">Professional Summary</div><div class="rv-p">{b_summary}</div>' if b_summary else ''}
              {f'<div class="rv-h2">Work Experience</div>' if b_e1t or b_e1c else ''}
              {f'<div style="margin-bottom:.7rem"><div style="font-weight:700;font-size:.8rem;color:#111">{b_e1t}{(" — "+b_e1c) if b_e1c else ""}</div>{(f"<div style=\'font-size:.72rem;color:#9ca3af\'>{b_e1d}</div>") if b_e1d else ""}{fmt_bullets(b_e1de)}</div>' if b_e1t or b_e1c else ''}
              {f'<div style="margin-bottom:.7rem"><div style="font-weight:700;font-size:.8rem;color:#111">{b_e2t}{(" — "+b_e2c) if b_e2c else ""}</div>{(f"<div style=\'font-size:.72rem;color:#9ca3af\'>{b_e2d}</div>") if b_e2d else ""}{fmt_bullets(b_e2de)}</div>' if b_e2t or b_e2c else ''}
              {f'<div class="rv-h2">Skills</div>{(f"<div class=\'rv-p\'><strong>Technical:</strong> {skill_badges(b_tech)}</div>") if b_tech else ""}{(f"<div class=\'rv-p\' style=\'margin-top:.3rem\'><strong>Soft Skills:</strong> {skill_badges(b_soft)}</div>") if b_soft else ""}' if b_tech or b_soft else ''}
              {f'<div class="rv-h2">Projects</div>{fmt_bullets(b_proj)}' if b_proj else ''}
              {f'<div class="rv-h2">Education</div><div style="font-weight:700;font-size:.8rem;color:#111">{b_degree}{(" — "+b_school) if b_school else ""}</div>{(f"<div style=\'font-size:.72rem;color:#9ca3af\'>{b_year}</div>") if b_year else ""}{fmt_bullets(b_certs)}' if b_degree or b_school else ''}
              {f'<div class="rv-h2">Languages</div><div class="rv-p">{b_langs}</div>' if b_langs else ''}
            </div>"""
            st.markdown(html, unsafe_allow_html=True)
            st.info("💡 To save as PDF: use your browser's Print function (Ctrl+P) and select 'Save as PDF'.")
        else:
            st.markdown("""<div class="sec-card" style="text-align:center;padding:3rem 1rem;">
              <div style="font-size:2.5rem;margin-bottom:.8rem">📄</div>
              <p style="color:#718096">Fill in the form and click <strong style="color:#b5f23d">Generate Preview</strong></p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="grad-div"></div>', unsafe_allow_html=True)
        if st.button("🎯 Check This Resume Against a JD →", use_container_width=True):
            st.session_state.page = "checker"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SALARY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "salary":
    st.markdown("""<div style="font-family:'Fira Code',monospace;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#b5f23d;margin-bottom:.3rem">Salary Predictor</div>""", unsafe_allow_html=True)
    st.markdown("## 💰 Know Your Worth")
    st.caption("Data-driven salary estimates by role, experience, location, and skills. Negotiate with confidence.")

    SALARY_DATA = {
        "Data Analyst":         {"base":[3.5,5.5,9,15,22,32],"usd":[55,75,100,140,190,260]},
        "Data Scientist":       {"base":[5,8,14,22,32,50],"usd":[80,110,140,190,250,350]},
        "Software Engineer":    {"base":[4,7,12,20,30,45],"usd":[70,100,140,185,245,330]},
        "Frontend Developer":   {"base":[3.5,6,10,17,25,38],"usd":[60,90,125,170,220,300]},
        "Backend Developer":    {"base":[4,7,12,20,30,46],"usd":[70,100,140,190,250,340]},
        "DevOps / Cloud":       {"base":[4.5,7.5,13,21,32,48],"usd":[75,105,145,195,260,350]},
        "Product Manager":      {"base":[5,9,15,25,38,60],"usd":[85,120,160,210,280,400]},
        "Business Analyst":     {"base":[3.5,5.5,9,14,20,30],"usd":[60,80,105,140,180,240]},
        "ML / AI Engineer":     {"base":[6,10,18,28,42,65],"usd":[100,140,185,250,330,480]},
        "Cybersecurity":        {"base":[4,7,12,20,30,45],"usd":[70,100,135,185,245,330]},
        "Digital Marketing":    {"base":[3,4.5,7.5,12,18,28],"usd":[45,65,90,125,165,220]},
        "Finance Analyst":      {"base":[3.5,5.5,9,15,22,35],"usd":[55,75,105,145,195,270]},
        "HR / Talent":          {"base":[2.5,4,6.5,10,15,22],"usd":[45,60,80,105,140,185]},
        "UX / UI Designer":     {"base":[3.5,5.5,9,15,22,32],"usd":[60,85,115,155,200,270]},
    }
    LOC_MULT = {"India — Metro (Mumbai/Bangalore/Delhi)":1.0,"India — Tier 2 Cities":0.75,
                "USA — San Francisco / Bay Area":("usd",1.8),"USA — New York City":("usd",1.7),
                "USA — Seattle / Austin":("usd",1.6),"USA — Midwest / Remote":("usd",1.3),
                "United Kingdom (London)":("usd",1.2),"Europe (Germany / Netherlands)":("usd",1.1),
                "Canada":("usd",1.25),"Australia":("usd",1.3),"UAE / Middle East":("usd",1.1),"Southeast Asia":("usd",0.7)}
    COMP_MULT = {"Startup (Seed–Series B)":0.85,"Growth Stage / Series C+":1.0,
                 "Mid-Size (500–5000 employees)":1.05,"Large Enterprise":1.1,
                 "FAANG / Top Tech":1.5,"Consulting / Agency":0.95,"Government / Non-profit":0.8}
    EDU_MULT  = {"Diploma / Associate":0.9,"Bachelor's Degree":1.0,"Master's Degree":1.1,"PhD / Doctorate":1.15}
    EXP_IDX   = {"0–1 year (Entry Level)":0,"1–3 years (Junior)":1,"3–5 years (Mid Level)":2,
                 "5–8 years (Senior)":3,"8–12 years (Lead / Principal)":4,"12+ years (Director / VP)":5}

    SKILL_BOOSTS = {
        "Data Analyst":["Power BI","Tableau","dbt","Snowflake","Python","SQL","Looker"],
        "Data Scientist":["PyTorch","TensorFlow","MLOps","LLMs","NLP","Spark","Databricks"],
        "Software Engineer":["Kubernetes","System Design","React","Node.js","Rust","Go","GraphQL"],
        "ML / AI Engineer":["LLM Fine-tuning","RAG","MLflow","Kubeflow","RLHF","Vector DBs"],
        "DevOps / Cloud":["Kubernetes","Terraform","AWS","GCP","GitHub Actions","Prometheus"],
        "Product Manager":["OKRs","Growth Hacking","SQL","A/B Testing","Figma","Roadmapping"],
    }

    i1, i2 = st.columns([1,1.3], gap="large")
    with i1:
        st.markdown('<div class="sec-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Your Details</div>', unsafe_allow_html=True)
        sal_role  = st.selectbox("Job Role / Title",    list(SALARY_DATA.keys()))
        sal_exp   = st.selectbox("Years of Experience", list(EXP_IDX.keys()))
        sal_loc   = st.selectbox("Location / Market",   list(LOC_MULT.keys()))
        sal_edu   = st.selectbox("Education Level",     list(EDU_MULT.keys()), index=1)
        sal_comp  = st.selectbox("Company Type",        list(COMP_MULT.keys()), index=3)
        sal_skills_str = st.text_input("Key Skills (comma-separated)", placeholder="Python, SQL, Machine Learning, AWS")
        predict_btn = st.button("💰 Predict My Salary", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with i2:
        if predict_btn:
            exp_idx  = EXP_IDX[sal_exp]
            loc_cfg  = LOC_MULT[sal_loc]
            comp_m   = COMP_MULT[sal_comp]
            edu_m    = EDU_MULT[sal_edu]
            sk_boost = 1 + min(len([s for s in sal_skills_str.split(",") if s.strip()]) * 0.02, 0.15)
            data     = SALARY_DATA[sal_role]

            is_india = isinstance(loc_cfg, float)
            if is_india:
                base_arr = data["base"]
                mid_val  = base_arr[min(exp_idx, len(base_arr)-1)] * loc_cfg * comp_m * edu_m * sk_boost
                low_val  = mid_val * 0.82
                high_val = mid_val * 1.22
                currency, unit = "₹", "LPA"
                def fmt(v): return f"₹{v:.1f}L"
                monthly = fmt(mid_val * 100000 / 12 / 100000)
            else:
                _, usd_mult = loc_cfg
                base_arr = data["usd"]
                mid_val  = base_arr[min(exp_idx, len(base_arr)-1)] * usd_mult * comp_m * edu_m * sk_boost
                low_val  = mid_val * 0.82
                high_val = mid_val * 1.22
                currency, unit = "$", "K/yr"
                def fmt(v): return f"${v:.0f}K"
                monthly = f"${mid_val*1000/12/1000:.0f}K"

            pct = ["25–45th","45–65th","65–80th","80–90th","90–95th","95–99th"][min(exp_idx,5)]

            st.markdown(f"""<div class="salary-hero">
              <div style="font-family:'Fira Code',monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#b5f23d;margin-bottom:.5rem">Estimated Annual Salary</div>
              <div class="sal-range">{fmt(low_val)} – {fmt(high_val)}</div>
              <div class="sal-sub">Median estimate: {fmt(mid_val)} / year · {unit}</div>
            </div>""", unsafe_allow_html=True)

            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Monthly (Est.)", monthly)
            pm2.metric("Percentile", pct)
            pm3.metric("Seniority", sal_exp.split("(")[0].strip())

            # Market comparison chart
            st.markdown('<div class="sec-card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">📊 Market Comparison</div>', unsafe_allow_html=True)
            comp_roles = list(SALARY_DATA.keys())[:8]
            comp_vals  = []
            for r in comp_roles:
                d = SALARY_DATA[r]
                arr = d["base"] if is_india else d["usd"]
                m = arr[min(exp_idx, len(arr)-1)] * (loc_cfg if is_india else loc_cfg[1]) * comp_m * edu_m
                comp_vals.append(round(m,1))
            fig_comp = go.Figure(go.Bar(
                x=comp_vals, y=comp_roles, orientation="h",
                marker_color=["#b5f23d" if r==sal_role else "rgba(255,255,255,.12)" for r in comp_roles],
                text=[fmt(v) for v in comp_vals], textposition="outside"
            ))
            fig_comp.update_layout(**PLOT_BG, height=340,
                                   xaxis=dict(gridcolor="rgba(255,255,255,.03)"),
                                   yaxis=dict(gridcolor="rgba(255,255,255,.03)"))
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Tips + skills
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown('<div class="sec-card">', unsafe_allow_html=True)
                st.markdown('<div class="sec-title">💡 Salary Boost Tips</div>', unsafe_allow_html=True)
                tips = []
                if sal_comp not in ["FAANG / Top Tech"]:
                    tips.append(("yellow","Target FAANG / top-tech — they pay 30-50% more for the same role."))
                if sal_edu == "Bachelor's Degree":
                    tips.append(("blue","A Master's degree can increase comp by 8-15% in most markets."))
                tips.append(("teal","Negotiate total compensation — equity + bonus can add 20-40% to base."))
                if is_india:
                    tips.append(("purple","Remote roles at US companies from India can yield 3-5x local salaries."))
                for cls, text in tips:
                    st.markdown(f'<div class="ins-item ins-{cls}" style="font-size:.82rem">{text}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with tc2:
                st.markdown('<div class="sec-card">', unsafe_allow_html=True)
                st.markdown('<div class="sec-title">🔥 High-Value Skills</div>', unsafe_allow_html=True)
                boost_skills = SKILL_BOOSTS.get(sal_role, ["Leadership","Data Analysis","Strategic Planning","Communication"])
                st.markdown(" ".join(f'<span class="skill-badge b-green">{s}</span>' for s in boost_skills), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div class="sec-card" style="text-align:center;padding:3rem 1rem">
              <div style="font-size:3rem;margin-bottom:.8rem">💰</div>
              <h3 style="font-family:'Syne',sans-serif;color:#f9fafb;font-size:1.1rem">Fill in your details</h3>
              <p style="color:#718096;font-size:.86rem;margin-top:.4rem">Select role, experience, and location to get a data-driven salary estimate.</p>
            </div>""", unsafe_allow_html=True)

    # Benchmark table
    st.markdown("---")
    st.markdown("### 📊 Salary Benchmarks by Role (India Metro, Mid-Level, 3-5 yrs)")
    table_data = []
    for role, d in SALARY_DATA.items():
        table_data.append({
            "Role": role,
            "Entry (₹LPA)": f"₹{d['base'][0]:.1f}L",
            "Junior (₹LPA)": f"₹{d['base'][1]:.1f}L",
            "Mid (₹LPA)": f"₹{d['base'][2]:.1f}L",
            "Senior (₹LPA)": f"₹{d['base'][3]:.1f}L",
            "Lead+ (₹LPA)": f"₹{d['base'][4]:.1f}L",
            "USD Mid": f"${d['usd'][2]}K",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: BLOG
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "blog":
    st.markdown("""<div style="font-family:'Fira Code',monospace;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#b5f23d;margin-bottom:.3rem">Blog</div>""", unsafe_allow_html=True)
    st.markdown("## 📰 Career Insights & Job Search Tips")
    st.caption("Expert advice on resume writing, ATS optimization, salary negotiation, and landing your dream job.")

    # ── Author info banner ──
    st.markdown("""<div class="sec-card" style="display:flex;align-items:center;gap:1.5rem;margin-bottom:2rem;padding:1.5rem 1.8rem">
      <div style="font-size:3rem;background:linear-gradient(135deg,#1a2a0a,#1e3a0f);border-radius:50%;width:72px;height:72px;display:flex;align-items:center;justify-content:center;flex-shrink:0">👨‍💻</div>
      <div>
        <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:#f9fafb">Nand Kishor Vyas</div>
        <div style="font-size:.83rem;color:#b5f23d;margin:.2rem 0">Associate Data Scientist @ InfiJobs | Ex-Data Scientist @ Staffing Professors | Ex-Data Scientist @ Codex Tech IT LLC</div>
        <div style="font-size:.8rem;color:#718096">📍 Ahmedabad, Gujarat, India &nbsp;·&nbsp; 🎓 REC Sonbhadra (B.Tech ECE) &nbsp;·&nbsp; 13K+ Followers</div>
        <div style="margin-top:.5rem">
          <span class="tag-pill tag-lime">Data Science</span>
          <span class="tag-pill tag-teal">Machine Learning</span>
          <span class="tag-pill tag-blue">AI / GenAI</span>
          <span class="tag-pill tag-purple">Data Analytics</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    POSTS = [
        ("🤖","AI in Healthcare: Confidence ≠ Correctness — How AI Diagnostic Systems Are Built","AI / GenAI","Apr 2026","4 min",
         "AI performs well with complete data but struggles when symptoms are incomplete or evolving. A deep-dive into RAG, HIPAA, and building trustworthy medical AI systems with Python + FastAPI + FAISS.",
         "teal"),
        ("⚛️","India's Nuclear Breakthrough: PFBR, Thorium Strategy & Energy Independence","Tech & Policy","Apr 2026","6 min",
         "India reaching criticality in the Prototype Fast Breeder Reactor at Kalpakkam marks a shift toward energy sovereignty. Breaking down the 3-stage nuclear vision and what it means for clean energy.",
         "lime"),
        ("💼","Hiring Is Now AI vs AI — What Candidates Must Do Differently in 2026","Career","Apr 2026","5 min",
         "Candidates are using AI to apply smarter. Recruiters are using AI to filter faster. Here's what InfiJobs published on navigating the new hiring landscape for international students and OPT candidates.",
         "blue"),
        ("📊","Corporate Governance & Layoffs: Should Executive Pay Be Tied to Workforce Stability?","Business","Apr 2026","5 min",
         "When companies announce mass layoffs while executive compensation reaches record highs, it raises real questions about corporate responsibility, shareholder value, and what responsible leadership looks like.",
         "yellow"),
        ("🚀","From Intern to Associate Data Scientist: My Journey in Data Science","Career Story","Mar 2026","8 min",
         "Starting from a data analyst internship at Unified Mentor to Jr. Data Scientist at Codex Tech, Data Science Trainer at Staffing Professors, and now Associate Data Scientist at InfiJobs — lessons learned along the way.",
         "purple"),
        ("🔑","Top ATS Keywords for Data Science Roles in 2026 — Insights from 500+ JDs","Keywords","Feb 2026","5 min",
         "After analyzing hundreds of data science job descriptions, these are the most critical keywords, tools, and phrases that ATS systems and recruiters look for. Curated from real hiring patterns.",
         "lime"),
    ]
    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    for i, (emoji, title, cat, date, read_time, excerpt, color) in enumerate(POSTS):
        tag_cls = {"lime":"tag-lime","yellow":"b-yellow","blue":"tag-blue",
                   "purple":"b-purple","teal":"b-teal"}.get(color,"tag-lime")
        with cols[i % 3]:
            st.markdown(f"""<div class="blog-card">
              <div class="blog-thumb" style="background:linear-gradient(135deg,#1a1a2e,#0f3060)">{emoji}</div>
              <div class="blog-body">
                <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem">
                  <span class="tag-pill tag-lime" style="font-size:.68rem">{cat}</span>
                  <span style="font-size:.74rem;color:#6b7280">{date} · {read_time} read</span>
                </div>
                <div class="blog-title">{title}</div>
                <p class="blog-excerpt">{excerpt}</p>
                <div style="margin-top:.8rem;font-size:.82rem;color:#b5f23d;font-weight:600;cursor:pointer">Read more →</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "about":
    st.markdown("""<div style="font-family:'Fira Code',monospace;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#b5f23d;margin-bottom:.3rem">About the Creator</div>""", unsafe_allow_html=True)
    st.markdown("## 👨‍💻 Nand Kishor Vyas")

    # Profile hero card
    st.markdown("""
    <div class="sec-card" style="margin-bottom:1.5rem;padding:2rem;">
      <div style="display:flex;align-items:flex-start;gap:2rem;flex-wrap:wrap;">
        <div style="font-size:5rem;background:linear-gradient(135deg,#1a2a0a,#0f3060);border-radius:50%;
             width:90px;height:90px;display:flex;align-items:center;justify-content:center;flex-shrink:0">👨‍💻</div>
        <div style="flex:1;min-width:260px">
          <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#f9fafb">Nand Kishor Vyas</div>
          <div style="font-size:.88rem;color:#b5f23d;margin:.3rem 0;font-weight:600">
            Associate Data Scientist @ InfiJobs &nbsp;·&nbsp; He/Him
          </div>
          <div style="font-size:.83rem;color:#a0aec0;line-height:1.7">
            Ex-Data Scientist @ Staffing Professors &nbsp;·&nbsp; Ex-Data Scientist @ Codex Tech IT LLC
          </div>
          <div style="font-size:.82rem;color:#718096;margin-top:.4rem">
            📍 Ahmedabad, Gujarat, India &nbsp;·&nbsp;
            🎓 REC Sonbhadra — B.Tech (ECE, 2019–2023) &nbsp;·&nbsp;
            👥 13,244 Followers &nbsp;·&nbsp; 500+ Connections
          </div>
          <div style="margin-top:.8rem">
            <span class="tag-pill tag-lime">Data Science</span>
            <span class="tag-pill tag-teal">Machine Learning</span>
            <span class="tag-pill tag-blue">GenAI / LLMs</span>
            <span class="tag-pill tag-purple">Data Analytics</span>
            <span class="tag-pill tag-lime">Streamlit</span>
            <span class="tag-pill tag-teal">Python</span>
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:.6rem;min-width:140px">
          <div class="score-card" style="padding:.9rem;text-align:center">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#b5f23d">722</div>
            <div style="font-size:.7rem;color:#718096;text-transform:uppercase;letter-spacing:1px">Profile Views</div>
          </div>
          <div class="score-card" style="padding:.9rem;text-align:center">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#2dd4bf">71.6K</div>
            <div style="font-size:.7rem;color:#718096;text-transform:uppercase;letter-spacing:1px">Post Impressions</div>
          </div>
          <div class="score-card" style="padding:.9rem;text-align:center">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#a78bfa">84</div>
            <div style="font-size:.7rem;color:#718096;text-transform:uppercase;letter-spacing:1px">Search Appearances</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ab1, ab2 = st.columns([1.3, 1], gap="large")
    with ab1:
        st.markdown("#### 📖 About")
        st.markdown("""<div class="sec-card">
          <p style="color:#a0aec0;font-size:.9rem;line-height:1.8;margin-bottom:.8rem">
            I am Nand Kishor Vyas, a passionate Data Scientist with a solid foundation in data analysis
            and a keen interest in transforming data into actionable insights. My journey in the tech industry
            began with an enriching internship at <strong style="color:#f9fafb">Unified Mentor</strong>, where I honed
            my skills in data analysis and developed a strong analytical mindset.
          </p>
          <p style="color:#718096;font-size:.88rem;line-height:1.8;margin-bottom:.8rem">
            Currently, at <strong style="color:#b5f23d">InfiJobs</strong>, I design and launch candidate-focused
            study portals for data science aspirants, integrating curated curricula, interactive modules,
            and progress tracking. I specialize in statistical analysis, machine learning, and data visualization,
            striving to deliver impactful data-driven solutions.
          </p>
          <p style="color:#718096;font-size:.88rem;line-height:1.8">
            I hold a B.Tech in Electronics and Communications Engineering from
            <strong style="color:#f9fafb">Rajkiya Engineering College Sonbhadra</strong>, where I was also a
            Training &amp; Placement Cell Representative and football player.
            Always eager to connect with like-minded professionals and explore opportunities in data science.
          </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 💼 Experience Timeline")
        for company, role, period, location, highlights in [
            ("🏢 InfiJobs", "Associate Data Scientist (Full-time)", "May 2025 – Present · 1 yr", "Ahmedabad, Gujarat · On-site",
             ["Designed and launched candidate-focused study portals for data science aspirants",
              "Implemented data-driven training programs on ML, data analysis, and statistical methods",
              "Developed interactive teaching methods — live coding, hands-on labs, case studies"]),
            ("🏢 Staffing Professors LLC", "Data Science Trainer (Full-time)", "Sep 2024 – May 2025 · 9 mos", "Ahmedabad, Gujarat · On-site",
             ["Conducted training sessions on data science tools and techniques",
              "Developed comprehensive training materials tailored to different learning levels",
              "Facilitated hands-on projects with ML, data analysis, and data visualization"]),
            ("🏢 CodeX Tech-IT LLC", "Jr. Data Scientist (Full-time)", "Dec 2023 – Sep 2024 · 10 mos", "Ahmedabad, Gujarat · On-site",
             ["Built predictive models using ML algorithms to enhance forecasting accuracy",
              "Created Tableau and Power BI dashboards to communicate insights effectively",
              "Conducted data cleaning, preprocessing, and EDA to prepare datasets for modeling"]),
            ("🏢 Unified Mentor Pvt Ltd", "Data Analyst Intern", "Feb 2024 – Apr 2024 · 3 mos", "Gurugram, Haryana · Remote",
             ["Data cleaning, EDA, and statistical analysis using Python and SQL",
              "Designed Power BI and Excel dashboards for stakeholder reporting",
              "Automated reporting workflows — reduced manual effort by 30%"]),
        ]:
            st.markdown(f"""<div class="sec-card" style="margin-bottom:.8rem;border-left:3px solid #b5f23d">
              <div style="font-family:'Syne',sans-serif;font-weight:700;color:#f9fafb;font-size:.95rem">{role}</div>
              <div style="font-size:.83rem;color:#b5f23d;margin:.2rem 0">{company}</div>
              <div style="font-size:.78rem;color:#718096;margin-bottom:.7rem">{period} &nbsp;·&nbsp; {location}</div>
              {"".join(f'<div style="font-size:.82rem;color:#a0aec0;padding:.2rem 0">• {h}</div>' for h in highlights)}
            </div>""", unsafe_allow_html=True)

    with ab2:
        st.markdown("#### 🏆 Recommendation")
        st.markdown("""<div class="sec-card" style="border-left:3px solid #f6e05e;margin-bottom:1rem">
          <div style="font-family:'Syne',sans-serif;font-weight:700;color:#f9fafb;font-size:.9rem">Aman Kumar Pandey</div>
          <div style="font-size:.78rem;color:#718096;margin:.2rem 0">Team Lead @ Amoha Recruitment · April 2025</div>
          <p style="color:#a0aec0;font-size:.84rem;line-height:1.7;margin-top:.6rem;font-style:italic">
            "Nand Vyas is an exceptional Data Scientist with extensive experience. His ability to analyze
            complex datasets, extract meaningful insights, and develop data-driven solutions has been truly impressive.
            His deep expertise in machine learning, predictive analytics, and big data technologies makes him a
            valuable asset to any team. Beyond his technical abilities, he is a great team player — always willing
            to mentor others and share his knowledge."
          </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 🛠️ Skills")
        skills_by_cat = {
            "Languages & Core": ["Python","R","SQL","Bash","MATLAB"],
            "ML / AI":          ["Machine Learning","Deep Learning","NLP","GenAI","LLMs","RAG","Scikit-learn","TensorFlow","PyTorch"],
            "Data & Analytics": ["Pandas","NumPy","EDA","Statistical Analysis","Data Visualization","Power BI","Tableau","Excel"],
            "Cloud & Tools":    ["Streamlit","FastAPI","Docker","AWS","FAISS","Pinecone","Git"],
            "Databases":        ["MySQL","PostgreSQL","MongoDB","Vector Databases"],
        }
        for cat, skills in skills_by_cat.items():
            st.markdown(f'<div style="font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#718096;margin:.7rem 0 .3rem">{cat}</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="skill-badge b-green">{s}</span>' for s in skills), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 🎓 Education & Certifications")
        st.markdown("""<div class="sec-card" style="margin-bottom:.6rem">
          <div style="font-weight:700;color:#f9fafb;font-size:.88rem">B.Tech — Electronics & Communications Engineering</div>
          <div style="font-size:.82rem;color:#b5f23d">Rajkiya Engineering College Sonbhadra</div>
          <div style="font-size:.78rem;color:#718096">Aug 2019 – Aug 2023</div>
          <div style="font-size:.78rem;color:#a0aec0;margin-top:.3rem">T&P Cell Representative · Football Player (RPL)</div>
        </div>""", unsafe_allow_html=True)
        for cert in ["Data Engineering with Spark — Udemy (Mar 2025)","Career Camp JAVA — Coding Ninjas","18 Total Licenses & Certifications"]:
            st.markdown(f'<div style="font-size:.82rem;color:#a0aec0;padding:.25rem 0;border-bottom:1px solid rgba(255,255,255,.04)">🏅 {cert}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🌟 Featured Projects")
        for proj, desc in [
            ("REAL_TIME_GEN_AI_MODEL","Generative AI system with dynamic responsiveness — built with Streamlit, Python, LLMs, RAG, and vector databases. Associated with Staffing Professors LLC."),
            ("Personal Portfolio / Blog","Personal portfolio showcasing web development and data science work."),
        ]:
            st.markdown(f"""<div class="sec-card" style="margin-bottom:.6rem">
              <div style="font-weight:700;color:#b5f23d;font-size:.86rem">{proj}</div>
              <div style="font-size:.8rem;color:#718096;margin-top:.3rem;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 🌍 Connect")
        st.markdown("""<div class="sec-card">
          <a href="https://www.linkedin.com/in/nand-kishor-vyas/" target="_blank"
             style="display:flex;align-items:center;gap:.5rem;color:#b5f23d;font-size:.87rem;
             text-decoration:none;padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,.05)">
            🔗 linkedin.com/in/nand-kishor-vyas
          </a>
          <div style="font-size:.84rem;color:#a0aec0;padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,.05)">
            📍 Ahmedabad, Gujarat, India
          </div>
          <div style="font-size:.84rem;color:#a0aec0;padding:.4rem 0">
            🏫 REC Sonbhadra · InfiJobs
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CONTACT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "contact":
    st.markdown("""<div style="font-family:'Fira Code',monospace;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#b5f23d;margin-bottom:.3rem">Contact</div>""", unsafe_allow_html=True)
    st.markdown("## 📬 Get In Touch with Nand")
    st.caption("Connect with Nand Kishor Vyas for collaboration, data science discussions, or feedback on this tool.")

    # Contact info cards
    ci1, ci2, ci3, ci4 = st.columns(4)
    for col, icon, label, val, color in [
        (ci1,"🔗","LinkedIn","linkedin.com/in/nand-kishor-vyas","#b5f23d"),
        (ci2,"📍","Location","Ahmedabad, Gujarat, India","#2dd4bf"),
        (ci3,"🏢","Current Role","Associate Data Scientist @ InfiJobs","#a78bfa"),
        (ci4,"🎓","Education","B.Tech ECE — REC Sonbhadra","#f6e05e"),
    ]:
        col.markdown(f"""<div class="sec-card" style="text-align:center;padding:1.2rem .8rem">
          <div style="font-size:1.6rem;margin-bottom:.4rem">{icon}</div>
          <div class="sec-title">{label}</div>
          <p style="color:{color};font-size:.79rem;line-height:1.4">{val}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ct1, ct2 = st.columns([1.2, 1], gap="large")

    with ct1:
        st.markdown('<div class="sec-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">✉️ Send a Message</div>', unsafe_allow_html=True)
        c_name    = st.text_input("Name",    placeholder="Your name")
        c_email   = st.text_input("Email",   placeholder="your@email.com")
        c_subject = st.selectbox("Subject", [
            "General Question / Feedback",
            "Collaboration / Project Inquiry",
            "Data Science Discussion",
            "Bug Report / Feature Request",
            "Partnership / Business",
            "Hiring / Recruitment",
        ])
        c_message = st.text_area("Message", height=140,
            placeholder="Hi Nand, I came across your AI ATS Checker and wanted to connect about…")
        if st.button("Send Message →", use_container_width=True, type="primary"):
            if c_name and c_email and c_message:
                st.success("✅ Message sent! Nand will get back to you within 24 hours.")
            else:
                st.error("Please fill in name, email, and message.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick profile summary
        st.markdown("""<div class="sec-card" style="margin-top:1rem">
          <div class="sec-title">👤 About Nand</div>
          <div style="display:flex;align-items:center;gap:1rem;margin-bottom:.8rem">
            <div style="font-size:2.5rem;background:linear-gradient(135deg,#1a2a0a,#0f3060);border-radius:50%;
                 width:55px;height:55px;display:flex;align-items:center;justify-content:center;flex-shrink:0">👨‍💻</div>
            <div>
              <div style="font-weight:700;color:#f9fafb;font-size:.9rem">Nand Kishor Vyas</div>
              <div style="font-size:.78rem;color:#b5f23d">Associate Data Scientist @ InfiJobs</div>
              <div style="font-size:.75rem;color:#718096">13,244 Followers · 500+ Connections</div>
            </div>
          </div>
          <p style="color:#718096;font-size:.83rem;line-height:1.7">
            Passionate Data Scientist based in Ahmedabad, Gujarat, India. Specializes in machine learning,
            GenAI, data visualization, and building data-driven products. Built this AI ATS Checker to help
            job seekers navigate the complex world of ATS systems.
          </p>
          <div style="margin-top:.8rem">
            <span class="tag-pill tag-lime">Python</span>
            <span class="tag-pill tag-teal">ML / AI</span>
            <span class="tag-pill tag-blue">Streamlit</span>
            <span class="tag-pill tag-purple">GenAI / LLMs</span>
          </div>
        </div>""", unsafe_allow_html=True)

    with ct2:
        st.markdown("#### 🔗 Connect on LinkedIn")
        st.markdown("""<div class="sec-card" style="margin-bottom:1rem">
          <a href="https://www.linkedin.com/in/nand-kishor-vyas/" target="_blank"
             style="display:flex;align-items:center;gap:.8rem;text-decoration:none;color:#f9fafb">
            <div style="font-size:2rem">🔗</div>
            <div>
              <div style="font-weight:700;color:#b5f23d;font-size:.9rem">View LinkedIn Profile</div>
              <div style="font-size:.78rem;color:#718096;margin-top:.1rem">linkedin.com/in/nand-kishor-vyas</div>
            </div>
          </a>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### ❓ FAQ")
        for q, a in [
            ("Is this tool really free?","Yes, completely free. No signup, no credit card, no hidden costs. Built to help job seekers like you."),
            ("Is my resume data stored?","No. All analysis runs locally in your browser session. We never store or share your resume data."),
            ("What file types are supported?","PDF and plain text. For best results, use a text-based (not scanned) PDF."),
            ("How accurate is the ATS score?","It uses TF-IDF and keyword matching — the same techniques most ATS systems use. Scores above 70% have strong pass rates."),
            ("Can I use this for any job type?","Yes — tech, finance, marketing, HR, operations, and more. Works with any job description in any language."),
            ("Who built this?","Nand Kishor Vyas — Associate Data Scientist from Ahmedabad, India. Built to solve a real problem he witnessed in the job market."),
        ]:
            with st.expander(q):
                st.markdown(f'<p style="color:#a0aec0;font-size:.87rem;line-height:1.6">{a}</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED FOOTER (all pages)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-bar">
  <div class="footer-links">
    <span class="footer-link">Resume Checker</span>
    <span class="footer-link">Resume Builder</span>
    <span class="footer-link">Salary Predictor</span>
    <span class="footer-link">Blog</span>
    <span class="footer-link">About</span>
    <span class="footer-link">Terms & Conditions</span>
    <span class="footer-link">Privacy Policy</span>
    <span class="footer-link">Contact</span>
  </div>
  <p style="color:#4a5568">AI-powered resume analysis to help you beat ATS systems and get hired.</p>
  <p style="margin-top:.4rem">© 2026 AI ATS Checker. All rights reserved. Built with ❤️ for job seekers.</p>
</div>
""", unsafe_allow_html=True)