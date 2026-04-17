# 🎯 AI ATS Resume Platform

An AI-powered web application that helps job seekers **analyze resumes, improve ATS scores, build resumes, and estimate salaries** — all in one place.

---

## 🚀 Features

### 📄 Resume Checker

* ATS-based resume scoring
* Semantic matching with job descriptions
* Skill gap analysis (matched, missing, bonus skills)
* Keyword overlap detection
* Resume section checker
* Smart recommendations & insights

### 🧾 Resume Builder

* Create structured, ATS-friendly resumes
* Input-based resume generation
* Clean and simple UI

### 💰 Salary Predictor

* Estimate salary based on:

  * Skills
  * Experience
* Quick insights into market value

### 🌐 Additional Pages

* ✍️ Blog (career tips & insights)
* 📞 Contact form
* ℹ️ About page

---<img width="1876" height="745" alt="image" src="https://github.com/user-attachments/assets/b1188bcc-dd08-4f09-97f0-d5f3f2e87454" />


## 🧠 Tech Stack

* **Frontend/UI**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Machine Learning / NLP**:

  * Scikit-learn (TF-IDF, Cosine Similarity)
  * Sentence Transformers (for semantic matching)
* **Visualization**: Plotly
* **PDF Parsing**: pdfplumber

---

## 📁 Project Structure

```
resume-analyzer/
│
├── app.py                     # Main landing page
├── pages/
│   ├── 1_Resume_Checker.py
│   ├── 2_Resume_Builder.py
│   ├── 3_Salary_Predictor.py
│   ├── 4_Blog.py
│   ├── 5_Contact.py
│   ├── 6_About.py
│
├── utils/                     # Helper functions
├── data/                      # Skills dataset
├── models/                    # ML logic (optional)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/resume-analyzer.git
cd resume-analyzer
```

### 2️⃣ Create virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the App

```
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 📊 How It Works

1. Upload your resume (PDF)
2. Paste job description
3. System analyzes:

   * Semantic similarity
   * Skills match
   * Keywords
   * Resume structure
4. Get:

   * ATS Score
   * Insights
   * Recommendations
   * Visual analytics

---

## 🎯 Use Cases

* Job seekers improving resumes
* Students preparing for placements
* Data analysts / developers applying for jobs
* Career coaches & recruiters

---

## 🚀 Future Enhancements

* 🔐 User authentication (login/signup)
* 📂 Save & manage resumes
* 📄 Download resume as PDF
* 🤖 AI resume rewriting (LLM-based)
* 🌍 Deploy on cloud (Streamlit Cloud / AWS)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

---

## 📞 Contact

For any queries or suggestions:

* Email: [your-email@example.com](mailto:your-email@example.com)
* LinkedIn: your-profile-link

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub!

---
<img width="1876" height="745" alt="image" src="https://github.com/user-attachments/assets/6cdcc489-4023-450a-a016-3be0d0327b96" />

**Built with ❤️ using Python & Streamlit**
