import pdfplumber

def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file object (Streamlit UploadedFile or path)."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        return f"ERROR: Could not read PDF — {e}"
    return text.strip()