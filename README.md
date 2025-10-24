# Advanced Streamlit ATS Score Checker (v2)

A powerful, AI-driven resume analyzer built with **Streamlit** that checks multiple resumes against job descriptions using advanced techniques like **OCR**, **NER**, **semantic matching**, and **weighted ATS scoring**. It helps recruiters and job seekers evaluate resume–JD fit, highlight missing keywords, and suggest ATS-friendly rewrites.

---

## 🚀 Features

✅ **Multi-Resume Upload**

* Supports `.pdf`, `.docx`, and `.txt` formats.
* Batch upload and evaluation with CSV export.

✅ **Advanced Text Extraction**

* Uses `pdfplumber` and `docx2txt` for clean text extraction.
* Built-in **OCR via Tesseract** for scanned PDFs.

✅ **Curated Skill Taxonomy**

* Upload or merge a skill taxonomy `.csv` to enhance keyword detection.
* Default skill list included.

✅ **Named Entity Recognition (NER)**

* Integrates **spaCy** for extracting:

  * Organizations (companies)
  * Institutions
  * Degrees and designations
  * Dates (for experience calculation)

✅ **Resume Parsing**

* Optional **pyresparser** integration for structured parsing.
* Fallback to heuristic section parsing when unavailable.

✅ **Keyword Weighting**

* Define and edit **weights per JD keyword**.
* Marks critical vs nice-to-have keywords.

✅ **Semantic Similarity**

* Uses **Sentence-Transformers** (`all-MiniLM-L6-v2`) for embedding-based comparison.
* Falls back to **TF–IDF cosine similarity** when not available.

✅ **ATS Scoring**

* Combines multiple metrics:

  * Exact keyword match
  * Skill overlap
  * Semantic relevance
  * Experience level
* Customizable weights in sidebar.

✅ **Rewrite & Improvement Suggestions**

* Highlights missing keywords and formatting issues.
* Provides per-sentence improvement suggestions.
* Optional **OpenAI integration** for high-quality rewrites.

✅ **Beautiful Streamlit UI**

* Clean interface with sidebar controls.
* Downloadable CSV reports for multiple resumes.

---

## 🧠 Tech Stack

* **Frontend:** Streamlit
* **NLP & AI:** spaCy, SentenceTransformers, scikit-learn
* **Parsing & OCR:** pdfplumber, docx2txt, pyresparser, pytesseract, pdf2image
* **Data Handling:** pandas, numpy

---

## 📦 Installation

1. Clone this repo:

   ```bash
   git clone 
   cd streamlit-ats-advanced
   ```

2. Install core dependencies:

   ```bash
   pip install streamlit pdfplumber docx2txt scikit-learn pandas numpy
   ```

3. Install optional (recommended) components:

   ```bash
   pip install pytesseract pdf2image pillow spacy pyresparser sentence-transformers openai
   python -m spacy download en_core_web_sm
   ```

4. **Install Tesseract OCR:**

   * **Windows:** [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
   * **Linux (Ubuntu):** `sudo apt install tesseract-ocr`
   * **macOS:** `brew install tesseract`

---

## ▶️ Usage

Run the app locally:

```bash
streamlit run streamlit_ats_advanced_v2.py
```

Then open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 🧩 Optional Configuration

* **API Key:**
  Add your OpenAI API key in the UI sidebar for rewrite suggestions.

* **Skill Taxonomy:**
  Upload a `.csv` or `.txt` with your industry-specific skills.

---

## 📊 Output

* Individual resume analysis with:

  * ATS Score
  * Matched and missing keywords
  * Semantic similarity
  * Experience estimation
  * Suggested rewrites
* Downloadable **CSV report** with all scores.

---

## 🧩 Roadmap

* [ ] Add downloadable PDF report per resume.
* [ ] Implement skill taxonomy manager.
* [ ] Add dashboard analytics for HR teams.
* [ ] Integrate with applicant-tracking systems (API-based).
* [ ] Fine-tune semantic embeddings for resume–JD domain.

---

## 🛠️ Troubleshooting

* **OCR not working?**
  Ensure Tesseract is installed and available in your system path.

* **Slow processing?**
  Disable semantic similarity or OCR in the sidebar.

* **`pyresparser` issues?**
  It’s optional — the app will automatically fall back to heuristic parsing.

---

## 🧑‍💻 Author

**Sayantan Roy**
Graduate Software Engineer @ Cognizant
Specialized in .NET, SQL Server, ServiceNow, AI/ML, and NLP Research.
📧 [sayantanr32@gmail.com](mailto:sayantanr32@gmail.com)

---

## 📜 License

This project is licensed under the **MIT License**.

---

### 💡 “Built to bridge the gap between humans and machines — making resumes smarter and hiring faster.”
