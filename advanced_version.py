# streamlit_ats_advanced.py
"""
Extremely advanced Streamlit app: ATS Score Checker + Keyword Searcher for multiple resumes
Features:
- Upload multiple resumes (PDF, DOCX, TXT)
- Upload or paste job description / keywords
- Parse resume text and detect sections (Experience, Education, Skills)
- Keyword matching + semantic similarity (Sentence-Transformers) with TF-IDF fallback
- Weighted ATS scoring with section importance and penalties
- "What to add" suggestions based on missing keywords
- Batch export results as CSV
- Interactive per-resume breakdown with highlighted matches
- Caching for embeddings, optional use of local sentence-transformers model

Requirements (install):
pip install streamlit pdfplumber python-docx docx2txt sentence-transformers scikit-learn pandas numpy nltk regex
(If you cannot install sentence-transformers, the app will fall back to TF-IDF semantic matching.)

Run: streamlit run streamlit_ats_advanced.py
"""

import streamlit as st
import pdfplumber
import docx2txt
import re
import io
import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import hashlib
import json

# Optional: use sentence-transformers for embeddings if installed
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    HAVE_EMBEDDINGS = True
except Exception:
    EMBEDDING_MODEL = None
    HAVE_EMBEDDINGS = False

# ---------- Utilities ----------

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

# Basic text extraction
def extract_text_from_pdf(file_stream) -> str:
    text = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return '\n'.join(text)

def extract_text_from_docx(file_stream) -> str:
    # docx2txt wants a path, so write to temp
    tmp_path = '/tmp/streamlit_docx_temp.docx'
    with open(tmp_path, 'wb') as f:
        f.write(file_stream.read())
    text = docx2txt.process(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return text

def extract_text(file) -> str:
    """Detect file type and extract text. `file` is an UploadedFile from Streamlit."""
    name = file.name.lower()
    file_bytes = file.read()
    stream = io.BytesIO(file_bytes)
    if name.endswith('.pdf'):
        return extract_text_from_pdf(stream)
    elif name.endswith('.docx'):
        # For docx, we need to pass bytes
        return extract_text_from_docx(io.BytesIO(file_bytes))
    elif name.endswith('.txt'):
        return file_bytes.decode('utf-8', errors='ignore')
    else:
        # Attempt PDF extraction then text fallback
        try:
            return extract_text_from_pdf(stream)
        except Exception:
            return file_bytes.decode('utf-8', errors='ignore')

# Section extraction heuristics
SECTION_PATTERNS = {
    'experience': re.compile(r"\b(experience|professional experience|work history|employment)\b", re.I),
    'education': re.compile(r"\b(education|academic|qualifications|degrees)\b", re.I),
    'skills': re.compile(r"\b(skills|technical skills|expertise|proficiencies)\b", re.I),
}

def split_sections(text: str) -> Dict[str, str]:
    # Heuristic: split by headings using uppercase lines or lines ending with ':' or known patterns
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections = {'header': '', 'experience': '', 'education': '', 'skills': '', 'other': ''}
    current = 'header'
    for ln in lines:
        # heading detection
        if len(ln) < 80 and ln.isupper():
            # uppercase line likely a heading
            matched = False
            for name, patt in SECTION_PATTERNS.items():
                if patt.search(ln):
                    current = name
                    matched = True
                    break
            if not matched:
                current = 'other'
            continue
        # pattern detection within line
        for name, patt in SECTION_PATTERNS.items():
            if patt.search(ln):
                current = name
                matched = True
                break
        sections[current] += ln + '\n'
    return sections

# Simple skill extractor using a skills list
DEFAULT_SKILLS = [
    # Short list; users can upload/extend their skill list file
    'python','java','c#','c++','sql','nosql','javascript','react','angular','node','django','flask',
    'aws','azure','gcp','docker','kubernetes','html','css','tensorflow','pytorch','nlp','computer vision',
]

def load_skill_list_from_text(text: str) -> List[str]:
    # Comma or newline separated
    parts = re.split(r'[\n,;]+', text)
    return [p.strip().lower() for p in parts if p.strip()]

def extract_skills_from_text(text: str, skill_list: List[str]) -> List[str]:
    lowered = text.lower()
    found = []
    for skill in skill_list:
        if skill in lowered:
            found.append(skill)
    return sorted(set(found), key=lambda s: lowered.count(s), reverse=True)

# Experience years extractor (heuristic)
def extract_years_of_experience(text: str) -> float:
    # look for patterns like 'X years' or 'X+ years' or date ranges
    m = re.findall(r"(\d+\.?\d*)\s*\+?\s*(?:years|yrs)", text, re.I)
    if m:
        nums = [float(x) for x in m]
        return max(nums)
    # date ranges like 2019-2022
    ranges = re.findall(r"(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}", text)
    if ranges:
        durations = []
        for a,b in ranges:
            try:
                durations.append(abs(int(b) - int(a)))
            except:
                pass
        if durations:
            return max(durations)
    return 0.0

# ---------- Matching & Scoring ----------

@dataclass
class ResumeResult:
    filename: str
    text: str
    sections: Dict[str,str]
    skills_found: List[str]
    years_exp: float
    raw_score: float
    detailed: Dict

# Embedding helpers
_embedding_cache = {}

def get_embedding(text: str):
    key = sha1_text(text)[:16]
    if key in _embedding_cache:
        return _embedding_cache[key]
    if HAVE_EMBEDDINGS:
        emb = EMBEDDING_MODEL.encode(text)
        _embedding_cache[key] = emb
        return emb
    else:
        return None


def semantic_similarity(a: str, b: str, tfidf_vectorizer=None, tfidf_matrix=None) -> float:
    # Try embeddings first
    if HAVE_EMBEDDINGS:
        ea = get_embedding(a)
        eb = get_embedding(b)
        if ea is None or eb is None:
            return 0.0
        sim = cosine_similarity([ea], [eb])[0][0]
        return float(sim)
    # Fallback: TF-IDF cosine on combined corpus
    if tfidf_vectorizer is not None:
        vecs = tfidf_vectorizer.transform([a,b])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(sim)
    return 0.0


def compute_ats_score(resume_text: str, jd_text: str, skill_list: List[str], weights: Dict[str,float], tfidf_vectorizer=None) -> Tuple[float, Dict]:
    # Keyword exact matches
    jd_keywords = sorted(set(re.findall(r"\b[\w\+#\-\.]+\b", jd_text.lower())))
    resume_tokens = resume_text.lower()
    exact_matches = [kw for kw in jd_keywords if kw in resume_tokens]
    exact_match_rate = len(exact_matches) / max(1, len(jd_keywords))

    # skills match
    skills_found = extract_skills_from_text(resume_text, skill_list)
    skills_req = [s for s in skill_list if s in jd_text.lower()]
    skills_match_rate = len(set(skills_found) & set(skills_req)) / max(1, len(set(skills_req))) if skills_req else 0.0

    # semantic similarity between resume and JD
    semantic_sim = semantic_similarity(resume_text, jd_text, tfidf_vectorizer)

    # years of experience matching (if JD mentions required years)
    required_years = 0.0
    m = re.search(r"(\d+)\+?\s*(?:years|yrs)", jd_text, re.I)
    if m:
        required_years = float(m.group(1))
    resume_years = extract_years_of_experience(resume_text)
    exp_score = min(1.0, resume_years / max(1e-6, required_years)) if required_years > 0 else 0.5 if resume_years>0 else 0.0

    # section importance: reward matches found inside Experience/Skills sections
    sections = split_sections(resume_text)
    exp_section = sections.get('experience','')
    skills_section = sections.get('skills','')
    in_exp_matches = sum(1 for kw in exact_matches if kw in exp_section.lower())
    in_skills_matches = sum(1 for kw in exact_matches if kw in skills_section.lower())
    section_bonus = (in_exp_matches * 0.02) + (in_skills_matches * 0.01)

    # Combine with weights
    score = (
        weights['exact'] * exact_match_rate +
        weights['skills'] * skills_match_rate +
        weights['semantic'] * semantic_sim +
        weights['experience'] * exp_score
    )
    score = float(max(0.0, min(1.0, score + section_bonus)))

    detailed = {
        'exact_matches_count': len(exact_matches),
        'exact_matches': exact_matches[:30],
        'exact_match_rate': exact_match_rate,
        'skills_found': skills_found,
        'skills_req': skills_req,
        'skills_match_rate': skills_match_rate,
        'semantic_similarity': semantic_sim,
        'required_years': required_years,
        'resume_years': resume_years,
        'experience_score': exp_score,
        'section_bonus': section_bonus,
    }
    return score, detailed

# ---------- Streamlit UI ----------

st.set_page_config(page_title='Advanced ATS Score Checker', layout='wide')
st.title('🔎 Advanced ATS Score Checker & Keyword Searcher')

with st.sidebar:
    st.header('Configuration')
    use_embeddings = st.checkbox('Use sentence-transformers embeddings (if available)', value=HAVE_EMBEDDINGS)
    if use_embeddings and not HAVE_EMBEDDINGS:
        st.warning('sentence-transformers not found — will fall back to TF-IDF.')
    st.markdown('---')
    st.header('Scoring weights (sum not required)')
    w_exact = st.slider('Exact keyword weight', 0.0, 2.0, 0.9, 0.01)
    w_skills = st.slider('Skills match weight', 0.0, 2.0, 0.6, 0.01)
    w_sem = st.slider('Semantic similarity weight', 0.0, 2.0, 0.3, 0.01)
    w_exp = st.slider('Experience match weight', 0.0, 2.0, 0.2, 0.01)
    weights = {'exact': w_exact, 'skills': w_skills, 'semantic': w_sem, 'experience': w_exp}

st.markdown("Upload resumes (PDF/DOCX/TXT). You can upload multiple files at once.")
files = st.file_uploader('Upload resumes', type=['pdf','docx','txt'], accept_multiple_files=True)

st.markdown('---')
col1, col2 = st.columns([2,1])
with col1:
    st.subheader('Job Description / Keywords')
    jd_text = st.text_area('Paste the job description or list of keywords', height=200)
    st.info('Tip: paste the JD to get best results. For quick checks, just paste comma-separated keywords.')
    skill_upload = st.file_uploader('Optional: upload a custom skills list (txt/csv)', type=['txt','csv'])
with col2:
    st.subheader('Search & Export')
    q = st.text_input('Search keyword across resumes (single keyword or phrase)')
    min_score = st.slider('Minimum ATS score to show', 0.0, 1.0, 0.0)
    st.markdown('---')
    if st.button('Export results to CSV'):
        st.session_state.get('export_csv', False)
        st.session_state['do_export'] = True

# Load skill list
skill_list = DEFAULT_SKILLS.copy()
if skill_upload is not None:
    try:
        content = skill_upload.getvalue().decode('utf-8')
    except Exception:
        content = skill_upload.getvalue().decode('latin-1')
    skill_list = load_skill_list_from_text(content) + skill_list

# Early checks
if not files:
    st.info('Upload one or more resumes to begin.')
    st.stop()

if not jd_text.strip():
    st.warning('Please paste a job description or keywords to compute ATS scores accurately.')
    st.stop()

# Prepare TF-IDF fallback
tfidf_vectorizer = None
if not HAVE_EMBEDDINGS or not use_embeddings:
    corpus = [jd_text]
    doc_texts = []
    for f in files:
        try:
            txt = extract_text(f)
        except Exception:
            txt = ''
        doc_texts.append(txt)
        corpus.append(txt)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000).fit(corpus)

# Process each resume
results = []
for f in files:
    try:
        text = extract_text(f)
    except Exception as e:
        text = ''
    sections = split_sections(text)
    skills_found = extract_skills_from_text(text, skill_list)
    years = extract_years_of_experience(text)
    score, detailed = compute_ats_score(text, jd_text, skill_list, weights, tfidf_vectorizer)
    res = ResumeResult(filename=f.name, text=text, sections=sections, skills_found=skills_found, years_exp=years, raw_score=score, detailed=detailed)
    results.append(res)

# Convert results to DataFrame
rows = []
for r in results:
    rows.append({
        'filename': r.filename,
        'ats_score': round(r.raw_score*100,2),
        'years_experience': r.years_exp,
        'skills_found': ', '.join(r.skills_found[:10]),
        'exact_matches_count': r.detailed['exact_matches_count']
    })

df = pd.DataFrame(rows).sort_values('ats_score', ascending=False)

# Filter by min_score
df_filtered = df[df['ats_score'] >= (min_score*100)]

# Search filter
if q.strip():
    qlow = q.lower()
    mask = df_filtered['filename'].apply(lambda fn: any(qlow in (res.filename.lower() if res.filename==fn else '') for res in results))
    # a better approach: search in texts
    mask = [qlow in next((res.text.lower() for res in results if res.filename==fn), '') for fn in df_filtered['filename']]
    df_filtered = df_filtered[mask]

st.subheader('Results')
st.dataframe(df_filtered.reset_index(drop=True))

# Detailed panels
for r in results:
    if r.raw_score < min_score:
        continue
    with st.expander(f"{r.filename} — ATS {round(r.raw_score*100,2)}%"):
        st.write('**Quick metrics**')
        st.write(f"Years experience (heuristic): {r.years_exp}")
        st.write(f"Skills found: {', '.join(r.skills_found) if r.skills_found else '—'}")
        st.write(f"Exact matches found: {r.detailed['exact_matches_count']}")
        st.write('---')
        st.write('**Matches (sample)**')
        st.write(', '.join(r.detailed['exact_matches'][:60]) if r.detailed['exact_matches'] else 'No exact keyword matches')
        st.write('---')
        st.write('**Section preview (Skills / Experience / Education)**')
        st.code('Skills:\n' + (r.sections.get('skills','')[:1000] or '—'))
        st.code('Experience:\n' + (r.sections.get('experience','')[:1000] or '—'))
        st.code('Education:\n' + (r.sections.get('education','')[:1000] or '—'))

        # Suggestions
        st.write('**Suggestions to improve ATS score**')
        missing = []
        jd_keywords = sorted(set(re.findall(r"\b[\w\+#\-\.]+\b", jd_text.lower())))
        for kw in jd_keywords:
            if kw not in r.text.lower():
                # heuristics: if kw is a skill or short word
                if len(kw) > 2:
                    missing.append(kw)
        if missing:
            st.write('Add or emphasise these keywords in relevant sections:')
            st.write(', '.join(missing[:60]))
        else:
            st.write('No obvious missing keywords found — resume already covers JD keywords.')

# Export CSV if requested
if st.session_state.get('do_export'):
    out_df = df[['filename','ats_score','years_experience','skills_found','exact_matches_count']]
    csv = out_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='ats_results.csv', mime='text/csv')
    st.session_state['do_export'] = False

st.markdown('---')
st.caption('Advanced ATS checker — heuristic and ML-assisted. For production-grade systems integrate OCR (Tesseract), richer resume parsing (e.g., spaCy models, resume parsers), and robust skill taxonomies.')
