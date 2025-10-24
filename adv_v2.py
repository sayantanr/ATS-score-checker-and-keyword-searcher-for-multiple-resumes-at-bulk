# streamlit_ats_advanced_v2.py
"""
Extremely advanced Streamlit app: ATS Score Checker + Keyword Searcher for multiple resumes
Version 2: integrates OCR (Tesseract), spaCy NER, pyresparser fallback, curated skill taxonomy upload,
JD keyword weighting, per-sentence suggestions, heuristic rewrites, and optional OpenAI rewrites/embeddings.

Run: streamlit run streamlit_ats_advanced_v2.py
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
import tempfile

# Optional heavy deps
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False

try:
    import spacy
    HAVE_SPACY = True
    try:
        NLP = spacy.load('en_core_web_sm')
    except Exception:
        NLP = None
except Exception:
    HAVE_SPACY = False
    NLP = None

try:
    from pyresparser import ResumeParser
    HAVE_PYRESPARSER = True
except Exception:
    HAVE_PYRESPARSER = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    HAVE_EMBEDDINGS = True
except Exception:
    EMBEDDING_MODEL = None
    HAVE_EMBEDDINGS = False

try:
    import openai
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

# ---------- Utilities ----------

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

# Text extraction
def extract_text_from_pdf(file_bytes) -> str:
    text = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception:
        pass
    joined = '\n'.join(text)
    if (not joined.strip()) and HAVE_OCR:
        try:
            images = convert_from_bytes(file_bytes)
            ocr_texts = []
            for img in images:
                ocr_texts.append(pytesseract.image_to_string(img))
            joined = '\n'.join(ocr_texts)
        except Exception:
            pass
    return joined


def extract_text_from_docx(file_bytes) -> str:
    tmp_path = os.path.join(tempfile.gettempdir(), 'streamlit_docx_temp.docx')
    with open(tmp_path, 'wb') as f:
        f.write(file_bytes)
    text = docx2txt.process(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return text


def extract_text(file) -> str:
    name = file.name.lower()
    file_bytes = file.read()
    if name.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    elif name.endswith('.txt'):
        return file_bytes.decode('utf-8', errors='ignore')
    else:
        try:
            return extract_text_from_pdf(file_bytes) or file_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return file_bytes.decode('utf-8', errors='ignore')

# Resume parsing helpers
SECTION_PATTERNS = {
    'experience': re.compile(r"\b(experience|professional experience|work history|employment)\b", re.I),
    'education': re.compile(r"\b(education|academic|qualifications|degrees)\b", re.I),
    'skills': re.compile(r"\b(skills|technical skills|expertise|proficiencies)\b", re.I),
}


def split_sections_heuristic(text: str) -> Dict[str, str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections = {'header': '', 'experience': '', 'education': '', 'skills': '', 'other': ''}
    current = 'header'
    for ln in lines:
        if len(ln) < 120 and (ln.isupper() or ln.endswith(':')):
            matched = False
            for name, patt in SECTION_PATTERNS.items():
                if patt.search(ln):
                    current = name
                    matched = True
                    break
            if not matched:
                current = 'other'
            continue
        for name, patt in SECTION_PATTERNS.items():
            if patt.search(ln):
                current = name
                break
        sections[current] += ln + '\n'
    return sections


def parse_with_pyresparser(file_path: str) -> Dict:
    try:
        data = ResumeParser(file_path).get_extracted_data()
        return data
    except Exception:
        return {}


def spacy_ner_parse(text: str) -> Dict:
    out = {'entities': [], 'titles': [], 'orgs': [], 'dates': [], 'degrees': []}
    if not HAVE_SPACY or NLP is None:
        return out
    doc = NLP(text)
    for ent in doc.ents:
        out['entities'].append((ent.text, ent.label_))
        if ent.label_ == 'ORG':
            out['orgs'].append(ent.text)
        if ent.label_ == 'DATE':
            out['dates'].append(ent.text)
    deg_patt = re.compile(r"\b(Bachelor|Master|B\.Sc|M\.Sc|B\.Tech|MBA|PhD|Doctor)\b", re.I)
    for m in deg_patt.finditer(text):
        out['degrees'].append(m.group(0))
    title_patt = re.compile(r"(Manager|Engineer|Director|Developer|Analyst|Consultant|Lead|Intern|Associate)", re.I)
    for m in title_patt.finditer(text):
        out['titles'].append(m.group(0))
    for k in out:
        out[k] = list(dict.fromkeys(out[k]))
    return out

# Skills
DEFAULT_SKILLS = [
    'python','java','c#','c++','sql','nosql','javascript','react','angular','node','django','flask',
    'aws','azure','gcp','docker','kubernetes','html','css','tensorflow','pytorch','nlp','computer vision',
]

def load_skill_list_from_text(text: str) -> List[str]:
    parts = re.split(r'[\n,;]+', text)
    return [p.strip().lower() for p in parts if p.strip()]

# Matching & scoring
@dataclass
class ResumeResult:
    filename: str
    text: str
    parsed: Dict
    sections: Dict[str,str]
    skills_found: List[str]
    years_exp: float
    raw_score: float
    detailed: Dict

_embedding_cache = {}

def get_embedding(text: str):
    key = sha1_text(text)[:16]
    if key in _embedding_cache:
        return _embedding_cache[key]
    if HAVE_EMBEDDINGS:
        emb = EMBEDDING_MODEL.encode(text)
        _embedding_cache[key] = emb
        return emb
    return None


def semantic_similarity(a: str, b: str, tfidf_vectorizer=None) -> float:
    if HAVE_EMBEDDINGS:
        ea = get_embedding(a)
        eb = get_embedding(b)
        if ea is None or eb is None:
            return 0.0
        sim = cosine_similarity([ea], [eb])[0][0]
        return float(sim)
    if tfidf_vectorizer is not None:
        vecs = tfidf_vectorizer.transform([a,b])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(sim)
    return 0.0


def extract_skills_from_text(text: str, skill_list: List[str]) -> List[str]:
    lowered = text.lower()
    found = []
    for skill in skill_list:
        if re.search(r"\b"+re.escape(skill)+r"\b", lowered):
            found.append(skill)
    return sorted(set(found), key=lambda s: lowered.count(s), reverse=True)


def extract_years_of_experience(text: str) -> float:
    m = re.findall(r"(\d+\.?\d*)\s*\+?\s*(?:years|yrs)", text, re.I)
    if m:
        nums = [float(x) for x in m]
        return max(nums)
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


def compute_ats_score(resume_text: str, jd_text: str, skill_list: List[str], weights: Dict[str,float], keyword_weights: Dict[str,float], tfidf_vectorizer=None) -> Tuple[float, Dict]:
    jd_keywords_raw = sorted(set(re.findall(r"\b[\w\+#\-\.]+\b", jd_text.lower())))
    jd_keywords = [kw for kw in jd_keywords_raw if len(kw) > 1]
    resume_lower = resume_text.lower()
    matched = {}
    total_kw_weight = 0.0
    matched_weight = 0.0
    for kw in jd_keywords:
        kw_weight = keyword_weights.get(kw, 1.0)
        total_kw_weight += kw_weight
        if re.search(r"\b"+re.escape(kw)+r"\b", resume_lower):
            matched[kw] = True
            matched_weight += kw_weight
        else:
            matched[kw] = False
    exact_match_rate = matched_weight / max(1e-6, total_kw_weight)
    skills_found = extract_skills_from_text(resume_text, skill_list)
    skills_req = [s for s in skill_list if re.search(r"\b"+re.escape(s)+r"\b", jd_text.lower())]
    skills_match_rate = len(set(skills_found) & set(skills_req)) / max(1, len(set(skills_req))) if skills_req else 0.0
    semantic_sim = semantic_similarity(resume_text, jd_text, tfidf_vectorizer)
    required_years = 0.0
    m = re.search(r"(\d+)\+?\s*(?:years|yrs)", jd_text, re.I)
    if m:
        required_years = float(m.group(1))
    resume_years = extract_years_of_experience(resume_text)
    exp_score = min(1.0, resume_years / max(1e-6, required_years)) if required_years > 0 else (0.7 if resume_years>0 else 0.0)
    sections = split_sections_heuristic(resume_text)
    exp_section = sections.get('experience','')
    skills_section = sections.get('skills','')
    in_exp_matches = sum(1 for kw,v in matched.items() if v and kw in exp_section.lower())
    in_skills_matches = sum(1 for kw,v in matched.items() if v and kw in skills_section.lower())
    section_bonus = (in_exp_matches * 0.02) + (in_skills_matches * 0.01)
    misplaced_penalty = 0.0
    for kw,v in matched.items():
        if v and (kw not in exp_section.lower() and kw not in skills_section.lower()):
            misplaced_penalty += 0.005
    score = (
        weights['exact'] * exact_match_rate +
        weights['skills'] * skills_match_rate +
        weights['semantic'] * semantic_sim +
        weights['experience'] * exp_score
    )
    score = float(max(0.0, min(1.0, score + section_bonus - misplaced_penalty)))
    detailed = {
        'keyword_weights_total': total_kw_weight,
        'exact_matched_weight': matched_weight,
        'exact_matches': [k for k,v in matched.items() if v][:200],
        'exact_miss': [k for k,v in matched.items() if not v][:200],
        'exact_match_rate': exact_match_rate,
        'skills_found': skills_found,
        'skills_req': skills_req,
        'skills_match_rate': skills_match_rate,
        'semantic_similarity': semantic_sim,
        'required_years': required_years,
        'resume_years': resume_years,
        'experience_score': exp_score,
        'section_bonus': section_bonus,
        'misplaced_penalty': misplaced_penalty,
    }
    return score, detailed

# Per-sentence suggestions

def sentence_suggestions(sentence: str) -> List[str]:
    suggestions = []
    words = sentence.split()
    if len(words) > 25:
        suggestions.append('This sentence is long; consider splitting it into two for readability.')
    if re.search(r"\b(I|we|my|our)\b", sentence):
        suggestions.append('Avoid starting sentences with first-person pronouns; focus on achievements and results.')
    if not re.search(r"\d", sentence):
        suggestions.append('Consider adding quantifiable metrics (e.g., percentages, counts) to demonstrate impact.')
    if re.search(r"responsible for|worked on|involved in", sentence, re.I):
        suggestions.append('Use stronger action verbs and concrete achievements rather than passive descriptions.')
    return suggestions

# Streamlit UI
st.set_page_config(page_title='Advanced ATS Score Checker', layout='wide')
st.title('🔎 Advanced ATS Score Checker & Resume Analyzer (v2)')

with st.sidebar:
    st.header('Configuration')
    use_embeddings = st.checkbox('Use local sentence-transformers embeddings (if available)', value=HAVE_EMBEDDINGS)
    use_openai_embeddings = st.checkbox('Use OpenAI embeddings (optional) — provides rewrites if API key supplied', value=False)
    openai_key = ''
    if use_openai_embeddings:
        openai_key = st.text_input('Paste OpenAI API key (will not be stored)', type='password')
        if openai_key and HAVE_OPENAI:
            openai.api_key = openai_key
    st.markdown('---')
    st.header('Scoring weights (relative importance)')
    w_exact = st.slider('Exact keyword weight', 0.0, 2.0, 0.9, 0.01)
    w_skills = st.slider('Skills match weight', 0.0, 2.0, 0.6, 0.01)
    w_sem = st.slider('Semantic similarity weight', 0.0, 2.0, 0.3, 0.01)
    w_exp = st.slider('Experience match weight', 0.0, 2.0, 0.2, 0.01)
    weights = {'exact': w_exact, 'skills': w_skills, 'semantic': w_sem, 'experience': w_exp}

st.markdown('Upload resumes (PDF/DOCX/TXT). The app will attempt OCR for scanned PDFs if Tesseract is installed.')
files = st.file_uploader('Upload resumes', type=['pdf','docx','txt'], accept_multiple_files=True)

st.markdown('---')
col1, col2 = st.columns([2,1])
with col1:
    st.subheader('Job Description / Keywords')
    jd_text = st.text_area('Paste the job description or list of keywords', height=250)
    st.info('Tip: paste the JD to get best results. For quick checks, paste comma-separated keywords.')
    jd_upload = st.file_uploader('Or upload JD (txt/pdf/docx)', type=['txt','pdf','docx'])
    if jd_upload and not jd_text.strip():
        jd_text = extract_text(jd_upload)
    skill_upload = st.file_uploader('Optional: upload a curated skills taxonomy (CSV or TXT)', type=['txt','csv'])
with col2:
    st.subheader('Search & JD keyword weights')
    q = st.text_input('Search keyword across resumes (single keyword or phrase)')
    min_score = st.slider('Minimum ATS score to show (%)', 0, 100, 0)
    st.markdown('---')
    if st.button('Export results to CSV'):
        st.session_state['do_export'] = True

# Load skill list
skill_list = DEFAULT_SKILLS.copy()
if skill_upload is not None:
    try:
        content = skill_upload.getvalue().decode('utf-8')
    except Exception:
        content = skill_upload.getvalue().decode('latin-1')
    try:
        df_sk = pd.read_csv(io.StringIO(content))
        skill_candidates = df_sk.iloc[:,0].astype(str).tolist()
    except Exception:
        skill_candidates = load_skill_list_from_text(content)
    skill_list = list(dict.fromkeys([s.lower() for s in skill_candidates] + skill_list))

if not files:
    st.info('Upload one or more resumes to begin.')
    st.stop()

if not jd_text.strip():
    st.warning('Please paste a job description or keywords to compute ATS scores accurately.')
    st.stop()

# Build JD keywords and default weights
jd_keywords = sorted(set(re.findall(r"\b[\w\+#\-\.]+\b", jd_text.lower())))
jd_keywords = [k for k in jd_keywords if len(k) > 1]
if 'keyword_weights' not in st.session_state:
    st.session_state['keyword_weights'] = {k: 1.0 for k in jd_keywords}
else:
    for k in jd_keywords:
        if k not in st.session_state['keyword_weights']:
            st.session_state['keyword_weights'][k] = 1.0
if jd_keywords:
    st.subheader('JD Keywords — set importance')
    kw_df = pd.DataFrame([{'keyword':k, 'weight':st.session_state['keyword_weights'].get(k,1.0)} for k in jd_keywords])
    edited = st.experimental_data_editor(kw_df, num_rows='dynamic')
    for _,row in edited.iterrows():
        w = float(row['weight']) if row['weight'] else 1.0
        st.session_state['keyword_weights'][row['keyword']] = w

# TF-IDF fallback
tfidf_vectorizer = None
if not HAVE_EMBEDDINGS or not use_embeddings:
    corpus = [jd_text]
    for f in files:
        try:
            txt = extract_text(f)
        except Exception:
            txt = ''
        corpus.append(txt)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000).fit(corpus)

# Process resumes
results = []
for f in files:
    try:
        text = extract_text(f)
    except Exception:
        text = ''
    parsed = {}
    tmp_path = None
    if HAVE_PYRESPARSER:
        try:
            tmp_path = os.path.join(tempfile.gettempdir(), f'{sha1_text(f.name)}.tmp')
            with open(tmp_path, 'wb') as wf:
                wf.write(f.getbuffer())
            parsed = parse_with_pyresparser(tmp_path)
        except Exception:
            parsed = {}
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
            except Exception:
                pass
    ner = spacy_ner_parse(text)
    if 'skills' in parsed and parsed['skills']:
        skills_found = [s.lower() for s in parsed['skills']]
    else:
        skills_found = extract_skills_from_text(text, skill_list)
    if 'total_experience' in parsed and parsed['total_experience']:
        years = parsed['total_experience']
    else:
        years = extract_years_of_experience(text)
    sections = split_sections_heuristic(text)
    score, detailed = compute_ats_score(text, jd_text, skill_list, weights, st.session_state['keyword_weights'], tfidf_vectorizer)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())[:200]
    sent_suggestions = []
    for s in sentences[:80]:
        sug = sentence_suggestions(s)
        if sug:
            sent_suggestions.append({'sentence': s[:400], 'suggestions': sug})
    res = ResumeResult(filename=f.name, text=text, parsed={'pyres': parsed, 'ner': ner}, sections=sections, skills_found=skills_found, years_exp=years, raw_score=score, detailed=detailed)
    res.detailed['sentence_suggestions'] = sent_suggestions
    results.append(res)

# Results table
rows = []
for r in results:
    rows.append({
        'filename': r.filename,
        'ats_score': round(r.raw_score*100,2),
        'years_experience': r.years_exp,
        'skills_found': ', '.join(r.skills_found[:10]),
        'exact_matches_count': len(r.detailed['exact_matches'])
    })

df = pd.DataFrame(rows).sort_values('ats_score', ascending=False)

df_filtered = df[df['ats_score'] >= min_score]
if q.strip():
    qlow = q.lower()
    mask = [qlow in next((res.text.lower() for res in results if res.filename==fn), '') for fn in df_filtered['filename']]
    df_filtered = df_filtered[mask]

st.subheader('Results')
st.dataframe(df_filtered.reset_index(drop=True))

for r in results:
    if r.raw_score*100 < min_score:
        continue
    with st.expander(f"{r.filename} — ATS {round(r.raw_score*100,2)}%"):
        st.write('**Quick metrics**')
        st.write(f"Years experience (heuristic): {r.years_exp}")
        st.write(f"Skills found: {', '.join(r.skills_found) if r.skills_found else '—'}")
        st.write(f"Exact matches found: {len(r.detailed['exact_matches'])}")
        st.write('---')
        st.write('**Top exact matches (sample)**')
        st.write(', '.join(r.detailed['exact_matches'][:60]) if r.detailed['exact_matches'] else 'No exact keyword matches')
        st.write('Missing important keywords (sample):')
        st.write(', '.join(r.detailed['exact_miss'][:60]) if r.detailed['exact_miss'] else '—')
        st.write('---')
        st.write('**NER / Parsed info (spaCy / pyresparser)**')
        st.write(r.parsed)
        st.write('---')
        st.write('**Section preview (Skills / Experience / Education)**')
        st.code('Skills:\n' + (r.sections.get('skills','')[:1000] or '—'))
        st.code('Experience:\n' + (r.sections.get('experience','')[:1000] or '—'))
        st.code('Education:\n' + (r.sections.get('education','')[:1000] or '—'))

        st.write('**Per-sentence suggestions (sample)**')
        for s in r.detailed.get('sentence_suggestions', [])[:8]:
            st.write('- Sentence:')
            st.write(s['sentence'])
            st.write('  Suggestions:')
            for sug in s['suggestions']:
                st.write('   - ' + sug)
        st.write('---')
        st.write('**Suggested keyword additions or emphasis**')
        if r.detailed['exact_miss']:
            st.write(', '.join(r.detailed['exact_miss'][:80]))
        else:
            st.write('Looks comprehensive for JD keywords.')

        st.write('**Rewrite examples (automated)**')
        sample_sent = (r.detailed.get('exact_matches')[:1] and r.detailed.get('exact_matches')[0]) or ''
        if sample_sent:
            st.write('Original (snippet):', sample_sent)
            if openai_key and HAVE_OPENAI:
                try:
                    prompt = f"Rewrite the following resume bullet to be more results-oriented and ATS-friendly:\n\n{sample_sent}"
                    resp = openai.Completion.create(engine='text-davinci-003', prompt=prompt, max_tokens=150)
                    rewrite = resp.choices[0].text.strip()
                    st.write('Rewrite (OpenAI):')
                    st.write(rewrite)
                except Exception as e:
                    st.write('OpenAI rewrite failed:', str(e))
            else:
                rewrite = sample_sent
                rewrite = re.sub(r"responsible for", "Led", rewrite, flags=re.I)
                rewrite = re.sub(r"worked on", "Delivered", rewrite, flags=re.I)
                st.write('Heuristic rewrite:')
                st.write(rewrite)

if st.session_state.get('do_export'):
    out_df = df[['filename','ats_score','years_experience','skills_found','exact_matches_count']]
    csv = out_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='ats_results_v2.csv', mime='text/csv')
    st.session_state['do_export'] = False

st.markdown('---')
st.caption('v2 — OCR, spaCy NER, pyresparser fallback, JD keyword weighting, and per-sentence suggestions.')
