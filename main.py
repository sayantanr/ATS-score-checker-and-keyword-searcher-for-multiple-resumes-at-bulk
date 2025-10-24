import streamlit as st
import PyPDF2
import re
from collections import Counter
import pandas as pd

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.lower()  # Convert to lowercase for matching

# Function to extract text from TXT
def extract_text_from_txt(txt_file):
    return txt_file.read().decode('utf-8').lower()

# Function to calculate ATS score based on keyword matches
def calculate_ats_score(resume_text, keywords):
    if not keywords:
        return 0
    keyword_list = [kw.lower().strip() for kw in keywords.split(',')]
    matches = []
    for kw in keyword_list:
        if re.search(r'\b' + re.escape(kw) + r'\b', resume_text):
            matches.append(kw)
    score = (len(matches) / len(keyword_list)) * 100 if keyword_list else 0
    return score, matches

# Streamlit App
st.title("ATS Score Checker and Keyword Searcher for Multiple Resumes")

# Sidebar for inputs
st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF or TXT files", 
    type=['pdf', 'txt'], 
    accept_multiple_files=True
)

st.sidebar.header("Job Keywords")
job_keywords = st.sidebar.text_area(
    "Enter keywords from the job description (comma-separated):",
    placeholder="e.g., Python, machine learning, SQL, leadership"
)

if st.sidebar.button("Analyze Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    elif not job_keywords:
        st.warning("Please enter some keywords.")
    else:
        results = []
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(uploaded_file)
            elif filename.endswith('.txt'):
                resume_text = extract_text_from_txt(uploaded_file)
            else:
                st.warning(f"Unsupported file type: {filename}")
                continue
            
            score, matched_keywords = calculate_ats_score(resume_text, job_keywords)
            results.append({
                'Resume': filename,
                'ATS Score (%)': round(score, 2),
                'Matched Keywords': ', '.join(matched_keywords),
                'Total Keywords': len(job_keywords.split(',')),
                'Resume Text Preview': resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
            })
        
        if results:
            df = pd.DataFrame(results)
            st.subheader("Results")
            st.dataframe(df[['Resume', 'ATS Score (%)', 'Matched Keywords']])
            
            # Display detailed previews
            for _, row in df.iterrows():
                with st.expander(f"Details for {row['Resume']} (Score: {row['ATS Score (%)']}%)"):
                    st.write("**Matched Keywords:**", row['Matched Keywords'])
                    st.write("**Preview:**", row['Resume Text Preview'])
            
            # Summary chart
            st.subheader("ATS Scores Summary")
            chart_data = df.set_index('Resume')['ATS Score (%)']
            st.bar_chart(chart_data)
