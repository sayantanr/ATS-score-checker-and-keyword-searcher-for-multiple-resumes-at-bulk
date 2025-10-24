# ATS Score Checker and Keyword Searcher for Multiple Resumes( simplest version main.py)

A Streamlit web application designed to analyze multiple resumes (PDF or TXT files) against job description keywords. It calculates an ATS (Applicant Tracking System) compatibility score based on keyword matches and provides a summary dashboard with tables, detailed previews, and visualizations.

## Features
- **Multi-File Upload**: Upload and process multiple PDF or TXT resumes simultaneously.
- **Keyword Matching**: Enter comma-separated keywords from a job description to search for exact word matches in resumes.
- **ATS Score Calculation**: Computes a percentage score based on the proportion of matched keywords (e.g., 80% if 4 out of 5 keywords are found).
- **Results Dashboard**: 
  - Interactive table showing resume names, scores, and matched keywords.
  - Expandable sections for detailed previews and full matched keyword lists.
  - Bar chart summarizing ATS scores across all resumes.
- **Error Handling**: Graceful handling of unsupported file types or reading errors.
- **Case-Insensitive Matching**: Searches are performed in lowercase for better accuracy.

## Prerequisites
- Python 3.8+ (tested on Python 3.12).
- Streamlit, PyPDF2, and Pandas libraries.

## Installation
1. Clone or download this repository:
   ```
   git clone <your-repo-url>
   cd ats-resume-analyzer
   ```
   
2. Install the required dependencies:
   ```
   pip install streamlit PyPDF2 pandas
   ```

3. Save the main application code to a file named `app.py` in your project directory.

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
   This will launch the app in your default web browser (typically at `http://localhost:8501`).

2. In the app sidebar:
   - Upload one or more PDF or TXT resume files.
   - Enter job keywords in the text area (e.g., `Python, machine learning, SQL, leadership`).

3. Click **Analyze Resumes** to process the files.

4. View results:
   - A table displays scores and matches.
   - Expand rows for previews.
   - A bar chart shows score distribution.

### Example Workflow
- Upload resumes: `resume1.pdf`, `resume2.txt`.
- Keywords: `data science, Python, AWS, agile`.
- Output: Scores like 75% for `resume1.pdf` (matched: `data science, Python, AWS`).

## Limitations
- Only supports PDF and TXT formats (PDF text extraction may vary based on document complexity).
- Keyword matching is exact-word only (e.g., "Python" won't match "Pythonic").
- No advanced NLP; scores are simple percentage-based.
- Previews are truncated to 500 characters for brevity.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss.

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/) for the UI.
- PDF handling via [PyPDF2](https://pypdf2.readthedocs.io/en/stable/).
- Data display with [Pandas](https://pandas.pydata.org/).

For issues or questions, open a GitHub issue or contact the maintainer. Last updated: October 24, 2025.
