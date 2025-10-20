import os
from flask import Flask, render_template, request
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- OCR and File Handling Imports ---
import fitz
from docx import Document
from PIL import Image
import io
# --- REVERTING TO THE SIMPLER PYTESSERACT ---
import pytesseract

# --- NO TESSERACT CONFIGURATION NEEDED ON LINUX ---

# Load the spaCy model once
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

# --- FILE READING FUNCTIONS ---

def extract_text_from_docx(file_stream):
    doc = Document(file_stream)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_stream):
    text = ""
    with fitz.open(stream=file_stream, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_txt(file_stream):
    return file_stream.read().decode('utf-8')

# --- REVISED IMAGE EXTRACTION FUNCTION ---
def extract_text_from_image(file_stream):
    """Reads an image file from an in-memory stream using pytesseract."""
    image = Image.open(file_stream)
    return pytesseract.image_to_string(image)

def extract_skills_and_name(text):
    doc = nlp(text)
    name = "Unknown Candidate"
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    skills = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2]
    from collections import Counter
    top_skills = [item[0] for item in Counter(skills).most_common(15)]
    return name, top_skills

# --- MAIN LOGIC (remains the same) ---
def calculate_weighted_score(resume_text, job_description_text, resume_skills):
    # ... (omitted for brevity, same as before)
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([job_description_text, resume_text])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        similarity_score = 0.0
    jd_doc = nlp(job_description_text)
    jd_skills = {token.text.lower() for token in jd_doc if token.pos_ in ['NOUN', 'PROPN']}
    resume_skills_set = {skill.lower() for skill in resume_skills}
    matched_skills_count = len(jd_skills.intersection(resume_skills_set))
    skill_score = matched_skills_count / len(jd_skills) if jd_skills else 0
    experience_keywords = ['senior', 'lead', 'manager', 'expert', 'years', 'project']
    found_keywords_count = sum(1 for keyword in experience_keywords if keyword in resume_text.lower())
    experience_score = found_keywords_count / len(experience_keywords) if experience_keywords else 0
    final_score = (similarity_score * 0.5) + (skill_score * 0.3) + (experience_score * 0.2)
    return min(round(final_score * 100, 2), 99.0)


# --- FLASK ROUTES (remains the same) ---
@app.route('/')
def index():
    # ... (omitted for brevity, same as before)
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # ... (omitted for brevity, same as before)
    uploaded_files = request.files.getlist('resumes')
    job_description = request.form['jd'].lower()
    candidates = []
    for file in uploaded_files:
        if file and file.filename:
            try:
                file_stream = io.BytesIO(file.read())
                filename_lower = file.filename.lower()
                resume_text = ""
                if filename_lower.endswith('.pdf'):
                    resume_text = extract_text_from_pdf(file_stream)
                elif filename_lower.endswith('.docx'):
                    resume_text = extract_text_from_docx(file_stream)
                elif filename_lower.endswith('.txt'):
                    resume_text = extract_text_from_txt(file_stream)
                elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
                    resume_text = extract_text_from_image(file_stream)
                else:
                    continue
                if resume_text:
                    name, skills = extract_skills_and_name(resume_text)
                    score = calculate_weighted_score(resume_text.lower(), job_description, skills)
                    candidates.append({'name': name, 'skills': skills, 'score': score})
            except Exception as e:
                print(f"FAILED TO PROCESS FILE: {file.filename}, Error: {e}")
    ranked_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    return render_template('results.html', candidates=ranked_candidates)

if __name__ == '__main__':
    app.run(debug=True)