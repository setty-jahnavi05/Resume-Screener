import os
import PyPDF2
import docx
import spacy
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to compute similarity
def compute_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity_score * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'resume' not in request.files or 'job_desc' not in request.files:
            return "No file uploaded"
        
        resume = request.files['resume']
        job_desc = request.files['job_desc']
        
        if resume.filename == '' or job_desc.filename == '':
            return "No selected file"
        
        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
        job_desc_path = os.path.join(app.config['UPLOAD_FOLDER'], job_desc.filename)
        
        resume.save(resume_path)
        job_desc.save(job_desc_path)
        
        # Extract text
        if resume.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_path)
        elif resume.filename.endswith('.docx'):
            resume_text = extract_text_from_docx(resume_path)
        else:
            return "Unsupported file format"
        
        if job_desc.filename.endswith('.pdf'):
            job_desc_text = extract_text_from_pdf(job_desc_path)
        elif job_desc.filename.endswith('.docx'):
            job_desc_text = extract_text_from_docx(job_desc_path)
        else:
            return "Unsupported file format"
        
        # Compute similarity score
        similarity_score = compute_similarity(resume_text, job_desc_text)
        
        return render_template('results.html', resume_text=resume_text, job_desc_text=job_desc_text, similarity_score=similarity_score)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
