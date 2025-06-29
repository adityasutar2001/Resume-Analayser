import joblib
from flask import Flask, request, render_template
import spacy
import fitz  # For PDFs
from docx import Document
import re

app = Flask(__name__)

# Load NLP and ML models
nlp = spacy.load("en_core_web_sm")
model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Job field requirements
job_requirements = {
    "Data Science": {"skills": ["Python", "Machine Learning", "Statistics", "SQL"]},
    "Software Engineer": {"skills": ["Java", "OOP", "Algorithms", "Databases"]},
    "Full Stack Developer": {"skills": ["HTML", "CSS", "JavaScript", "React", "Node.js"]},
    "Data Analytics": {"skills": ["Excel", "SQL", "Data Visualization", "Power BI"]}
}

# Skill set for detection
skill_list = {"Python", "Machine Learning", "SQL", "Statistics", "Java", "React", "Power BI", "OOP", "CSS", "JavaScript", "HTML", "Databases", "Algorithms", "Node.js", "Data Visualization", "Excel"}

# Extract text from PDF or DOCX
def extract_text_from(file_path):
    if file_path.endswith('.pdf'):
        with fitz.open(file_path) as doc:
            return " ".join([page.get_text() for page in doc])
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return "Unsupported File Type"

# Extract skills using keyword matching
def extract_skills(text):
    matched_skills = []
    text_lower = text.lower()

    for skill in skill_list:
        if skill.lower() in text_lower:
            matched_skills.append(skill)
    return matched_skills

#def extract_skills(text):
 #   matched_skills = []
  #  for skill in skill_list:
   #     pattern = r'\b' + re.escape(skill.lower()) + r'\b'
    #    if re.search(pattern, text.lower()):
     #       matched_skills.append(skill)
    #return matched_skills

# Predict job category using ML
def predict_job_category(text):
    text_vectorized = vectorizer.transform([text])
    return model.predict(text_vectorized)[0]

# Check suitability logic
def check_suitability(selected_job, extracted_skills):
    required_skills = set(job_requirements[selected_job]["skills"])
    matched_skills = set(extracted_skills).intersection(required_skills)
    missing_skills = required_skills - matched_skills
    match_ratio = len(matched_skills) / len(required_skills)

    if match_ratio == 1.0:
        suitability = "✅ Suitable"
        advice = "Your resume fully matches the job requirements!"
    elif match_ratio >= 0.9:
        suitability = "✅ Suitable"
        advice = f"Almost perfect! Just missing: {', '.join(missing_skills)}"
    elif match_ratio > 0:
        suitability = "❌ Unsuitable"
        advice = f"You are missing important skills: {', '.join(missing_skills)}"
    else:
        suitability = "❌ Unsuitable"
        advice = f"No required skills matched. Missing: {', '.join(missing_skills)}"

    return suitability, advice

# Flask Route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, suitability, advice = "", "", ""

    if request.method == 'POST':
        job_field = request.form['job_field']
        file = request.files['resume']

        if not file.filename.endswith(('.pdf', '.docx')):
            return "⚠️ Unsupported file format. Upload PDF or DOCX only."

        file_path = "uploaded_resume.docx"
        file.save(file_path)

        # Extract and process resume
        text = extract_text_from(file_path)
        extracted_skills = extract_skills(text)
        prediction = predict_job_category(text)  # Show ML-based guess
        suitability, advice = check_suitability(job_field, extracted_skills)

    return render_template('index.html', prediction=prediction, suitability=suitability, advice=advice)

if __name__ == '__main__':
    app.run(debug=True)
