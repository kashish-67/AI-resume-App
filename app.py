from flask import Flask, request
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# PDF se text extract karne ka function
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Resume aur Job Description ka match score calculate karne ka function
def match_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])
    return score[0][0] * 100

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("resume")
        jd = request.form.get("jd")

        if not file or not jd:
            return "<h3>Please upload resume and enter Job Description!</h3><a href='/'>Back</a>"

        file.save("resume.pdf")  # Save uploaded file temporarily
        resume_text = extract_text_from_pdf("resume.pdf")
        score = match_score(resume_text, jd)

        return f"<h2>Match Score: {score:.2f}%</h2><br><a href='/'>Back</a>"

    # GET request ke liye form
    return '''
    <h2>AI Resume Screening</h2>
    <form method="post" enctype="multipart/form-data">
        Upload Resume: <input type="file" name="resume" required><br><br>
        Job Description:<br>
        <textarea name="jd" rows="6" cols="40" placeholder="Type job description here..." required></textarea><br><br>
        <input type="submit" value="Submit">
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)