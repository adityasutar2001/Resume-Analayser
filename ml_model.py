import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample dataset (Replace this with a real dataset)
data = [
    ("I have experience in Python, Machine Learning, and SQL", "Data Science"),
    ("I develop web apps using React, Node.js, and JavaScript", "Full Stack Developer"),
    ("I work with Excel, SQL, and Power BI for analytics", "Data Analytics"),
    ("I design algorithms and work with Java and databases", "Software Engineer"),
]

df = pd.DataFrame(data, columns=["resume_text", "job_category"])

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume_text"])
y = df["job_category"]

# Train ML model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")

