from flask import Flask, request, jsonify # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from flask import Flask # type: ignore
from flask_cors import CORS # type: ignore
 

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
 

# Load merged_data or create a sample dataframe for demonstration
merged_data = pd.read_csv('Merged_data.csv')
df = pd.DataFrame(merged_data)

# Apply stemming to the combined text of certificate_name, skill_name, and tech_stack
def stem_text(text):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text.split(',')]
    return ' '.join(stemmed_words)

df['combined_text'] = df['certificate_name'] + ',' + df['skill_table'] + ',' + df['tech_stack']
df['stemmed_text'] = df['combined_text'].apply(stem_text)

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get required tech stack and proficiency from request
    data = request.json
    required_tech_stack = data.get("tech_stack", "")
    required_proficiency = data.get("proficiency", {})

    # Convert required_proficiency to a string
    proficiency_str = ','.join(required_proficiency)
    # Create feature vector for required tech stack
    required_tech_stack=required_tech_stack+ ',' +proficiency_str
    required_tech_stack_vector = stem_text(required_tech_stack)

    # Create feature vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['stemmed_text'])

    # Calculate cosine similarity with the required tech stack vector
    similarities = cosine_similarity(X, vectorizer.transform([required_tech_stack_vector]))

    # Add similarity scores to DataFrame
    df['similarity_score'] = similarities.flatten()

    # Sort employees based on similarity score and select top 10
    top_10_employees = df.nlargest(10, 'similarity_score')

    # Convert DataFrame to JSON and return
    top_recommendations = top_10_employees[['EMP_ID', 'EMPName', 'similarity_score']].to_dict(orient='records')
    return jsonify(top_recommendations)

if __name__ == "__main__":
    app.run(debug=True)
