# -----------------------------
# 1️⃣ Import Libraries
# -----------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # For allowing cross-origin requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.stem.porter import PorterStemmer  # For stemming words

# -----------------------------
# 2️⃣ Initialize FastAPI App
# -----------------------------
app = FastAPI()

# 🔹 Enable CORS (Cross-Origin Resource Sharing)
#    This allows API to be accessed from any frontend or tool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# -----------------------------
# 3️⃣ Load & Prepare Dataset
# -----------------------------
try:
    df = pd.read_csv('job_dataset.csv')  # Load the CSV dataset
    df = df.head(2000)                  # Limit to 2000 rows for faster processing
except:
    df = pd.DataFrame(columns=['Title', 'Tags'])  # Empty dataframe if CSV missing

# -----------------------------
# 4️⃣ Text Preprocessing
# -----------------------------
ps = PorterStemmer()  # Initialize PorterStemmer for word stemming

def clean_text(text):
    """
    1. Remove all non-alphabetic characters
    2. Convert text to lowercase
    3. Split text into words
    4. Apply stemming on each word
    5. Join words back into a single string
    """
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower().split()
    y = [ps.stem(word) for word in text]
    return " ".join(y)

# -----------------------------
# 5️⃣ Vectorization & Similarity
# -----------------------------
if not df.empty:
    # Create a new column with cleaned tags
    df['clean_tags'] = df['Tags'].apply(clean_text)
    
    # Convert text to TF-IDF vectors
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(df['clean_tags']).toarray()
    
    # Compute cosine similarity matrix between all job vectors
    similarity = cosine_similarity(vectors)

# -----------------------------
# 6️⃣ API Endpoints
# -----------------------------
@app.get("/")
def home():
    """
    Home endpoint
    """
    return {"message": "Job API is Live & Authenticated!"}

@app.get("/recommend/{job_title}")
def recommend(job_title: str):
    """
    Recommend jobs based on a given job title
    1. Find the index of the job that contains the input job title
    2. Get cosine similarity scores for this job with all other jobs
    3. Sort and pick top 5 similar jobs (excluding itself)
    4. Return job Title, Tags, and Similarity Score
    """
    try:
        # Search job ignoring case
        idx = df[df['Title'].str.contains(job_title, case=False)].index[0]
        
        # Get similarity scores
        distances = similarity[idx]
        
        # Sort by similarity descending & pick top 5
        jobs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
        
        # Prepare response
        results = []
        for i in jobs_list:
            results.append({
                "Title": df.iloc[i[0]].Title,
                "Tags": df.iloc[i[0]].Tags,
                "Score": str(round(i[1], 2))
            })
        return {"status": "success", "data": results}
    
    except:
        return {"status": "error", "message": "Job not found"}
