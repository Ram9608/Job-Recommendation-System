# AI Job Recommendation System 

## Objective
This project is an AI-powered Job Recommendation System developed as part of the AI/ML Intern Assignment. It matches a candidate's profile (Skills, Domain, Experience) with relevant job descriptions using Natural Language Processing (NLP) and Machine Learning.

##  Features
- **Content-Based Filtering:** Uses TF-IDF and Cosine Similarity to find the best matches.
- **NLP Preprocessing:** Implements Stemming, Stop-word removal, and cleaning.
- **API Support:** Built with FastAPI to provide recommendation endpoints.
- **Custom Dataset:** Generated a synthetic dataset of jobs for testing logic.

##  Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn, NLTK, FastAPI, Uvicorn
- **Tool:** Google Colab

##  Sample Output
Check the file `output_sample.png` in this repository to see the API working in real-time.

##  How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the API: `uvicorn main:app --reload`
