# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")
st.title("🎬 Smart Movie Recommender")

@st.cache_data
def load_data():
    df = pd.read_csv("movie.csv")
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    return df, vectorizer, cosine_sim

df, vectorizer, cosine_sim = load_data()

user_input = st.text_input("🎯 Enter any keyword (e.g., genre, movie title, director):")

if user_input:
    from sklearn.feature_extraction.text import TfidfVectorizer

    user_vec = vectorizer.transform([user_input])
    scores = cosine_similarity(user_vec, cosine_sim).flatten()
    top_indices = scores.argsort()[-10:][::-1]
    top_movies = df.iloc[top_indices][['original_title', 'genres', 'overview']]

    st.subheader("🔮 Recommended Movies:")
    for _, row in top_movies.iterrows():
        st.markdown(f"**🎬 {row['original_title']}**")
        st.write(f"**Genres**: {row['genres']}")
        st.write(f"**Overview**: {row['overview'][:300]}...")
        st.markdown("---")
