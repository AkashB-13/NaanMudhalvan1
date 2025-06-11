import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit config
st.set_page_config(page_title="MovieVerse AI Recommender", layout="wide")
st.title("🎬 AI-Based Movie Recommendation System")

# Load data
@st.cache_data

def load_data():
    movies = pd.read_csv("movie.csv")
    links = pd.read_csv("link.csv")

    # Merge on movieId
    df = pd.merge(movies, links, on="movieId", how="inner")
    df.dropna(subset=['tmdbId'], inplace=True)
    df['tmdbId'] = df['tmdbId'].astype(int)
    
    # Process genres
    df['genres'] = df['genres'].apply(lambda x: x.replace('|', ' '))
    df['tags'] = df['title'] + ' ' + df['genres']

    # Vectorize tags
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['tags'])
    similarity = cosine_similarity(tfidf_matrix)

    return df, vectorizer, similarity

# Fetch posters from TMDB API
def fetch_poster(tmdb_id):
    api_key = "fd09c6f07ac096efb6bf5af91fa69803"
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("poster_path"):
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except:
        pass
    return "https://via.placeholder.com/300x450?text=No+Image"

df, vectorizer, similarity = load_data()

# Sidebar filters
st.sidebar.header("🔍 Search Preferences")
user_title = st.sidebar.text_input("Enter Movie Title (optional)")
user_genre = st.sidebar.text_input("Enter Genre (optional)")
user_director = st.sidebar.text_input("Enter Director (optional)")

# Recommend movies
@st.cache_data

def recommend(title=None, genre=None):
    indices = []

    if title:
        matches = df[df['title'].str.contains(title, case=False, na=False)]
        if not matches.empty:
            idx = matches.index[0]
            sim_scores = list(enumerate(similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
            indices = [i[0] for i in sim_scores]

    elif genre:
        indices = df[df['genres'].str.contains(genre, case=False, na=False)].index.tolist()

    else:
        indices = df.sample(20).index.tolist()

    results = df.iloc[indices].copy()
    results['poster'] = results['tmdbId'].apply(fetch_poster)
    return results

# Show Recommendations
st.markdown("""<style>.block-container{padding-top:2rem;}</style>""", unsafe_allow_html=True)

if user_title or user_genre or user_director:
    st.subheader("✨ Your Personalized Recommendations")
    results = recommend(user_title, user_genre)
    cols = st.columns(4)
    for i, row in results.iterrows():
        with cols[i % 4]:
            st.image(row['poster'], use_column_width=True)
            st.markdown(f"**{row['title']}**")
            st.caption(row['genres'])
else:
    st.info("Enter a movie name, genre, or director in the sidebar to get started!")

# Add background image using markdown CSS
st.markdown(
    """
    <style>
    body {
        background-image: url("https://images.unsplash.com/photo-1542206395-9feb3edaa68c?auto=format&fit=crop&w=1650&q=80");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
