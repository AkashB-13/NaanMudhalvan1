import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Page configuration
st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🍿 AI-Based Movie Recommendation System 🎥</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Check if files exist
    if not os.path.exists("tmdb_5000_movies.csv") or not os.path.exists("tmdb_5000_credits.csv"):
        st.error("❌ Movie and credits data not found.")
        st.stop()

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    df = movies.merge(credits, on='title')

    df['cast'] = df['cast'].fillna('[]')
    df['crew'] = df['crew'].fillna('[]')
    df['genres'] = df['genres'].fillna('[]')

    def get_director(crew_json):
        try:
            for member in ast.literal_eval(crew_json):
                if member.get('job') == 'Director':
                    return member.get('name', '')
        except:
            return ''
        return ''

    def get_top_cast(cast_json):
        try:
            return [member.get('name', '') for member in ast.literal_eval(cast_json)[:3]]
        except:
            return []

    def get_genres(genre_json):
        try:
            return ' '.join([genre['name'].lower().replace(" ", "") for genre in ast.literal_eval(genre_json)])
        except:
            return ''

    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(get_top_cast)
    df['genres'] = df['genres'].apply(get_genres)

    df['tags'] = df.apply(lambda row: f"{' '.join(row['cast'])} {row['director']} {row['genres']}", axis=1)
    return df[['title', 'genres', 'cast', 'director', 'tags']]

@st.cache_data
def load_posters():
    try:
        posters_df = pd.read_csv("movies_metadata.csv")
        posters_df = posters_df[['Title', 'Poster']].dropna()
        return posters_df
    except:
        return pd.DataFrame(columns=['Title', 'Poster'])

df = load_data()
posters = load_posters()

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['tags'])
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_movies(query):
    query = query.lower()
    matches = df[
        df['title'].str.lower().str.contains(query) |
        df['genres'].str.lower().str.contains(query) |
        df['director'].str.lower().str.contains(query) |
        df['cast'].apply(lambda x: any(query in member.lower() for member in x) if isinstance(x, list) else False)
    ]

    if matches.empty:
        return pd.DataFrame(), "❌ No matches found."

    idx = matches.index[0]
    scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in scores]
    return df.iloc[movie_indices][['title', 'genres', 'cast', 'director']], "✅ Recommendations for you:"

# Sidebar input
st.sidebar.title("🔍 Search Movies")
user_input = st.sidebar.text_input("Enter title, genre, actor, or director")

# Main display: Show random movie posters
if not posters.empty:
    st.markdown("### 🎞️ Featured Movies")
    sample_posters = posters.sample(3)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(sample_posters.iterrows()):
        with cols[idx]:
            st.image(row['Poster'], caption=row['Title'], use_column_width=True)

# Search and results
if user_input:
    recommendations, msg = recommend_movies(user_input)
    st.subheader(msg)
    for _, row in recommendations.iterrows():
        st.markdown(f"### 🎬 {row['title']}")
        st.markdown(f"**🎭 Genres:** {row['genres']}")
        st.markdown(f"**🎬 Director:** {row['director']}")
        st.markdown(f"**⭐ Top Cast:** {', '.join(row['cast']) if isinstance(row['cast'], list) else row['cast']}")
        # Show poster if available
        matched = posters[posters['Title'].str.lower() == row['title'].lower()]
        if not matched.empty:
            st.image(matched.iloc[0]['Poster'], use_column_width=False, width=300)
        st.markdown("---")
else:
    st.info("🔎 Type a movie, genre, actor, or director to get started.")

