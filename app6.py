import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Based Movie Recommendation System")

@st.cache_data
def load_data():
    if os.path.getsize("tmdb_5000_credits.csv") == 0 or os.path.getsize("tmdb_5000_movies.csv") == 0:
        st.error("❌ One or more files are empty. Please re-upload.")
        st.stop()

    # Load TMDB data
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    df = movies.merge(credits, on='title')

    # Clean columns
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

    def create_tags(row):
        cast = ' '.join(row['cast']) if isinstance(row['cast'], list) else ''
        return f"{cast} {row['director']} {row['genres']}"

    df['tags'] = df.apply(create_tags, axis=1)

    return df[['title', 'genres', 'cast', 'director', 'tags']]

@st.cache_data
def load_metadata():
    try:
        metadata = pd.read_csv("/mnt/data/8be23759-6eba-4327-805e-39a6b8951bfd.csv", low_memory=False)
        metadata = metadata[metadata['poster_path'].notna()]
        metadata = metadata[['title', 'poster_path']]
        metadata['title_lower'] = metadata['title'].str.lower()
        return metadata
    except:
        return pd.DataFrame(columns=['title', 'poster_path', 'title_lower'])

# Load data
df = load_data()
metadata = load_metadata()

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['tags'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Recommendation engine
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

    recommended = df.iloc[movie_indices][['title', 'genres', 'cast', 'director']]
    recommended['title_lower'] = recommended['title'].str.lower()
    return recommended, "✅ Here are your movie recommendations:"

# Sidebar input
st.sidebar.title("🔍 Search")
user_input = st.sidebar.text_input("Enter movie title, genre, actor or director")

# Background section - random posters
st.sidebar.markdown("## 🎥 Random Movie Posters")
if not metadata.empty:
    sample_posters = metadata.sample(3)
    for _, row in sample_posters.iterrows():
        poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
        st.sidebar.image(poster_url, caption=row['title'], use_column_width=True)

# Results
if user_input:
    recommendations, msg = recommend_movies(user_input)
    st.subheader(msg)
    if recommendations.empty:
        st.warning("Try a different search.")
    else:
        for _, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                poster_row = metadata[metadata['title_lower'] == row['title_lower']]
                if not poster_row.empty:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_row.iloc[0]['poster_path']}"
                    st.image(poster_url, width=120)
                else:
                    st.image("https://via.placeholder.com/120x180?text=No+Image", width=120)
            with col2:
                st.markdown(f"### 🎬 {row['title']}")
                st.markdown(f"**Genres:** {row['genres']}")
                st.markdown(f"**Director:** {row['director']}")
                st.markdown(f"**Top Cast:** {', '.join(row['cast']) if isinstance(row['cast'], list) else row['cast']}")
                st.markdown("---")
else:
    st.info("🔎 Enter a movie, genre, actor, or director to get recommendations.")

