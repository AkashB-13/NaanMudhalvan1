import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Based Movie Recommendation System")

@st.cache_data
def load_data():
    # Load datasets
    movie = pd.read_csv("movie.csv")
    link = pd.read_csv("link.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Clean and convert ids
    link = link.dropna(subset=['tmdbId'])
    link['tmdbId'] = link['tmdbId'].astype(int)
    credits['movie_id'] = credits['movie_id'].astype(int)

    # Merge movies with links
    movie_linked = movie.merge(link, on='movieId')

    # Merge with credits on tmdbId and movie_id
    df = movie_linked.merge(credits, left_on='tmdbId', right_on='movie_id', how='left')

    # Fill missing and parse JSON
    df['cast'] = df['cast'].fillna('[]')
    df['crew'] = df['crew'].fillna('[]')
    df['genres'] = df['genres'].fillna('')

    def get_director(x):
        for i in ast.literal_eval(x):
            if i['job'] == 'Director':
                return i['name']
        return ''

    def get_top_cast(x):
        return [i['name'] for i in ast.literal_eval(x)[:3]]

    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(get_top_cast)
    df['genres'] = df['genres'].apply(lambda x: x.replace('|', ' ').lower())

    def create_tags(row):
        cast = ' '.join(row['cast']) if isinstance(row['cast'], list) else ''
        genres = row['genres']
        return f"{cast} {row['director']} {genres}"

    df['tags'] = df.apply(create_tags, axis=1)
    return df

df = load_data()

# TF-IDF Vectorization
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
        return pd.DataFrame(), "No matches found for your input."

    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in scores]

    return df.iloc[movie_indices][['title', 'genres', 'cast', 'director']], "Here are some recommended movies for you:"

st.sidebar.title("🔍 Movie Finder")
user_input = st.sidebar.text_input("Enter genre, movie name, director, or actor")

if user_input:
    recommendations, msg = recommend_movies(user_input)
    st.subheader(msg)
    for _, row in recommendations.iterrows():
        st.markdown(f"### 🎬 {row['title']}")
        st.markdown(f"**Genres:** {row['genres']}")
        st.markdown(f"**Director:** {row['director']}")
        st.markdown(f"**Top Cast:** {', '.join(row['cast']) if isinstance(row['cast'], list) else row['cast']}")
        st.markdown("---")
else:
    st.info("Please enter a movie, genre, director, or actor in the sidebar to get recommendations.")
