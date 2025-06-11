import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

st.set_page_config(page_title="🎬 MovieVerse: Movie Pedia", layout="wide")
st.title("🎬 MovieVerse: Movie Pedia")

@st.cache_data
def load_main_data():
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
            return [genre['name'] for genre in ast.literal_eval(genre_json)]
        except:
            return []

    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(get_top_cast)
    df['genres'] = df['genres'].apply(get_genres)

    def create_tags(row):
        cast = ' '.join(row['cast']) if isinstance(row['cast'], list) else ''
        genres = ' '.join(row['genres']) if isinstance(row['genres'], list) else ''
        return f"{cast} {row['director']} {genres}"

    df['tags'] = df.apply(create_tags, axis=1)
    return df

@st.cache_data
def load_metadata():
    meta = pd.read_csv("movies_metadata.csv", low_memory=False)
    meta = meta[meta['poster_path'].notna()]
    meta = meta[['title', 'poster_path']].dropna()
    return meta

df = load_main_data()
metadata = load_metadata()

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['tags'])
cosine_sim = cosine_similarity(tfidf_matrix)

def get_recommendations(query):
    query = query.lower()
    matches = df[
        df['title'].str.lower().str.contains(query) |
        df['genres'].apply(lambda genres: any(query in g.lower() for g in genres)) |
        df['director'].str.lower().str.contains(query) |
        df['cast'].apply(lambda cast: any(query in c.lower() for c in cast))
    ]

    if matches.empty:
        return pd.DataFrame(), "No matches found."

    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:9]
    indices = [i[0] for i in scores]
    return df.iloc[indices], "Found some great matches based on your input! 🎯"

st.sidebar.title("🎭 Chatbot")
user_input = st.sidebar.text_input("Ask me anything about movies!", key="chat")

st.sidebar.markdown("---")

st.markdown("#### Search movies by title:")
search_query = st.text_input("Enter a movie name, genre, actor or director")

if user_input or search_query:
    results, msg = get_recommendations(user_input if user_input else search_query)
    st.markdown(f"### {msg}")
    if not results.empty:
        cols = st.columns(4)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 4]:
                st.subheader(row['title'])
                poster_row = metadata[metadata['title'].str.lower() == row['title'].lower()]
                if not poster_row.empty:
                    poster_url = "https://image.tmdb.org/t/p/w500" + poster_row.iloc[0]['poster_path']
                    st.image(poster_url, use_column_width=True)
                genres_str = ", ".join(row['genres'])
                st.markdown(f"**Genres:** {genres_str}")
                st.markdown(f"**Director:** {row['director']}")
                st.markdown(f"**Top Cast:** {', '.join(row['cast'])}")
                st.markdown("---")
else:
    st.info("Enter a keyword or movie to get started.")
