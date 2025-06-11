import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Based Movie Recommendation System")
st.title("🎬 Cine Match")

# 🔽 Display 4 small images in a row at the top
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image("2461.jpg", caption="2461", use_container_width=True)
with col2:
    st.image("2544.jpg", caption="2544", use_container_width=True)
with col3:
    st.image("2795.jpg", caption="2795", use_container_width=True)
with col4:
    st.image("2844.jpg", caption="2844", use_container_width=True)
with col5:
    st.image("2985.jpg", caption="2844", use_container_width=True)
with col6:
    st.image("3014.jpg", caption="2844", use_container_width=True)
with col7:
    st.image("3016.jpg", caption="2844", use_container_width=True)
with col8:
    st.image("3037.jpg", caption="2844", use_container_width=True)
with col9:
    st.image("3165.jpg", caption="2844", use_container_width=True)

@st.cache_data
def load_data():
    if os.path.getsize("tmdb_5000_credits.csv") == 0 or os.path.getsize("tmdb_5000_movies.csv") == 0:
        st.error("❌ One or more files are empty. Please re-upload.")
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

    def create_tags(row):
        cast = ' '.join(row['cast']) if isinstance(row['cast'], list) else ''
        return f"{cast} {row['director']} {row['genres']}"

    df['tags'] = df.apply(create_tags, axis=1)
    return df[['title', 'genres', 'cast', 'director', 'tags']]

@st.cache_data
def load_metadata():
    try:
        meta = pd.read_csv("movies_metadata.csv", low_memory=False)
        meta = meta[meta['poster_path'].notna()]
        meta = meta[['title', 'poster_path']]
        return meta
    except:
        return pd.DataFrame(columns=['title', 'poster_path'])

df = load_data()
metadata = load_metadata()

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

    return df.iloc[movie_indices][['title', 'genres', 'cast', 'director']], "✅ Here are your movie recommendations:"

st.sidebar.title("🔍 Search")
user_input = st.sidebar.text_input("Enter movie title, genre, actor or director")

st.sidebar.markdown("## 🎥 Random Movie Posters")
if not metadata.empty:
    sample_posters = metadata.sample(3)
    for _, row in sample_posters.iterrows():
        poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
        st.sidebar.image(poster_url, caption=row['title'], use_container_width=True)

if user_input:
    recommendations, msg = recommend_movies(user_input)
    st.subheader(msg)
    for _, row in recommendations.iterrows():
        st.markdown(f"### 🎬 {row['title']}")
        st.markdown(f"**Genres:** {row['genres']}")
        st.markdown(f"**Director:** {row['director']}")
        st.markdown(f"**Top Cast:** {', '.join(row['cast']) if isinstance(row['cast'], list) else row['cast']}")
        st.markdown("---")

