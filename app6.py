import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 Cine Match")
st.markdown("### 🎯 Enter movie title, genre, actor or director in the search bar")

st.markdown("### 🎞️ Featured Movie Posters")


row1 = st.columns(4)
with row1[0]:
    st.image("2461.jpg", caption="FREDRICK WARDE", use_container_width=True)
with row1[1]:
    st.image("2544.jpg", caption="RICHARD LOPUS", use_container_width=True)
with row1[2]:
    st.image("2795.jpg", caption="GRIFFTH", use_container_width=True)
with row1[3]:
    st.image("2844.jpg", caption="FANTOMAS", use_container_width=True)


row2 = st.columns(4)
with row2[0]:
    st.image("2985.jpg", caption="THE FAIRY KING", use_container_width=True)
with row2[1]:
    st.image("3014.jpg", caption="A MAN THERE WAS", use_container_width=True)
with row2[2]:
    st.image("3016.jpg", caption="THE INSIDE OF THE WHITE SLAVE", use_container_width=True)
with row2[3]:
    st.image("3037.jpg", caption="FANTAMOS", use_container_width=True)


row3=st.columns(4)
with row2[0]:
    st.image("3419.jpg", caption="THE STUDENT OF PRAGUE", use_container_width=True)
with row2[1]:
    st.image("3471.jpg", caption="TRAFFIC IN SOULS", use_container_width=True)
with row2[2]:
    st.image("3489.jpg", caption="THE LAST DAY OF POMPII", use_container_width=True)
with row2[3]:
    st.image("3643.jpg", caption="THE AVENGING CONSCIENCE", use_container_width=True)

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

# Load datasets
df = load_data()
metadata = load_metadata()

# TF-IDF & Similarity
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

    return df.iloc[movie_indices][['title', 'genres', 'cast', 'director']], "✅ Here are your movie recommendations:"

# Sidebar search
st.sidebar.title("🔍 Search")
user_input = st.sidebar.text_input("Enter movie title, genre, actor or director")

# Sidebar poster display
st.sidebar.markdown("## 🎥 Random Movie Posters")
if not metadata.empty:
    sample_posters = metadata.sample(3)
    for _, row in sample_posters.iterrows():
        poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
        st.sidebar.image(poster_url, caption=row['title'], use_container_width=True)

# Show recommendations
if user_input:
    recommendations, msg = recommend_movies(user_input)
    st.subheader(msg)
    for _, row in recommendations.iterrows():
        st.markdown(f"### 🎬 {row['title']}")
        st.markdown(f"**Genres:** {row['genres']}")
        st.markdown(f"**Director:** {row['director']}")
        st.markdown(f"**Top Cast:** {', '.join(row['cast']) if isinstance(row['cast'], list) else row['cast']}")
        st.markdown("---")

