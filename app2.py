import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Page Setup
st.set_page_config(page_title="🎬 Cine Match - AI Movie Recommender", layout="wide", page_icon="🎥")
st.title("🎬 Cine Match")
st.markdown("#### Your Smart Movie Recommender 🎯")

# Featured Posters
st.markdown("### 🎞️ Featured Classics")
featured_rows = [
    ["2461.jpg", "FREDRICK WARDE"], ["2544.jpg", "RICHARD LOPUS"],
    ["2795.jpg", "GRIFFTH"], ["2844.jpg", "FANTOMAS"],
    ["2985.jpg", "THE FAIRY KING"], ["3014.jpg", "A MAN THERE WAS"],
    ["3016.jpg", "WHITE SLAVE"], ["3037.jpg", "FANTAMOS"],
    ["3419.jpg", "STUDENT OF PRAGUE"], ["3471.jpg", "TRAFFIC IN SOULS"],
    ["3489.jpg", "LAST DAY OF POMPEII"], ["3643.jpg", "AVENGING CONSCIENCE"]
]

for i in range(0, len(featured_rows), 4):
    cols = st.columns(4)
    for j in range(4):
        if i + j < len(featured_rows):
            with cols[j]:
                st.image(featured_rows[i + j][0], caption=featured_rows[i + j][1], use_container_width=True)

# Caching
@st.cache_data(show_spinner=False)
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
            return next((member['name'] for member in ast.literal_eval(crew_json) if member.get('job') == 'Director'), '')
        except:
            return ''

    def get_top_cast(cast_json):
        try:
            return [member['name'] for member in ast.literal_eval(cast_json)[:3]]
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

@st.cache_data(show_spinner=False)
def load_metadata():
    try:
        meta = pd.read_csv("movies_metadata.csv", low_memory=False)
        meta = meta[meta['poster_path'].notna()][['title', 'poster_path']]
        return meta
    except:
        return pd.DataFrame(columns=['title', 'poster_path'])

# Load data
df = load_data()
metadata = load_metadata()

# Build TF-IDF model
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

# Sidebar
st.sidebar.title("🔍 Smart Search")
user_input = st.sidebar.text_input("Enter movie title, genre, actor or director")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎥 Discover Movies")
if not metadata.empty:
    for _, row in metadata.sample(3).iterrows():
        poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
        st.sidebar.image(poster_url, caption=row['title'], use_container_width=True)

# Recommendation Anchor
st.markdown("<a name='recommendations'></a>", unsafe_allow_html=True)

# Main Area: Show Recommendations
if user_input:
    st.experimental_set_query_params(scroll="recommendations")
    recommendations, msg = recommend_movies(user_input)
    st.subheader(msg)

    for _, row in recommendations.iterrows():
        st.markdown(f"### 🎬 {row['title']}")
        st.markdown(f"**Genres:** {row['genres']}")
        st.markdown(f"**Director:** {row['director']}")
        st.markdown(f"**Top Cast:** {', '.join(row['cast']) if isinstance(row['cast'], list) else row['cast']}")
        st.markdown("---")

    # Smooth scroll
    st.markdown("""
        <script>
        const anchor = document.querySelector("a[name='recommendations']");
        if (anchor) {
            anchor.scrollIntoView({behavior: 'smooth'});
        }
        </script>
    """, unsafe_allow_html=True)
