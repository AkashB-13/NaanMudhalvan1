import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎬 Enhanced Movie Recommender", layout="wide")

# Add background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4");
        background-size: cover;
        color: white;
    }}
    .css-1d391kg {{
        background-color: rgba(0, 0, 0, 0.5) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Enhanced AI Movie Recommender")

@st.cache_data
def load_data():
    movie = pd.read_csv("movie.csv")
    link = pd.read_csv("link.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    
    link = link.dropna(subset=['tmdbId'])
    link['tmdbId'] = link['tmdbId'].astype(int)
    credits['movie_id'] = credits['movie_id'].astype(int)
    movie_linked = movie.merge(link, on='movieId')
    df = movie_linked.merge(credits, left_on='tmdbId', right_on='movie_id', how='left')

    df['cast'] = df['cast'].fillna('[]')
    df['crew'] = df['crew'].fillna('[]')
    df['genres'] = df['genres'].fillna('')

    def get_director(x):
        for i in ast.literal_eval(x):
            if i.get('job') == 'Director':
                return i.get('name')
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
    return df[['title', 'genres', 'cast', 'director', 'tags']]

df = load_data()

# WordCloud Display
with st.expander("📊 Genre WordCloud"):
    genre_text = " ".join(df['genres'].dropna())
    wordcloud = WordCloud(background_color="black").generate(genre_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

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
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in scores]

    return df.iloc[movie_indices][['title', 'genres', 'cast', 'director']], "✅ Recommendations:"

st.sidebar.title("🎥 Movie Finder")
user_input = st.sidebar.text_input("🔍 Enter title, genre, actor, or director:")

if user_input:
    results, msg = recommend_movies(user_input)
    st.subheader(msg)
    for _, row in results.iterrows():
        with st.container():
            st.markdown(f"### 🎬 {row['title']}")
            st.markdown(f"**🎭 Genres:** {row['genres']}")
            st.markdown(f"**🎬 Director:** {row['director']}")
            st.markdown(f"**🌟 Cast:** {', '.join(row['cast']) if isinstance(row['cast'], list) else row['cast']}")
            st.markdown("---")
else:
    st.info("Enter something to get movie suggestions!")


