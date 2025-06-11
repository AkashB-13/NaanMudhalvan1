import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Based Movie Recommendation System")

@st.cache_data
def load_data():
    # Load datasets
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge on 'title'
    df = movies.merge(credits, on='title')

    # Fill nulls to avoid issues
    df['cast'] = df['cast'].fillna('[]')
    df['crew'] = df['crew'].fillna('[]')
    df['genres'] = df['genres'].fillna('[]')

    # Extract director
    def get_director(crew_json):
        try:
            crew_list = ast.literal_eval(crew_json)
            for member in crew_list:
                if member.get('job') == 'Director':
                    return member.get('name', '')
        except:
            return ''
        return ''

    # Extract top 3 cast
    def get_top_cast(cast_json):
        try:
            cast_list = ast.literal_eval(cast_json)
            return [member.get('name', '') for member in cast_list[:3]]
        except:
            return []

    # Extract genres
    def get_genres(genre_json):
        try:
            genre_list = ast.literal_eval(genre_json)
            return ' '.join([genre['name'].lower().replace(" ", "") for genre in genre_list])
        except:
            return ''

    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(get_top_cast)
    df['genres'] = df['genres'].apply(get_genres)

    # Create a single text feature for each movie
    def create_tags(row):
        cast = ' '.join(row['cast']) if isinstance(row['cast'], list) else ''
        return f"{cast} {row['director']} {row['genres']}"

    df['tags'] = df.apply(create_tags, axis=1)

    return df[['title', 'genres', 'cast', 'director', 'tags']]

# Load and preprocess data
df = load_data()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['tags'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Recommendation function
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

# Sidebar input
st.sidebar.title("🔍 Movie Finder")
user_input = st.sidebar.text_input("Enter genre, movie name, director, or actor")

# Display recommendations
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

