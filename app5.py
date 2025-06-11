import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 AI-Based Movie Recommendation System")

@st.cache_data
def load_data():
    movies = pd.read_csv("movie.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge datasets on proper columns
    df = movies.merge(credits, left_on='movieId', right_on='movie_id', how='left')

    # Fill missing fields and process
    df['overview'] = df['overview'].fillna('')
    df['cast'] = df['cast'].fillna('[]')
    df['crew'] = df['crew'].fillna('[]')
    df['genres'] = df['genres'].fillna('')

    # Convert cast JSON to top 3 names
    def get_top_cast(x):
        try:
            return [i['name'] for i in ast.literal_eval(x)[:3]]
        except:
            return []

    # Extract director
    def get_director(x):
        try:
            for i in ast.literal_eval(x):
                if i['job'] == 'Director':
                    return i['name']
        except:
            return ''

    df['cast'] = df['cast'].apply(get_top_cast)
    df['director'] = df['crew'].apply(get_director)

    # Create "tags" field
    def create_tags(row):
        return ' '.join(row['cast']) + ' ' + row['director'] + ' ' + row['genres'] + ' ' + row['overview']

    df['tags'] = df.apply(create_tags, axis=1)
    return df

def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return tfidf, cosine_sim

# Load data
with st.spinner("Loading and processing data..."):
    df = load_data()
    tfidf, cosine_sim = build_model(df)

# Sidebar for user input
st.sidebar.header("Search for a Movie 🎥")
title_input = st.sidebar.text_input("Enter Movie Title")
director_input = st.sidebar.text_input("Enter Director Name")
genre_input = st.sidebar.text_input("Enter Genre")

# Function to search movies
def recommend_movies(title=None, director=None, genre=None):
    matched_df = df.copy()
    
    if title:
        matched_df = matched_df[matched_df['title'].str.contains(title, case=False, na=False)]
    if director:
        matched_df = matched_df[matched_df['director'].str.contains(director, case=False, na=False)]
    if genre:
        matched_df = matched_df[matched_df['genres'].str.contains(genre, case=False, na=False)]

    if matched_df.empty:
        return [], "No movies found matching your input. Try different keywords."

    idx = matched_df.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:11]]

    return df.iloc[top_indices], "Here are some movies you might like:"

# Trigger recommendation
if title_input or director_input or genre_input:
    results, msg = recommend_movies(title_input, director_input, genre_input)
    st.subheader(msg)
    
    if len(results):
        cols = st.columns(2)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 2]:
                st.markdown(f"### {row['title']}")
                st.write(f"**Director:** {row['director']}")
                st.write(f"**Cast:** {', '.join(row['cast']) if isinstance(row['cast'], list) else row['cast']}")
                st.write(f"**Genres:** {row['genres']}")
                st.write(row['overview'][:300] + ("..." if len(row['overview']) > 300 else ""))
                st.markdown("---")
else:
    st.info("Please enter a movie title, director, or genre in the sidebar to get recommendations.")
