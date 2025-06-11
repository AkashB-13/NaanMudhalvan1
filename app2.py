import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

# Set Streamlit page config
st.set_page_config(page_title="🎬 MovieVerse Recommender", layout="wide")
st.title("🎬 MovieVerse: Your AI‑Driven Movie Recommender")

@st.cache_resource(show_spinner=False)
def load_data():
    df = pd.read_pickle("movies_with_posters.pkl")
    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("als_model.pkl", "rb") as f:
        als_model = pickle.load(f)
    with open("user_to_idx.pkl", "rb") as f:
        user_to_idx = pickle.load(f)
    with open("movie_to_idx.pkl", "rb") as f:
        movie_to_idx = pickle.load(f)
    return df, cosine_sim, vectorizer, als_model, user_to_idx, movie_to_idx

df, cosine_sim, vectorizer, als_model, user_to_idx, movie_to_idx = load_data()

# Genre filter
all_genres = sorted(set(g for sub in df["genres"] for g in sub))
selected_genres = st.sidebar.multiselect("🎭 Filter by Genre", all_genres)

# Search
search_query = st.text_input("🔍 Search for a movie by title")

# Recommend by text
if search_query:
    indices = df[df['original_title'].str.contains(search_query, case=False, na=False)].index.tolist()
    if indices:
        idx = indices[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        movie_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[movie_indices]
        st.subheader(f"🔁 Recommendations based on **{df.iloc[idx]['original_title']}**")
    else:
        recommendations = pd.DataFrame()
        st.warning("No movie found matching that title.")
else:
    recommendations = df.copy()

# Apply genre filter
if selected_genres:
    recommendations = recommendations[recommendations["genres"].apply(lambda x: any(g in x for g in selected_genres))]

# Display function
def show_movies(dataframe):
    for i in range(0, len(dataframe), 4):
        cols = st.columns(4)
        for j, (_, row) in enumerate(dataframe.iloc[i:i+4].iterrows()):
            with cols[j]:
                st.subheader(row['original_title'])
                st.image(row.get("poster_url", "https://via.placeholder.com/500x750?text=No+Image"), use_column_width=True)
                st.write(f"**Genres:** {', '.join(row['genres'])}")
                st.write(row['overview'][:150] + ("..." if len(row['overview']) > 150 else ""))
                st.markdown("---")

if not recommendations.empty:
    show_movies(recommendations.head(12))
else:
    st.write("No movies to display.")
