import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")
st.title("🎬 AI-Based Movie Recommendation System")

@st.cache_data
def load_data():
    movies = pd.read_csv("movie.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    links = pd.read_csv("link.csv")
    df = movies.merge(credits, left_on='id', right_on='movie_id', how='left')

    def get_director(x):
        try:
            for i in ast.literal_eval(x):
                if i['job'] == 'Director':
                    return i['name']
        except:
            return ''
        return ''

    def get_top_cast(x):
        try:
            return [i['name'] for i in ast.literal_eval(x)[:3]]
        except:
            return []

    def parse_genres(x):
        try:
            return [i['name'] for i in ast.literal_eval(x)]
        except:
            return []

    df['cast'] = df['cast'].fillna('[]').apply(get_top_cast)
    df['crew'] = df['crew'].fillna('[]').apply(get_director)
    df['genres'] = df['genres'].fillna('[]').apply(parse_genres)
    df['overview'] = df['overview'].fillna('')
    df['tags'] = df.apply(lambda row: ' '.join(row['cast']) + ' ' + row['crew'] + ' ' + ' '.join(row['genres']) + ' ' + row['overview'], axis=1)
    return df

df = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_by_index(index, top_n=10):
    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in scores]
    return df.iloc[movie_indices][['original_title', 'genres', 'crew']]

# Sidebar filters
st.sidebar.header("🎯 Your Preferences")
title_input = st.sidebar.text_input("Enter Movie Name (optional)").strip().lower()
genre_input = st.sidebar.text_input("Enter Genre (optional)").strip().lower()
director_input = st.sidebar.text_input("Enter Director Name (optional)").strip().lower()

filtered_df = df.copy()
if title_input:
    filtered_df = filtered_df[filtered_df['original_title'].str.lower().str.contains(title_input)]
if genre_input:
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: any(genre_input in g.lower() for g in x))]
if director_input:
    filtered_df = filtered_df[filtered_df['crew'].str.lower().str.contains(director_input)]

if not filtered_df.empty:
    st.success(f"✅ Found {len(filtered_df)} match(es). Showing recommendations based on the top result:")
    idx = filtered_df.index[0]
    recs = recommend_by_index(idx)
    st.subheader("🎉 Recommended Movies:")
    for _, row in recs.iterrows():
        st.markdown(f"**🎬 {row['original_title']}**")
        st.write(f"Genres: {', '.join(row['genres'])}")
        st.write(f"Director: {row['crew']}")
        st.markdown("---")
else:
    st.warning("❗ No matches found. Try different keywords.")

st.caption("Built with ❤ using Streamlit")
