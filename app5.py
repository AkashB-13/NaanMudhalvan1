import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="🎬 Smart Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Background styling
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1600172454520-134b2dcdb3ea");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #1f1f2e;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #f9c74f;'>🍿 AI-Powered Movie Recommendation Engine</h1>
    <p style='text-align: center; color: #f1faee;'>Find your next favorite movie by searching by genre, cast, or director!</p>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    df = movies.merge(credits, on='title')

    # Clean data
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
        return f"{' '.join(row['cast'])} {row['director']} {row['genres']}"

    df['tags'] = df.apply(create_tags, axis=1)
    return df[['title', 'genres', 'cast', 'director', 'tags']]

# Load and process data
df = load_data()

# TF-IDF + Cosine similarity
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
st.sidebar.title("🎯 Search Criteria")
user_input = st.sidebar.text_input("Enter movie name, genre, actor, or director")

# Display results
if user_input:
    recommendations, msg = recommend_movies(user_input)
    st.subheader(msg)
    
    for _, row in recommendations.iterrows():
        st.markdown(f"""
            <div style='background-color:rgba(0,0,0,0.6);padding:1rem;margin-bottom:1rem;border-radius:10px;'>
                <h3 style='color:#f94144;'>🎬 {row['title']}</h3>
                <p><strong>Genres:</strong> {row['genres']}</p>
                <p><strong>Director:</strong> {row['director']}</p>
                <p><strong>Top Cast:</strong> {', '.join(row['cast'])}</p>
            </div>
        """, unsafe_allow_html=True)

    # WordCloud of Tags
    st.subheader("🔍 Visual Insight: Tag WordCloud")
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(" ".join(df['tags']))
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.info("Please enter something in the sidebar to get movie suggestions.")
