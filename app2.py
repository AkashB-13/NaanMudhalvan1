import streamlit as st
import sqlite3
import pickle
import pandas as pd
import requests
import scipy.sparse as sparse
from streamlit_extras.bottom_container import bottom
from streamlit_chat_widget import chat_input_widget

# TMDB API Key from secrets
tmdb_api_key = st.secrets["TMDB_KEY"]

# --- Database Setup ---
conn = sqlite3.connect('watchhistory.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS history (user_id INTEGER, movie_id INTEGER, ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')

# --- Helper Functions for Auth and History ---
def signup(username, password):
    return c.execute('INSERT OR IGNORE INTO users(username,password) VALUES(?,?)', (username, password)).rowcount

def login(username, password):
    return c.execute('SELECT id FROM users WHERE username=? AND password=?', (username, password)).fetchone()

def log_history(user_id, movie_id):
    c.execute('INSERT INTO history(user_id, movie_id) VALUES(?,?)', (user_id, movie_id))
    conn.commit()

# --- Load Models and Data ---
@st.cache_data
def load_data():
    movies = pd.read_pickle('movies_with_posters.pkl')
    cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    als_model = pickle.load(open('als_model.pkl', 'rb'))
    user_to_idx = pickle.load(open('user_to_idx.pkl', 'rb'))
    movie_to_idx = pickle.load(open('movie_to_idx.pkl', 'rb'))
    return movies, cosine_sim, tfidf_vectorizer, als_model, user_to_idx, movie_to_idx

movies, cosine_sim, tfidf_vectorizer, als_model, user_to_idx, movie_to_idx = load_data()

# --- Recommendation Functions ---
def recommend_by_title(title, n=10):
    idx = movies[movies['original_title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return pd.DataFrame()
    sim_scores = list(enumerate(cosine_sim[idx[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return movies.iloc[[i for i, _ in sim_scores]]

def recommend_by_user(user_id, n=10):
    if user_id not in user_to_idx:
        return pd.DataFrame()
    recs = als_model.recommend(user_to_idx[user_id], sparse.csr_matrix(cosine_sim), N=n)
    return movies.iloc[[i for i, _ in recs]]

@st.cache_data(ttl=3600)
def load_trending():
    res = requests.get(f'https://api.themoviedb.org/3/trending/movie/day?api_key={tmdb_api_key}')
    return res.json().get('results', [])

def show_movies(df):
    cols = st.columns(4)
    for idx, (_, row) in enumerate(df.iterrows()):
        col = cols[idx % 4]
        col.image(row['poster_url'], use_column_width=True)
        col.subheader(row['original_title'])
        col.write(f"Genres: {', '.join(row['genres']) if isinstance(row['genres'], list) else 'N/A'}")
        col.write(f"Year: {row.get('release_year', 'N/A')} • Rating: {row.get('vote_average', 'N/A')}")
        if st.session_state.get('user'):
            col.button("Mark Watched", key=f"watch_{row.name}", on_click=log_history,
                       args=(st.session_state.user['id'], row['id']))

# --- Page Config and Sidebar ---
st.set_page_config("MovieVerse PRO", layout="wide")

st.sidebar.title("🔐 Account")
if 'user' not in st.session_state:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        res = login(username, password)
        if res:
            st.session_state.user = {'id': res[0], 'username': username}
        else:
            st.sidebar.error("Invalid credentials")
    if st.sidebar.button("Sign up"):
        if signup(username, password):
            st.sidebar.success("Account created. Please login.")
        else:
            st.sidebar.error("Username already exists")
    st.stop()
else:
    st.sidebar.write(f"Welcome, {st.session_state.user['username']}")
    if st.sidebar.button("Logout"):
        del st.session_state.user

st.sidebar.title("🎛 Filters")
genres = sorted({g for sub in movies['genres'] for g in sub})
selected_genres = st.sidebar.multiselect("Genres", genres)
year_range = st.sidebar.slider("Year", 1950, 2025, (2000, 2025))
rating_min = st.sidebar.slider("Min Rating", 0.0, 10.0, 7.0)

st.sidebar.title("🌐 Language")
language = st.sidebar.selectbox("API Language", ['en', 'fr', 'es', 'hi', 'ta'])

# --- Main Area ---
st.header("🔥 Trending Today")
trending = load_trending()[:8]
cols = st.columns(4)
for i, movie in enumerate(trending):
    with cols[i % 4]:
        st.image(f"https://image.tmdb.org/t/p/w200{movie['poster_path']}", use_column_width=True)
        st.caption(f"{movie['title']} ({movie['release_date'][:4]})")

st.header("🎬 Search or Get Recommendations")
mode = st.radio("Mode", ["By Title", "By User ID"], horizontal=True)
query = st.text_input("Enter title or user ID")

results = pd.DataFrame()
if mode == "By Title" and query:
    results = recommend_by_title(query)
elif mode == "By User ID" and query:
    try:
        uid = int(query)
        results = recommend_by_user(uid)
    except:
        st.error("Invalid user ID")

# Filter results
if not results.empty:
    results = results[(results['release_year'].between(*year_range)) &
                      (results['vote_average'] >= rating_min)]
    if selected_genres:
        results = results[results['genres'].apply(lambda gs: any(g in gs for g in selected_genres))]

if results.empty:
    st.info("No results. Try different input or filters.")
else:
    show_movies(results)

# --- Chatbot with voice input ---
with bottom():
    chat = chat_input_widget()
if chat and chat.get("text"):
    st.write(f"**Bot (echo)**: {chat['text']}")
