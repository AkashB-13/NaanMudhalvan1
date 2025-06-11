
import streamlit as st
import pandas as pd

# App config
st.set_page_config(page_title="🎬 Movie Genre Explorer", layout="wide")

# Custom background and style
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: white;
        }
        .movie-card {
            background-color: #222;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 10px rgba(255, 255, 255, 0.1);
        }
        .title {
            font-size: 24px;
            color: #FFD700;
        }
        .genre {
            font-size: 18px;
            color: #ADFF2F;
        }
        .poster {
            max-height: 300px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🍿 Movie Genre Visual Explorer 🎞️</h1>", unsafe_allow_html=True)

# Upload the MovieGenre dataset
file = st.file_uploader("📤 Upload MovieGenre.csv", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
        df = df.dropna(subset=['Title', 'Genre'])

        st.markdown("## 🔍 Browse Movies by Genre")

        # Option to filter by genre
        all_genres = sorted(set(g.strip() for sublist in df['Genre'].dropna().str.split('|') for g in sublist))
        selected_genre = st.selectbox("🎯 Select Genre to Filter", options=['All'] + all_genres)

        if selected_genre != "All":
            df = df[df['Genre'].str.contains(selected_genre, case=False, na=False)]

        # Show movie cards
        for _, row in df.head(15).iterrows():
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='title'>🎬 {row['Title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='genre'>📽️ {row['Genre']}</div>", unsafe_allow_html=True)
            if 'Poster' in row and pd.notna(row['Poster']):
                st.image(row['Poster'], width=200)
            st.markdown("</div>", unsafe_allow_html=True)

        st.success("✅ Displayed successfully!")

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("⬆️ Please upload the `MovieGenre.csv` file to begin.")
