@st.cache_data
def load_data():
    import ast

    # Load datasets
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge on title or appropriate key
    df = movies.merge(credits, on='title')

    # Fill nulls to prevent parsing errors
    df['cast'] = df['cast'].fillna('[]')
    df['crew'] = df['crew'].fillna('[]')
    df['genres'] = df['genres'].fillna('[]')

    # Get director from crew column
    def get_director(crew_json):
        try:
            crew_list = ast.literal_eval(crew_json)
            for member in crew_list:
                if member.get('job') == 'Director':
                    return member.get('name', '')
        except Exception:
            return ''
        return ''

    # Get top 3 cast members
    def get_top_cast(cast_json):
        try:
            cast_list = ast.literal_eval(cast_json)
            return [member.get('name', '') for member in cast_list[:3]]
        except Exception:
            return []

    # Extract genres as string
    def get_genres(genre_json):
        try:
            genre_list = ast.literal_eval(genre_json)
            return ' '.join([genre['name'] for genre in genre_list])
        except Exception:
            return ''

    # Apply parsing functions
    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(get_top_cast)
    df['genres'] = df['genres'].apply(get_genres)

    # Create combined tags column
    def create_tags(row):
        cast = ' '.join(row['cast']) if isinstance(row['cast'], list) else ''
        return f"{cast} {row['director']} {row['genres']}"

    df['tags'] = df.apply(create_tags, axis=1)

    # Return only the needed columns
    return df[['title', 'genres', 'cast', 'director', 'tags']]

