import streamlit as st, sqlite3, pickle, pandas as pd, ast, requests
import scipy.sparse as sparse, implicit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.bottom_container import bottom
from streamlit_chat_widget import chat_input_widget

TMDB_KEY = st.secrets['TMDB_KEY']

# --- Login & DB ---
:contentReference[oaicite:17]{index=17}
:contentReference[oaicite:18]{index=18}
:contentReference[oaicite:19]{index=19}
              :contentReference[oaicite:20]{index=20}
              :contentReference[oaicite:21]{index=21}
:contentReference[oaicite:22]{index=22}
              :contentReference[oaicite:23]{index=23}

:contentReference[oaicite:24]{index=24}
:contentReference[oaicite:25]{index=25}
:contentReference[oaicite:26]{index=26}

# --- Load Pickles ---
@st.cache_data
def load_data():
    :contentReference[oaicite:27]{index=27}
    :contentReference[oaicite:28]{index=28}
    :contentReference[oaicite:29]{index=29}
    :contentReference[oaicite:30]{index=30}
    :contentReference[oaicite:31]{index=31}
    :contentReference[oaicite:32]{index=32}
    :contentReference[oaicite:33]{index=33}
:contentReference[oaicite:34]{index=34}

# --- Utility functions ---
:contentReference[oaicite:35]{index=35}
:contentReference[oaicite:36]{index=36}
    :contentReference[oaicite:37]{index=37}
    :contentReference[oaicite:38]{index=38}
    :contentReference[oaicite:39]{index=39}
    :contentReference[oaicite:40]{index=40}
    :contentReference[oaicite:41]{index=41}
:contentReference[oaicite:42]{index=42}
    :contentReference[oaicite:43]{index=43}
    :contentReference[oaicite:44]{index=44}
    :contentReference[oaicite:45]{index=45}

:contentReference[oaicite:46]{index=46}
def load_trending():
    :contentReference[oaicite:47]{index=47}
    :contentReference[oaicite:48]{index=48}

:contentReference[oaicite:49]{index=49}
    :contentReference[oaicite:50]{index=50}
    for idx, (_:contentReference[oaicite:51]{index=51}
        :contentReference[oaicite:52]{index=52}
        :contentReference[oaicite:53]{index=53}
        :contentReference[oaicite:54]{index=54}
        :contentReference[oaicite:55]{index=55}
        :contentReference[oaicite:56]{index=56}
        :contentReference[oaicite:57]{index=57}
            :contentReference[oaicite:58]{index=58}_:contentReference[oaicite:59]{index=59}
                      :contentReference[oaicite:60]{index=60}

# --- Layout & Sidebar ---
:contentReference[oaicite:61]{index=61}
:contentReference[oaicite:62]{index=62}
:contentReference[oaicite:63]{index=63}
    :contentReference[oaicite:64]{index=64}
    :contentReference[oaicite:65]{index=65}
    :contentReference[oaicite:66]{index=66}
        :contentReference[oaicite:67]{index=67}
        if res:
            :contentReference[oaicite:68]{index=68}
        else:
            :contentReference[oaicite:69]{index=69}
    :contentReference[oaicite:70]{index=70}
        :contentReference[oaicite:71]{index=71}
            :contentReference[oaicite:72]{index=72}
        else:
            :contentReference[oaicite:73]{index=73}
    st.stop()
else:
    :contentReference[oaicite:74]{index=74}
    :contentReference[oaicite:75]{index=75}
        :contentReference[oaicite:76]{index=76}

:contentReference[oaicite:77]{index=77}
:contentReference[oaicite:78]{index=78}
:contentReference[oaicite:79]{index=79}
:contentReference[oaicite:80]{index=80}
:contentReference[oaicite:81]{index=81}

:contentReference[oaicite:82]{index=82}
:contentReference[oaicite:83]{index=83}

# --- Main UI ---
:contentReference[oaicite:84]{index=84}
:contentReference[oaicite:85]{index=85}
:contentReference[oaicite:86]{index=86}
:contentReference[oaicite:87]{index=87}
:contentReference[oaicite:88]{index=88}
    :contentReference[oaicite:89]{index=89}
    :contentReference[oaicite:90]{index=90}
    :contentReference[oaicite:91]{index=91}

:contentReference[oaicite:92]{index=92}
:contentReference[oaicite:93]{index=93}
:contentReference[oaicite:94]{index=94}

:contentReference[oaicite:95]{index=95}
:contentReference[oaicite:96]{index=96}
    :contentReference[oaicite:97]{index=97}
else:
    try:
        :contentReference[oaicite:98]{index=98}
        :contentReference[oaicite:99]{index=99}
    except:
        pass

# apply filters
mask = (
    :contentReference[oaicite:100]{index=100}
    :contentReference[oaicite:101]{index=101}
    :contentReference[oaicite:102]{index=102}
)
:contentReference[oaicite:103]{index=103}

:contentReference[oaicite:104]{index=104}
    :contentReference[oaicite:105]{index=105}
else:
    show_movies(result_df)

# --- Chatbot with Voice ---
with bottom():
    chat = chat_input_widget()
if chat:
    :contentReference[oaicite:106]{index=106}
    if txt:
        :contentReference[oaicite:107]{index=107}

