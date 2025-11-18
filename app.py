import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------
# Arabic NLP Support
# ------------------------------------------
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager as fm
from nltk.stem.isri import ISRIStemmer
from wordcloud import WordCloud
import nltk
nltk.download('punkt')

# Arabic font
font_path = "Amiri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)

def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

# ------------------------------------------
# UI Styling (Professional & Clean)
# ------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4f7fb 0%, #e8eef5 100%);
        font-family: "Segoe UI", sans-serif;
    }
    h1 {
        text-align: center;
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #1a3d5c !important;
        letter-spacing: 1px;
        padding-bottom: 5px;
    }
    h2, h3 {
        color: #27496d !important;
        font-weight: 600 !important;
    }
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 18px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1558b0;
        transform: scale(1.03);
    }
    textarea {
        border-radius: 12px !important;
        border: 1px solid #c4c7ce !important;
        padding: 10px !important;
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Text Cleaning
# ------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------------------
# Stopwords
# ------------------------------------------
arabic_stopwords = set([
    "في","على","هذا","هذه","ذلك","من","ما","ماذا","عن","إلى","الى","هو","هي",
    "ثم","كما","قد","أو","او","بل","و","وهو","وهي","لقد","لكن","كانت","كان","إن","أن",
    "اذا","إذا","بين","حتى","كل","لم","لن","هل","هناك","هنا","مع","أي","اي","أنا"
])

def remove_stopwords(words):
    return [w for w in words if w not in arabic_stopwords]

# ------------------------------------------
# Stemming
# ------------------------------------------
stemmer = ISRIStemmer()

def stem_words(words):
    return [stemmer.stem(w) for w in words]

# ------------------------------------------
# Streamlit UI
# ------------------------------------------
st.set_page_config(page_title="Arabic Text Analysis Suite", layout="wide")

st.title("Arabic Text Analysis Suite – منصة التحليل اللغوي العربي")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("قم بإدخال نص عربي ثم اضغط **حلّل النص** لعرض النتائج.")
st.markdown('</div>', unsafe_allow_html=True)

text_input = st.text_area("أدخل النص العربي هنا:", height=200)
analyze_button = st.button("حلّل النص")

# ------------------------------------------
# Pipeline
# ------------------------------------------
if analyze_button:

    if text_input.strip() == "":
        st.warning("الرجاء إدخال نص.")
        st.stop()

    cleaned = clean_text(text_input)
    words = cleaned.split()
    no_stop = remove_stopwords(words)
    stems = stem_words(no_stop)

    df = pd.DataFrame(no_stop, columns=["word"])
    freq = df["word"].value_counts().reset_index()
    freq.columns = ["word", "count"]

    col1, col2 = st.columns(2)

    # ------------------------------------------
    # Col 1 – Stats + Table
    # ------------------------------------------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("إحصائيات النص")
        st.write(f"عدد الكلمات: {len(words)}")
        st.write(f"بعد إزالة كلمات التوقف: {len(no_stop)}")
        st.write(f"عدد الجذور المستخرجة: {len(stems)}")

        st.subheader("جدول التكرار")
        st.dataframe(freq.head(15))
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # Col 2 – WordCloud + Bar Chart
    # ------------------------------------------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("سحابة الكلمات")

        reshaped = " ".join([fix_arabic(w) for w in no_stop])

        wc = WordCloud(
            font_path=font_path,
            width=700,
            height=350,
            background_color="white"
        ).generate(reshaped)

        fig_wc, ax_wc = plt.subplots(figsize=(10,4))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        st.subheader("مخطط التكرار (Top 15)")

        plt.figure(figsize=(16,5))
        labels = [fix_arabic(w) for w in freq.head(15)["word"]]
        plt.bar(labels, freq.head(15)["count"], color='blue')
        plt.xticks(rotation=60, fontproperties=font_prop, fontsize=14)
        plt.yticks(fontproperties=font_prop, fontsize=14)
        plt.xlabel(fix_arabic("الكلمات"), fontproperties=font_prop, fontsize=18)
        plt.ylabel(fix_arabic("التكرار"), fontproperties=font_prop, fontsize=18)
        st.pyplot(plt)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # Roots Table
    # ------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("الجذور العربية (Stemming)")
    stem_df = pd.DataFrame(stems, columns=["root"])
    st.dataframe(
        stem_df.value_counts()
        .reset_index()
        .rename(columns={0:"count"})
    )
    st.markdown('</div>', unsafe_allow_html=True)
