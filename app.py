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
nltk.download('punkt', quiet=True)

# Arabic font
font_path = "Amiri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)

def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

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
st.set_page_config(page_title="Arabic Text Analyzer", layout="wide")

# ------------------------------------------
# HEADER
# ------------------------------------------
st.image("logo.png", width=120)

html_header = """
<div style="
    background: rgba(240,240,240,0.55);
    border-radius: 14px;
    padding: 25px;
    margin-top: 5px;
    margin-bottom: 25px;
    text-align:center;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
">
    <h1 style="
        font-size:32px;
        font-weight:800;
        color:#1d3557;
        margin-bottom:10px;
    ">
        منصة التحليل اللغوي الرقمي لنصوص التراث العربي
    </h1>
</div>
"""
st.markdown(html_header, unsafe_allow_html=True)

# ------------------------------------------
# Text Input
# ------------------------------------------
text_input = st.text_area("أدخل النص العربي هنا:", height=200)
analyze_button = st.button("حلّل النص")

# ------------------------------------------
# Pipeline
# ------------------------------------------
if analyze_button:
    if not text_input.strip():
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
    
    with col1:
        st.subheader("إحصائيات النص")
        st.write(f"عدد الكلمات: **{len(words):,}**")
        st.write(f"بعد إزالة كلمات التوقف: **{len(no_stop):,}**")
        st.write(f"عدد الجذور المستخرجة: **{len(set(stems)):,}** (جذر فريد)")
        
        st.subheader("جدول التكرار")
        st.dataframe(freq.head(15), use_container_width=True)
    
    with col2:
        st.subheader("سحابة الكلمات")
        reshaped = " ".join([fix_arabic(w) for w in no_stop])
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            prefer_horizontal=0.9
        ).generate(reshaped)
        
        fig_wc, ax_wc = plt.subplots(figsize=(10,5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
        
        st.subheader("مخطط التكرار (أعلى 15 كلمة)")
        top15 = freq.head(15)
        fig_bar, ax_bar = plt.subplots(figsize=(12,6))
        bars = ax_bar.bar([fix_arabic(w) for w in top15["word"]], top15["count"], color='#1d3557')
        ax_bar.set_xticklabels([fix_arabic(w) for w in top15["word"]], rotation=45, ha='right', fontproperties=font_prop, fontsize=12)
        ax_bar.set_ylabel("التكرار", fontproperties=font_prop, fontsize=14)
        ax_bar.set_xlabel("الكلمات", fontproperties=font_prop, fontsize=14)
        plt.tight_layout()
        st.pyplot(fig_bar)
    
    st.subheader("الجذور العربية المستخرجة (Stemming)")
    stem_df = pd.DataFrame(stems, columns=["root"])
    stem_freq = stem_df.value_counts().reset_index()
    stem_freq.columns = ["الجذر", "التكرار"]
    st.dataframe(stem_freq.head(20), use_container_width=True)

# ------------------------------------------
# Footer - التذييل النهائي
# ------------------------------------------
st.markdown("---")

footer = """
<div style="
    text-align: center;
    padding: 30px;
    margin-top: 50px;
    background: linear-gradient(135deg, #1d3557, #457b9d);
    border-radius: 16px;
    color: white;
    font-family: 'Amiri', serif;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
">
    <p style="
        font-size: 26px;
        margin: 0;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    ">
        تم إعداده وتطويره بواسطة<br>
        <span style="font-size: 32px; color: #a8dadc;">مـــحمــد الجزائري</span>
    </p>
    <p style="margin-top: 15px; font-size: 18px; opacity: 0.9;">
        باحث في اللسانيات الحاسوبية والتراث العربي الرقمي
    </p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
