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
# Streamlit Page Config
# ------------------------------------------
st.set_page_config(page_title="Digital Heritage Linguistic Analyzer", layout="wide")

# ------------------------------------------
# University Logo (logo.png)
# ------------------------------------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.image("logo.png", width=140)   # ✔️ Make sure the file exists in GitHub
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# Title + Developer Signature (fully corrected)
# ------------------------------------------
st.markdown("""
    <div style="
        background: rgba(240,240,240,0.55);
        border-radius: 14px;
        padding: 25px;
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

        <p style="
            font-size:18px;
            color:#444;
            margin-top:10px;
        ">
            تم تطويره بواسطة <strong>محمد الجزائري</strong>
        </p>
    </div>
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
# Main UI
# ------------------------------------------
st.write("أدخل نصًا عربيًا ثم اضغط **حلّل النص** لعرض النتائج.")

text_input = st.text_area("أدخل النص العربي هنا:", height=200)
analyze_button = st.button("حلّل النص")

# ------------------------------------------
# Pipeline
# ------------------------------------------
if analyze_button:

    if text_input.strip() == "":
        st.warning("الرجاء إدخال نص.")
        st.stop()

    # Cleaning
    cleaned = clean_text(text_input)

    # Tokenizing
    words = cleaned.split()

    # Remove Stopwords
    no_stop = remove_stopwords(words)

    # Stemming
    stems = stem_words(no_stop)

    # Frequencies
    df = pd.DataFrame(no_stop, columns=["word"])
    freq = df["word"].value_counts().reset_index()
    freq.columns = ["word", "count"]

    col1, col2 = st.columns(2)

    # ------------------------------------------
    # Column 1 (Statistics + Table)
    # ------------------------------------------
    with col1:
        st.subheader("إحصائيات النص")
        st.write(f"عدد الكلمات: {len(words)}")
        st.write(f"بعد إزالة كلمات التوقف: {len(no_stop)}")
        st.write(f"عدد الجذور المستخرجة: {len(stems)}")

        st.subheader("جدول التكرار")
        st.dataframe(freq.head(15))

    # ------------------------------------------
    # Column 2 (WordCloud + Plot)
    # ------------------------------------------
    with col2:
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

    # ------------------------------------------
    # Roots Table
    # ------------------------------------------
    st.subheader("الجذور العربية (Stemming)")
    stem_df = pd.DataFrame(stems, columns=["root"])
    st.dataframe(
        stem_df.value_counts()
        .reset_index()
        .rename(columns={0:"count"})
    )

