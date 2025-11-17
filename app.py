import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Arabic Plot Support (Font + Reshaping + RTL Fix)
# --------------------------------------------------

import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager as fm

# Load Arabic Font (must exist in same folder)
font_path = "Amiri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)

# Fix Arabic for proper display in charts
def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

# --------------------------------------------------
# Safe Arabic Cleaning Function (No word corruption)
# --------------------------------------------------

def clean_text(text):
    text = text.lower()

    # Remove Arabic diacritics (Harakat)
    diacritics = re.compile(r"[ًٌٍَُِّْـ]")
    text = re.sub(diacritics, "", text)

    # Remove punctuation (Arabic + English)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    # SAFE normalization (does NOT corrupt words)
    text = re.sub("[إأآا]", "ا", text)   # unify alif variations
    text = re.sub("ى", "ي", text)        # unify alef maqsura to ya

    # DO NOT TOUCH:
    # ة — keep ta marbuta
    # ؤ — keep hamza
    # ئ — keep hamza

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --------------------------------------------------
# Arabic Stopwords
# --------------------------------------------------

arabic_stopwords = set([
    "في","على","هذا","هذه","ذلك","تلك","من","ما","ماذا","عن","الى","إلى","هو","هي",
    "ثم","كما","قد","او","أو","بل","و","وهو","وهي","لقد","لكن","كانت","كان","ان","إن",
    "اذا","إذا","بين","حتى","كل","لم","لن","هل","هناك","هنا","مع","أي","اي","انا"
])

def remove_stopwords(words):
    return [w for w in words if w not in arabic_stopwords]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="Arabic Text Analyzer", layout="wide")
st.title("Arabic Text Analyzer – نظام تحليل النص العربي")

st.write("""
Paste any Arabic text below and click **Analyze Text** to begin.  
قم بلصق أي نص عربي في الخانة أدناه ثم اضغط **حلّل النص** للبدء.
""")

text_input = st.text_area("ادخل النص العربي هنا / Enter Arabic text here:", height=200)
analyze_button = st.button("Analyze Text / حلّل النص")

# --------------------------------------------------
# Analysis Pipeline
# --------------------------------------------------

if analyze_button:

    if text_input.strip() == "":
        st.warning("Please enter some text. / الرجاء إدخال نص.")
    else:

        # Step 1 – Clean text
        cleaned = clean_text(text_input)

        # Display cleaned text
        st.subheader("Cleaned Text / النص بعد التنظيف")
        st.write(cleaned)

        # Step 2 – Tokenize
        words = cleaned.split()

        # Step 3 – Remove Stopwords
        words = remove_stopwords(words)

        # Step 4 – Frequency Count
        df = pd.DataFrame(words, columns=["word"])
        freq = df["word"].value_counts().reset_index()
        freq.columns = ["word", "count"]

        st.subheader("Word Frequency Table / جدول تكرار الكلمات")
        st.dataframe(freq)

        # Step 5 – Plot
        st.subheader("Word Frequency Plot / مخطط تكرار الكلمات")

        freq = freq.head(15)
        plt.figure(figsize=(18, 6))

        # Fix Arabic labels
        labels = [fix_arabic(w) for w in freq["word"]]

        plt.bar(labels, freq["count"], color='blue', width=0.9)
        plt.xticks(rotation=60, fontproperties=font_prop, fontsize=18)
        plt.yticks(fontproperties=font_prop, fontsize=16)

        plt.xlabel(fix_arabic("الكلمات"), fontproperties=font_prop, fontsize=22)
        plt.ylabel(fix_arabic("التكرار"), fontproperties=font_prop, fontsize=22)

        st.pyplot(plt)

