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

# Load Arabic Font
font_path = "Amiri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)

# Fix Arabic for proper display
def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

# --------------------------------------------------
# Strong Arabic Cleaning Function
# --------------------------------------------------

def clean_text(text):
    text = text.lower()

    # Remove Arabic diacritics
    diacritics = re.compile("""
        ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ
    """, re.VERBOSE)
    text = re.sub(diacritics, "", text)

    # Normalize different forms of Arabic letters
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ؤ', 'ء', text)
    text = re.sub('ئ', 'ء', text)
    text = re.sub('ة', 'ه', text)

    # Remove all punctuation (Arabic + English)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

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

        # Step 3 – Cleaning
        cleaned = clean_text(text_input)

        # Display cleaned text
        st.subheader("Cleaned Text / النص بعد التنظيف")
        st.write(cleaned)

        # Step 4 – Tokenizing
        words = cleaned.split()

        # Step 5 – Remove Stopwords
        words = remove_stopwords(words)

        # Step 6 – Frequency Count
        df = pd.DataFrame(words, columns=["word"])
        freq = df["word"].value_counts().reset_index()
        freq.columns = ["word", "count"]

        st.subheader("Word Frequency Table / جدول تكرار الكلمات")
        st.dataframe(freq)

        # Step 7 – Plot
        st.subheader("Word Frequency Plot / مخطط تكرار الكلمات")

        freq = freq.head(15)
        plt.figure(figsize=(18, 6))

        # Arabic-fixed labels for the chart
        labels = [fix_arabic(w) for w in freq["word"]]

        plt.bar(labels, freq["count"], color='blue', width=0.9)

        plt.xticks(rotation=60, fontproperties=font_prop, fontsize=18)
        plt.yticks(fontproperties=font_prop, fontsize=16)

        plt.xlabel(fix_arabic("الكلمات"), fontproperties=font_prop, fontsize=22)
        plt.ylabel(fix_arabic("التكرار"), fontproperties=font_prop, fontsize=22)

        st.pyplot(plt)
