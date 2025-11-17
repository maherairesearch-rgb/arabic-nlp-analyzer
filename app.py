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
font_path = "Amiri-Regular.ttf"   # Make sure this file exists in the project folder
font_prop = fm.FontProperties(fname=font_path)

# Fix Arabic for proper display
def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

# --------------------------------------------------
# Step 2 – Cleaning Function
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)          # remove punctuation + symbols
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces
    return text

# --------------------------------------------------
# Step 5 – Arabic Stopwords
# --------------------------------------------------

arabic_stopwords = set([
    "في","على","هذا","هذه","ذلك","تلك","من","ما","ماذا","عن","إلى","الى","هو","هي",
    "ثم","كما","قد","أو","او","بل","و","وهو","وهي","لقد","لكن","كانت","كان","إن","أن",
    "اذا","إذا","بين","حتى","كل","لم","لن","هل","هناك","هنا","مع","أي","اي","أنا"
])

def remove_stopwords(words):
    return [w for w in words if w not in arabic_stopwords]

# --------------------------------------------------
# Step 1 – Streamlit UI
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
# Steps 3–7 – Analysis Pipeline
# --------------------------------------------------

if analyze_button:

    if text_input.strip() == "":
        st.warning("Please enter some text. / الرجاء إدخال نص.")
    else:
        # Step 3 – Cleaning
        cleaned = clean_text(text_input)

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

        # Show only top 15 most frequent words
        freq = freq.head(15)

        plt.figure(figsize=(18, 6))

        # Arabic-fixed labels
        labels = [fix_arabic(w) for w in freq["word"]]

        # Wider bars + larger font
        plt.bar(labels, freq["count"], color='blue', width=0.9)

        plt.xticks(rotation=60, fontproperties=font_prop, fontsize=18)
        plt.yticks(fontproperties=font_prop, fontsize=16)

        # Enlarged axis labels
        plt.xlabel(fix_arabic("الكلمات"), fontproperties=font_prop, fontsize=22)
        plt.ylabel(fix_arabic("التكرار"), fontproperties=font_prop, fontsize=22)

        st.pyplot(plt)
