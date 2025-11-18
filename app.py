import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager as fm
from nltk.stem.isri import ISRIStemmer
from wordcloud import WordCloud
import nltk

nltk.download('punkt', quiet=True)

# ------------------------------------------
# Arabic Font & Helpers
# ------------------------------------------
font_path = "Amiri-Regular.ttf"  # تأكد أن الملف موجود في نفس المجلد
font_prop = fm.FontProperties(fname=font_path)

def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

# ------------------------------------------
# Text Processing Functions
# ------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

arabic_stopwords = {
    "في","على","هذا","هذه","ذلك","من","ما","ماذا","عن","إلى","الى","هو","هي","ثم","كما",
    "قد","أو","او","بل","و","وهو","وهي","لقد","لكن","كانت","كان","إن","أن","اذا","إذا",
    "بين","حتى","كل","لم","لن","هل","هناك","هنا","مع","أي","اي","أنا","نحن","أنت","هم","هن"
}

def remove_stopwords(words):
    return [w for w in words if w not in arabic_stopwords]

stemmer = ISRIStemmer()
def stem_words(words):
    return [stemmer.stem(w) for w in words]

# ------------------------------------------
# Streamlit Page Config + Beautiful Background
# ------------------------------------------
st.set_page_config(page_title="منصة التحليل اللغوي للتراث العربي", layout="wide")

# خلفية متدرجة أنيقة وهادئة
page_bg = '''
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f1faee 0%, #a8dadc 50%, #457b9d 100%);
        background-attachment: fixed;
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    .css-1d391kg {padding-top: 1rem;}
</style>
'''
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------------------------------
# الـ Header الجديد الفاخر والاحترافي
# ------------------------------------------
col1, col2, col3 = st.columns([1, 2.5, 1])

with col1:
    try:
        st.image("logo.png", width=140)
    except:
        st.markdown("<h1 style='font-size:60px; text-align:center;'>التراث</h1>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 35px 20px;
            background: linear-gradient(90deg, rgba(29, 53, 87, 0.97), rgba(69, 123, 157, 0.97));
            border-radius: 24px;
            box-shadow: 0 15px 45px rgba(0,0,0,0.3);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 2px solid rgba(168, 218, 220, 0.4);
            margin: 20px 0;
        ">
            <h1 style="
                font-family: 'Amiri', serif;
                font-size: 46px;
                font-weight: 900;
                color: #f1faee;
                margin: 0;
                text-shadow: 3px 3px 10px rgba(0,0,0,0.5);
                letter-spacing: 1.5px;
            ">
                منصة التحليل اللغوي الرقمي
            </h1>
            <p style="
                font-size: 26px;
                color: #a8dadc;
                margin: 14px 0 0;
                font-weight: 400;
                letter-spacing: 3px;
            ">
                لنصوص التراث العربي
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 40px;">
            <span style="font-size: 60px; opacity: 0.8;">꧁</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------
# Text Input Area
# ------------------------------------------
text_input = st.text_area(
    "أدخل النص العربي هنا لتحليله:",
    height=220,
    placeholder="الصق أو اكتب أي نص من كتب التراث، القرآن، الحديث، الأدب، الشعر..."
)

analyze_button = st.button("🚀 حلّل النص الآن", use_container_width=True)

# ------------------------------------------
# Analysis Pipeline
# ------------------------------------------
if analyze_button:
    if not text_input.strip():
        st.error("الرجاء إدخال نص عربي للتحليل.")
        st.stop()

    with st.spinner("جاري معالجة النص واستخراج الجذور..."):
        cleaned = clean_text(text_input)
        words = cleaned.split()
        no_stop = remove_stopwords(words)
        stems = stem_words(no_stop)

        df = pd.DataFrame(no_stop, columns=["كلمة"])
        freq = df["كلمة"].value_counts().reset_index()
        freq.columns = ["الكلمة", "التكرار"]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.success("✅ تم التحليل بنجاح!")
            st.markdown(f"**عدد الكلمات الأصلية:** {len(words):,} كلمة")
            st.markdown(f"**بعد إزالة كلمات التوقف:** {len(no_stop):,} كلمة")
            st.markdown(f"**عدد الجذور الفريدة:** {len(set(stems)):,}")

            st.markdown("### 📊 أكثر 20 كلمة تكرارًا")
            st.dataframe(freq.head(20), use_container_width=True)

            st.markdown("### 🌿 الجذور العربية المستخرجة")
            stem_df = pd.DataFrame(stems, columns=["الجذر"])
            stem_freq = stem_df["الجذر"].value_counts().reset_index()
            stem_freq.columns = ["الجذر", "التكرار"]
            st.dataframe(stem_freq.head(20), use_container_width=True)

        with col2:
            st.markdown("### ☁️ سحابة الكلمات")
            text_for_wc = " ".join([fix_arabic(w) for w in no_stop])
            wc = WordCloud(
                font_path=font_path,
                width=800, height=500,
                background_color="white",
                colormap="viridis",
                max_words=200,
                prefer_horizontal=0.9
            ).generate(text_for_wc)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            st.markdown("### 📈 مخطط التكرار (أعلى 15 كلمة)")
            top15 = freq.head(15)
            fig2, ax2 = plt.subplots(figsize=(11, 6))
            ax2.bar([fix_arabic(w) for w in top15["الكلمة"]], top15["التكرار"], color="#1d3557")
            ax2.set_xticklabels([fix_arabic(w) for w in top15["الكلمة"]], rotation=45, ha='right', fontproperties=font_prop)
            ax2.set_ylabel("التكرار", fontproperties=font_prop, fontsize=14)
            ax2.set_xlabel("الكلمات", fontproperties=font_prop, fontsize=14)
            plt.tight_layout()
            st.pyplot(fig2)

# ------------------------------------------
# الـ Footer الأنيق في الأسفل
# ------------------------------------------
st.markdown("---")

footer = """
<div style="
    text-align: center;
    padding: 40px 20px;
    margin-top: 60px;
    background: linear-gradient(135deg, #1d3557, #457b9d);
    border-radius: 20px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    border: 1px solid rgba(168, 218, 220, 0.3);
">
    <p style="
        font-size: 28px;
        margin: 0;
        font-weight: bold;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
    ">
        تم إعداده وتطويره بحب وعناية بواسطة
    </p>
    <h2 style="
        font-size: 42px;
        color: #a8dadc;
        margin: 15px 0;
        letter-spacing: 2px;
    ">
        مـــحمــد الجزائري
    </h2>
    <p style="font-size: 19px; opacity: 0.9; margin-top: 10px;">
        باحث في اللسانيات الحاسوبية والتراث العربي الرقمي
    </p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
