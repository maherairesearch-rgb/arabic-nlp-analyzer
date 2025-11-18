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
# Arabic Support
# ------------------------------------------
font_path = "Amiri-Regular.ttf"  # تأكد من وجود الخط في المجلد
font_prop = fm.FontProperties(fname=font_path)

def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

# ------------------------------------------
# Text Processing
# ------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

arabic_stopwords = {
    "في","على","من","إلى","عن","هو","هي","كان","يكون","قد","لم","لن","لا","ما","ذلك",
    "هذا","هذه","التي","الذي","وما","ومن","وعن","بين","حتى","كل","مع","أن","إن","أو","بل",
    "و","ثم","لكن","كما","اذا","إذا","هنا","هناك","اي","أي","نحن","أنا","انت","هم","هن"
}

def remove_stopwords(words):
    return [w for w in words if w not in arabic_stopwords]

stemmer = ISRIStemmer()
def stem_words(words):
    return [stemmer.stem(w) for w in words]

# ------------------------------------------
# Page Config & Dark Academic Theme
# ------------------------------------------
st.set_page_config(page_title="منصة تحليل النصوص العربية", layout="wide")

# خلفية داكنة أكاديمية فاخرة مع نصوص بيضاء واضحة
dark_theme = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0b132b, #1c2526);
        color: #e0e1dd;
    }
    [data-testid="stHeader"] {background: transparent;}
    .css-1d391kg, .css-1v0mbdj {padding-top: 2rem;}
    h1, h2, h3, h4 {color: #a8dadc;}
    .stButton>button {
        background: #457b9d; color: white; border-radius: 12px; padding: 0.6rem 2rem;
        font-weight: bold; border: none;
    }
    .stTextArea>div>div>textarea {background: #1c2526; color: #e0e1dd; border: 1px solid #457b9d;}
    .stDataFrame {background: #1c2526;}
</style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

# ------------------------------------------
# Header رسمي وأنيق
# ------------------------------------------
col1, col2 = st.columns([1, 4])

with col1:
    try:
        st.image("logo.png", width=110)
    except:
        pass

with col2:
    st.markdown(
        """
        <div style="
            text-align: right;
            direction: rtl;
            padding: 30px 40px;
            background: linear-gradient(90deg, rgba(27, 38, 59, 0.95), rgba(69, 123, 157, 0.15));
            border-right: 6px solid #a8dadc;
            border-radius: 0 16px 16px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            margin: 20px 0;
        ">
            <h1 style="
                font-family: 'Amiri', serif;
                font-size: 44px;
                color: #f1faee;
                margin: 0;
                font-weight: 900;
                letter-spacing: 1px;
            ">
                منصة التحليل اللغوي الرقمي
            </h1>
            <p style="
                font-size: 24px;
                color: #a8dadc;
                margin: 10px 0 0;
                font-weight: normal;
            ">
                لنصوص التراث العربي والدراسات اللغوية
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ------------------------------------------
# Text Input
# ------------------------------------------
text_input = st.text_area(
    "أدخل النص العربي المراد تحليله:",
    height=240,
    placeholder="مثال: نصوص من كتب التراث، القرآن الكريم، الحديث النبوي، الشعر الجاهلي، النثر الأدبي..."
)

analyze_button = st.button("بدء التحليل", use_container_width=True)

# ------------------------------------------
# Analysis Pipeline
# ------------------------------------------
if analyze_button:
    if not text_input.strip():
        st.error("يرجى إدخال نص عربي أولاً.")
        st.stop()

    with st.spinner("جاري معالجة النص واستخراج الجذور..."):
        cleaned = clean_text(text_input)
        words = cleaned.split()
        no_stop = remove_stopwords(words)
        stems = stem_words(no_stop)

        freq = pd.Series(no_stop).value_counts().reset_index()
        freq.columns = ["الكلمة", "التكرار"]

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### الإحصائيات الأساسية")
            st.write(f"عدد الكلمات الأصلية: **{len(words):,}**")
            st.write(f"عدد الكلمات بعد إزالة كلمات التوقف: **{len(no_stop):,}**")
            st.write(f"عدد الجذور الفريدة المستخرجة: **{len(set(stems)):,}**")

            st.markdown("### أعلى الكلمات تكرارًا")
            st.dataframe(freq.head(20), use_container_width=True)

            st.markdown("### الجذور المستخرجة (Stemming)")
            stem_freq = pd.Series(stems).value_counts().reset_index()
            stem_freq.columns = ["الجذر", "التكرار"]
            st.dataframe(stem_freq.head(20), use_container_width=True)

        with col_b:
            st.markdown("### سحابة الكلمات")
            wc_text = " ".join([fix_arabic(w) for w in no_stop])
            wc = WordCloud(
                font_path=font_path,
                width=800, height=500,
                background_color="#0b132b",
                color_func=lambda *args, **kwargs: "#a8dadc",
                prefer_horizontal=0.9,
                max_words=150
            ).generate(wc_text)

            fig1, ax1 = plt.subplots(figsize=(12, 7))
            ax1.imshow(wc, interpolation="bilinear")
            ax1.axis("off")
            st.pyplot(fig1)

            st.markdown("### توزيع التكرار (أعلى 15 كلمة)")
            top15 = freq.head(15)
            fig2, ax2 = plt.subplots(figsize=(11, 6))
            ax2.bar([fix_arabic(w) for w in top15["الكلمة"]], top15["التكرار"], color="#457b9d")
            ax2.set_xticklabels([fix_arabic(w) for w in top15["الكلمة"]], rotation=45, ha='right', fontproperties=font_prop, fontsize=11)
            ax2.set_facecolor('#0b132b')
            fig2.patch.set_facecolor('#0b132b')
            ax2.tick_params(colors='white')
            ax2.yaxis.label.set_color('white')
            ax2.xaxis.label.set_color('white')
            ax2.spines['bottom'].set_color('gray')
            ax2.spines['top'].set_color('gray')
            ax2.spines['left'].set_color('gray')
            ax2.spines['right'].set_color('gray')
            plt.tight_layout()
            st.pyplot(fig2)

# ------------------------------------------
# Footer رسمي ومحترم
# ------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="
        text-align: center;
        padding: 30px;
        color: #778da9;
        font-size: 18px;
        direction: rtl;
    ">
        تم إعداد هذه المنصة وتطويرها<br>
        <strong style="font-size: 26px; color: #a8dadc;">محمد الجزائري</strong><br>
        باحث في اللسانيات الحاسوبية والتراث العربي الرقمي
    </div>
    """,
    unsafe_allow_html=True
)
