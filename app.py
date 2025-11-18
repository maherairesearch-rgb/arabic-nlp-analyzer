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
font_path = "Amiri-Regular.ttf"  # تأكد من وجود الخط في نفس المجلد
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
    "و","ثم","لكن","كما","اذا","إذا","هنا","هناك","اي","أي","نحن","أنا","انت","هم","هن",
    "الى","عن","فى","عن","على","من","إلى","في","عن","على","من","الذي","التي","الذين","اللاتي",
    "ومن","وعلى","وفي","بين","بين","حتى","مع","أن","إن","لكن","بل","ثم","كما","ايضا","أيضا"
}

def remove_stopwords(words):
    return [w for w in words if w not in arabic_stopwords]

stemmer = ISRIStemmer()
def stem_words(words):
    return [stemmer.stem(w) for w in words]

# ------------------------------------------
# Page Config & Dark Academic Theme (فخم ومركزي)
# ------------------------------------------
st.set_page_config(page_title="منصة التحليل اللغوي الرقمي", layout="wide")

# ثيم داكن فاخر جدًا مع نصوص فاتحة
dark_theme = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e17 0%, #0b1317 50%, #0a0f1a 100%);
        color: #e8ecef;
    }
    [data-testid="stHeader"] {background: transparent;}
    .css-1d391kg, .css-1v0mbdj {padding-top: 1rem;}
    h1, h2, h3, h4 {color: #d4e6e8; font-family: 'Amiri', serif;}
    .stButton>button {
        background: #2a9d8f;
        color: white;
        border-radius: 12px;
        padding: 0.7rem 2.5rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 20px rgba(42, 157, 143, 0.4);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #248f84;
        transform: translateY(-2px);
    }
    .stTextArea>div>div>textarea {
        background: #111a20;
        color: #e8ecef;
        border: 1.5px solid #2a9d8f;
        border-radius: 10px;
    }
    .stDataFrame {background: #111a20; border: 1px solid #2a5d6f;}
    section[data-testid="stSidebar"] {background: #0a0e17;}
</style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

# ------------------------------------------
# العنوان الرئيسي في المنتصف (أنيق وفخم)
# ------------------------------------------
st.markdown(
    """
    <div style="
        text-align: center;
        direction: rtl;
        padding: 60px 20px 50px;
        background: linear-gradient(to bottom, rgba(10,14,23,0.98), rgba(11,19,23,0.9));
        border-bottom: 5px solid #2a9d8f;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.7);
        margin-bottom: 40px;
    ">
        <h1 style="
            font-family: 'Amiri', serif;
            font-size: 56px;
            color: #f1faee;
            margin: 0;
            font-weight: 900;
            letter-spacing: 2px;
            text-shadow: 0 4px 15px rgba(42,157,143,0.5);
        ">
            منصة التحليل اللغوي الرقمي
        </h1>
        <p style="
            font-size: 28px;
            color: #a8dadc;
            margin: 18px 0 0;
            font-weight: 500;
            opacity: 0.95;
        ">
            لنصوص التراث العربي والدراسات اللغوية المعاصرة
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# لوجو في المنتصف أسفل العنوان (اختياري)
try:
    st.image("logo.png", width=140, use_column_width=False)
except:
    pass

st.markdown("---")

# ------------------------------------------
# Text Input
# ------------------------------------------
text_input = st.text_area(
    "أدخل النص العربي المراد تحليله:",
    height=260,
    placeholder="الصق هنا نصوصًا من القرآن الكريم، الحديث الشريف، الشعر الجاهلي، كتب التراث، أو أي نص أدبي عربي..."
)

analyze_button = st.button("بدء التحليل اللغوي", use_container_width=True)

# ------------------------------------------
# Analysis Pipeline
# ------------------------------------------
if analyze_button:
    if not text_input.strip():
        st.error("يرجى إدخال نص عربي أولاً.")
        st.stop()

    with st.spinner("جاري تنظيف النص واستخراج الجذور والإحصاءات..."):
        cleaned = clean_text(text_input)
        words = cleaned.split()
        no_stop = remove_stopwords(words)
        stems = stem_words(no_stop)

        freq = pd.Series(no_stop).value_counts().reset_index()
        freq.columns = ["ال Kلمة", "التكرار"]

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### الإحصاءات الأساسية")
            st.write(f"عدد الكلمات الأصلية: **{len(words):,}**")
            st.write(f"عدد الكلمات بعد إزالة كلمات التوقف: **{len(no_stop):,}**")
            st.write(f"عدد الجذور الفريدة المستخرجة: **{len(set(stems)):,}**")

            st.markdown("### أعلى الكلمات تكرارًا")
            st.dataframe(freq.head(20), use_container_width=True)

            st.markdown("### الجذور الأكثر تكرارًا (Stemming)")
            stem_freq = pd.Series(stems).value_counts().reset_index()
            stem_freq.columns = ["الجذر", "التكرار"]
            st.dataframe(stem_freq.head(20), use_container_width=True)

        with col_b:
            st.markdown("### سحابة الكلمات الأكثر تكرارًا")
            wc_text = " ".join(no_stop)
            wc = WordCloud(
                font_path=font_path,
                width=900, height=550,
                background_color="#0b1118",
                color_func=lambda *args, **kwargs: "#a8dadc",
                prefer_horizontal=0.9,
                max_words=200,
                contour_width=1,
                contour_color="#2a9d8f"
            ).generate(wc_text)

            fig1, ax1 = plt.subplots(figsize=(13, 8))
            ax1.imshow(wc, interpolation="bilinear")
            ax1.axis("off")
            ax1.set_facecolor('#0b1118')
            fig1.patch.set_facecolor('#0b1118')
            st.pyplot(fig1)

            st.markdown("### توزيع التكرار (أعلى 15 كلمة)")
            top15 = freq.head(15)
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            bars = ax2.bar([fix_arabic(w) for w in top15["ال Kلمة"]], top15["التكرار"], color="#2a9d8f", edgecolor="#1d7a72")
            ax2.set_xticklabels([fix_arabic(w) for w in top15["ال Kلمة"]], rotation=45, ha='right', fontproperties=font_prop, fontsize=12)
            ax2.set_facecolor('#0b1118')
            fig2.patch.set_facecolor('#0b1118')
            ax2.tick_params(colors='#e8ecef')
            ax2.spines['bottom'].set_color('#444')
            ax2.spines['top'].set_color('#444')
            ax2.spines['left'].set_color('#444')
            ax2.spines['right'].set_color('#444')
            plt.tight_layout()
            st.pyplot(fig2)

# ------------------------------------------
# Footer أنيق ومحترم
# ------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="
        text-align: center;
        padding: 40px 20px;
        color: #778da9;
        font-size: 19px;
        direction: rtl;
        background: rgba(10,14,23,0.7);
        border-top: 3px solid #2a9d8f;
        border-radius: 16px 16px 0 0;
        margin-top: 50px;
    ">
        تم تطوير هذه المنصة بكل حب وإخلاص<br>
        <strong style="font-size: 28px; color: #a8dadc;">محمد الجزائري</strong><br>
        <span style="font-size: 18px;">باحث في اللسانيات الحاسوبية والتراث العربي الرقمي</span>
    </div>
    """,
    unsafe_allow_html=True
)

