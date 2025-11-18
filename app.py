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
# إعدادات الصفحة والخط العربي
# ------------------------------------------
st.set_page_config(
    page_title="معجم | أداة تحليل النصوص العربية المتقدمة",
    page_icon="🇸🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل الخط العربي
font_path = "Amiri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)

def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

# ------------------------------------------
# تنسيق النصوص العربية في matplotlib
# ------------------------------------------
def arabic_text(text):
    return fix_arabic(text)

# ------------------------------------------
# كلمات التوقف الموسعة
# ------------------------------------------
arabic_stopwords = {
    "في", "من", "على", "إلى", "الى", "عن", "مع", "عما", "حتى", "بين", "لدى", "كان", "كانت",
    "هو", "هي", "هم", "هن", "أنا", "أنت", "أنتم", "هن", "نحن", "ذلك", "هذا", "هذه", "تلك",
    "ما", "ماذا", "متى", "أين", "كيف", "لماذا", "ليس", "لا", "ولا", "بل", "لكن", "ثم",
    "أو", "إن", "أن", "إذا", "اذا", "لو", "لعل", "قد", "سوف", "لن", "لم", "كل", "بعض",
    "جميع", "كثير", "قليل", "أكثر", "أقل", "اي", "أي", "وهو", "وهي", "وهم", "نعم", "لا"
}

# ------------------------------------------
# دوال المعالجة
# ------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\ufb50-\ufc3f\ufe70-\ufefc\s]', ' ', text)  # فقط الحروف العربية
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

stemmer = ISRIStemmer()

def process_text(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    filtered = [w for w in words if w not in arabic_stopwords and len(w) > 2]
    stems = [stemmer.stem(w) for w in filtered]
    return words, filtered, stems

# ------------------------------------------
# الواجهة الاحترافية
# ------------------------------------------
# ترويسة مميزة
st.markdown("""
<style>
    .main-title {
        font-size: 48px !important;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        font-family: 'Amiri', serif;
    }
    .subtitle {
        font-size: 22px;
        text-align: center;
        color: #475569;
        margin-bottom: 30px;
    }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">معجم</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">أداة ذكية لتحليل النصوص العربية باستخدام الجذور والإحصاءات المتقدمة</p>', unsafe_allow_html=True)

st.markdown("---")

# منطقة إدخال النص
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📝 أدخل النص العربي المراد تحليله")
        text_input = st.text_area(
            "",
            placeholder="الصق النص العربي هنا أو اكتبه مباشرة...",
            height=220,
            label_visibility="collapsed"
        )

if st.button("🚀 بدء التحليل الآن", use_container_width=True, type="primary"):
    if not text_input or text_input.strip() == "":
        st.error("⚠️ يرجى إدخال نص عربي أولاً")
        st.stop()

    with st.spinner("جاري معالجة النص واستخراج الجذور..."):
        original_words, filtered_words, stems = process_text(text_input)

        # إحصاءات سريعة
        total_words = len(original_words)
        after_stopwords = len(filtered_words)
        unique_words = len(set(filtered_words))
        unique_roots = len(set(stems))

        # إحصائيات التكرار
        freq_df = pd.DataFrame(filtered_words, columns=["كلمة"])
        word_freq = freq_df["كلمة"].value_counts().head(20).reset_index()
        word_freq.columns = ["الكلمة", "التكرار"]

        # جذور التكرار
        root_freq = pd.Series(stems).value_counts().head(20).reset_index()
        root_freq.columns = ["الجذر", "عدد الكلمات"]

    st.success("✅ تم تحليل النص بنجاح!")

    st.markdown("---")

    # الإحصائيات في صناديق ملونة
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='stats-box'><h3>{total_words:,}</h3><p>إجمالي الكلمات</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stats-box'><h3>{after_stopwords:,}</h3><p>بعد حذف كلمات التوقف</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stats-box'><h3>{unique_words}</h3><p>كلمات فريدة</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='stats-box'><h3>{unique_roots}</h3><p>جذور مختلفة</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # النتائج في أعمدة
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 📊 سحابة الكلمات الأكثر تكرارًا")
        text_cloud = " ".join([fix_arabic(w) for w in filtered_words])
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=500,
            background_color="white",
            colormap="viridis",
            max_words=150,
            contour_width=1,
            contour_color='steelblue'
        ).generate(text_cloud)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.markdown("### 🔝 أعلى 15 كلمة تكرارًا")
        top15 = word_freq.head(15)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars = ax2.barh(range(len(top15)-1, -1, -1), top15["التكرار"], color="#3b82f6")
        ax2.set_yticks(range(len(top15)-1, -1, -1))
        ax2.set_yticklabels([arabic_text(w) for w in top15["الكلمة"]], fontproperties=font_prop, fontsize=13)
        ax2.set_xlabel("عدد التكرارات", fontproperties=font_prop, fontsize=14)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")

    # جدول الجذور + جدول الكلمات
    tab1, tab2 = st.tabs(["🧬 الجذور العربية المستخرجة", "📋 الكلمات الأكثر تكرارًا"])

    with tab1:
        st.markdown("#### أقوى 20 جذرًا في النص")
        root_display = root_freq.copy()
        root_display["الجذر"] = root_display["الجذر"].apply(arabic_text)
        st.dataframe(root_display.style.background_gradient(cmap='Blues'), use_container_width=True)

    with tab2:
        st.markdown("#### جدول الكلمات الأكثر تكرارًا")
        word_display = word_freq.copy()
        word_display["الكلمة"] = word_display["الكلمة"].apply(arabic_text)
        st.dataframe(word_display.style.background_gradient(cmap='Greens'), use_container_width=True)

    # تذييل
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#64748b; font-size:14px;'>"
        "معجم – أداة تحليل النصوص العربية مفتوحة المصدر 🌟 | تم التصميم بـ ❤️ للعقول العربية المتألقة"
        "</p>",
        unsafe_allow_html=True
    )
