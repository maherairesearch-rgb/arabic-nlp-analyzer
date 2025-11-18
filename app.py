import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem.isri import ISRIStemmer
import nltk

nltk.download('punkt', quiet=True)

# ───────────────────────────────
# إزالة كل المسافات البيضاء الزايدة من الأعلى والجوانب
# ───────────────────────────────
st.set_page_config(page_title="أداة تحليل النصوص العربية", layout="centered")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .main > div {
        padding-top: 0rem !important;
    }
    h1 {
        margin-bottom: 0.5rem !important;
    }
    .stMarkdown {
        margin-bottom: 0rem !important;
    }
    /* إخفاء الهيدر والفوتر الافتراضي لـ Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────
# التصميم الرسمي الأنيق
# ───────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 46px; color: #1e40af; font-weight: 700; margin:0;">أداة تحليل النصوص العربية</h1>
    <p style="font-size: 20px; color: #475569; margin:0.5rem 0 2rem 0;">
        تحليل إحصائي ولغوي متقدم للنصوص العربية
    </p>
</div>
""", unsafe_allow_html=True)

# إدخال النص بدون بياض زايد
text_input = st.text_area(
    "أدخل النص العربي المراد تحليله:",
    placeholder="الصق النص هنا...",
    height=200,
    label_visibility="collapsed"
)

analyze = st.button("بدء التحليل", use_container_width=True)

# ───────────────────────────────
# القوائم والمعالجة
# ───────────────────────────────
arabic_stopwords = {
    "في","على","من","إلى","عن","ما","هذا","هذه","ذلك","التي","الذي","الذين","كان","يكون","هو","هي",
    "و","أن","إن","لا","ليس","لم","لن","قد","كما","ثم","حتى","مع","عند","بين","أو","بل","لكن","اي","أي",
    "أنا","انت","نحن","هم","هن","هؤلاء","اللذان","اللتان","اللاتي","اللائي"
}

stemmer = ISRIStemmer()

def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ───────────────────────────────
# التحليل
# ───────────────────────────────
if analyze:
    if not text_input.strip():
        st.error("الرجاء إدخال نص للتحليل.")
        st.stop()

    with st.spinner("جاري معالجة النص..."):
        cleaned = clean_text(text_input)
        words = cleaned.split()
        no_stop = [w for w in words if w not in arabic_stopwords and len(w) > 2]
        stems = [stemmer.stem(w) for w in no_stop]

        freq = pd.Series(no_stop).value_counts().head(20).reset_index()
        freq.columns = ["الكلمة", "التكرار"]

        total_words = len(words)
        after_stop = len(no_stop)
        unique_roots = len(set(stems))

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("الإحصائيات العامة")
            st.write(f"• عدد الكلمات الأصلية: **{total_words:,}**")
            st.write(f"• بعد إزالة كلمات التوقف: **{after_stop:,}**")
            st.write(f"• عدد الجذور المتميزة: **{unique_roots:,}**")

            st.subheader("أعلى الكلمات تكرارًا")
            st.dataframe(freq.head(15), use_container_width=True, hide_index=True)

            st.subheader("الجذور الشائعة")
            roots = pd.Series(stems).value_counts().head(20).reset_index()
            roots.columns = ["الجذر", "التكرار"]
            st.dataframe(roots, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("سحابة الكلمات")
            wc = WordCloud(
                width=800, height=500,
                background_color="white",
                colormap="Blues",
                min_font_size=10,
                max_font_size=120,
                relative_scaling=0.6,
                regexp=r"[\u0600-\u06FF]+"
            ).generate(" ".join(no_stop))

            fig, ax = plt.subplots(figsize=(11, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("توزيع التكرار")
            top15 = freq.head(15)[::-1]
            fig2, ax2 = plt.subplots(figsize=(11, 8))
            bars = ax2.barh(range(len(top15)), top15["التكرار"], color="#1e40af")
            ax2.set_yticks(range(len(top15)))
            ax2.set_yticklabels(top15["الكلمة"])
            ax2.set_xlabel("عدد التكرارات")
            ax2.grid(axis='x', alpha=0.3)
            for i, bar in enumerate(bars):
                ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        str(bar.get_width()), va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    st.success("تم إكمال التحليل بنجاح.")

# تذييل بسيط وأنيق
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:14px; margin-top:50px;'>"
            "أداة تحليل نصوص عربية • مفتوحة المصدر • تعمل دون ملفات خارجية</p>", 
            unsafe_allow_html=True)

# تذييل أكاديمي
st.markdown("---")
st.caption("أداة تحليل نصوص عربية مفتوحة المصدر • تعمل دون الحاجة لملفات خارجية • مُطوّرة لدعم البحث اللغوي")

