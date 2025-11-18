import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem.isri import ISRIStemmer
import nltk

nltk.download('punkt', quiet=True)

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="أداة تحليل النصوص العربية", layout="wide")

# تصميم هادئ ورسمي جدًا
st.markdown("""
<style>
    .title {
        font-size: 46px !important;
        font-weight: 700;
        text-align: center;
        color: #1e40af;
        margin-bottom: 10px;
        font-family: 'Segoe UI', sans-serif;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #475569;
        margin-bottom: 40px;
    }
    .card {
        background: white;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 18px 0;
    }
    .analyze-btn {
        display: block;
        margin: 35px auto;
        background: #1e40af !important;
        color: white !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 14px 50px !important;
        border-radius: 12px !important;
        border: none !important;
    }
    .analyze-btn:hover {
        background: #1e3a8a !important;
    }
    .stTextArea > div > div > textarea {
        font-size: 17px !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرسمي
st.markdown('<h1 class="title">أداة تحليل النصوص العربية</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">تحليل إحصائي ولغوي للنصوص العربية<br>إحصاءات الكلمات • تكرار المفردات • استخراج الجذور • سحابة الكلمات</p>', unsafe_allow_html=True)

# إدخال النص
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    text_input = st.text_area(
        "أدخل النص العربي المراد تحليله:",
        placeholder="الصق النص هنا...",
        height=200,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("بدء التحليل", use_container_width=True)

# قوائم التوقف والجذر
arabic_stopwords = {
    "في","على","من","إلى","عن","ما","هذا","هذه","ذلك","التي","الذي","الذين","كان","يكون","هو","هي",
    "و","أن","إن","لا","ليس","لم","لن","قد","كما","ثم","حتى","مع","عند","بين","أو","بل","لكن","اي","أي",
    "أنا","انت","نحن","هم","هن","هؤلاء","ذلك","تلك","اللذان","اللتان","اللاتي","اللائي"
}

stemmer = ISRIStemmer()

def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# التحليل
if analyze:
    if not text_input or text_input.strip() == "":
        st.error("الرجاء إدخال نص للتحليل.")
        st.stop()

    with st.spinner("جاري معالجة النص واستخراج الإحصائيات..."):
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
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("الإحصائيات العامة")
            st.write(f"• عدد الكلمات في النص الأصلي: **{total_words:,}**")
            st.write(f"• عدد الكلمات بعد إزالة كلمات التوقف: **{after_stop:,}**")
            st.write(f"• عدد الجذور المتميزة المستخرجة: **{unique_roots:,}**")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("أعلى الكلمات تكرارًا (15 كلمة)")
            st.dataframe(freq.head(15), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("الجذور الشائعة المستخرجة")
            roots = pd.Series(stems).value_counts().head(20).reset_index()
            roots.columns = ["الجذر", "عدد مرات الظهور"]
            st.dataframe(roots, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("توزيع التكرار (أفقي)")
            top15 = freq.head(15)[::-1]

            fig2, ax2 = plt.subplots(figsize=(11, 8))
            bars = ax2.barh(range(len(top15)), top15["التكرار"], color="#1e40af")

            ax2.set_yticks(range(len(top15)))
            ax2.set_yticklabels(top15["الكلمة"])
            ax2.set_xlabel("عدد التكرارات")
            ax2.grid(axis='x', alpha=0.3)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                        str(width), va='center', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            st.markdown('</div>', unsafe_allow_html=True)

    # رسالة نهاية محترمة بدون بالونات
    st.markdown("---")
    st.success("تم إكمال التحليل بنجاح.")

# تذييل أكاديمي
st.markdown("---")
st.caption("أداة تحليل نصوص عربية مفتوحة المصدر • تعمل دون الحاجة لملفات خارجية • مُطوّرة لدعم البحث اللغوي")
