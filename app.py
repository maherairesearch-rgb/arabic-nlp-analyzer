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
# إعدادات الخط العربي
# ------------------------------------------
font_path = "Amiri-Regular.ttf"  # تأكد أن الملف موجود في نفس المجلد
font_prop = fm.FontProperties(fname=font_path)

def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))

# ------------------------------------------
# تنسيق الصفحة
# ------------------------------------------
st.set_page_config(page_title="منصة تحليل النصوص العربية", layout="wide")

# CSS مُحسّن وأنيق جدًا
st.markdown("""
<style>
    .big-title {
        font-size: 48px !important;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 22px;
        color: #475569;
        margin-bottom: 30px;
    }
    .main-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .analyze-btn {
        display: block;
        margin: 30px auto;
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        color: white !important;
        font-size: 22px !important;
        font-weight: bold !important;
        padding: 15px 50px !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    .analyze-btn:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(59, 130, 246, 0.5);
    }
    .stTextArea > div > div > textarea {
        font-size: 18px !important;
        border-radius: 15px !important;
        border: 2px solid #e2e8f0 !important;
    }
    .result-header {
        font-size: 28px;
        color: #1e40af;
        text-align: center;
        padding: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# العنوان الرئيسي
# ------------------------------------------
st.markdown('<h1 class="big-title">منصة تحليل النصوص العربية الذكية</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">أدخل أي نص عربي وسيتم تحليله تلقائيًا: إحصائيات • تكرار الكلمات • سحابة كلمات • استخراج الجذور</p>', unsafe_allow_html=True)

# ------------------------------------------
# منطقة إدخال النص
# ------------------------------------------
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    text_input = st.text_area(
        "📝 أدخل النص العربي هنا:",
        placeholder="مثال: اليوم هو يوم مشمس وجميل في مدينة الرياض...",
        height=220,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# زر التحليل الجميل
# ------------------------------------------
analyze_button = st.button("🚀 حلّل النص الآن", key="analyze", use_container_width=True)

# ------------------------------------------
# كلمات التوقف والجذر
# ------------------------------------------
arabic_stopwords = {
    "في","على","من","إلى","عن","ما","هذا","هذه","ذلك","التي","الذي","الذين","كان","يكون","هو","هي",
    "و","أن","إن","لا","ليس","لم","لن","قد","كما","ثم","حتى","مع","عند","بين","أو","بل","لكن","اي","أي"
}

stemmer = ISRIStemmer()

def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\ufb50-\ufdff\ufe70-\ufeff\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ------------------------------------------
# التحليل عند الضغط على الزر
# ------------------------------------------
if analyze_button:
    if not text_input or text_input.strip() == "":
        st.error("⚠️ الرجاء إدخال نص عربي أولاً!")
        st.stop()

    with st.spinner("🔄 جاري تحليل النص... هذا لن يستغرق سوى ثوانٍ"):
        # تنظيف النص
        cleaned = clean_text(text_input)
        words = cleaned.split()
        no_stop = [w for w in words if w not in arabic_stopwords and len(w) > 2]
        stems = [stemmer.stem(w) for w in no_stop]

        # إحصائيات
        freq = pd.Series(no_stop).value_counts().head(20).reset_index()
        freq.columns = ["الكلمة", "التكرار"]

        # سحابة الكلمات
        text_for_wc = " ".join([fix_arabic(w) for w in no_stop])

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.markdown("<h2 class='result-header'>📊 الإحصائيات الأساسية</h2>", unsafe_allow_html=True)
            st.success(f"**عدد الكلمات الأصلية:** {len(words):,}")
            st.success(f"**بعد إزالة كلمات التوقف:** {len(no_stop):,}")
            st.success(f"**عدد الجذور المستخرجة:** {len(set(stems)):,}")
            st.markdown("### 🔝 أكثر 15 كلمة تكرارًا")
            st.dataframe(freq.head(15), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # جدول الجذور
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.markdown("<h2 class='result-header'>🧬 الجذور العربية المستخرجة</h2>", unsafe_allow_html=True)
            roots_df = pd.Series(stems).value_counts().head(20).reset_index()
            roots_df.columns = ["الجذر", "عدد التكرارات"]
            st.dataframe(roots_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.markdown("<h2 class='result-header'>☁️ سحابة الكلمات</h2>", unsafe_allow_html=True)
            try:
                wc = WordCloud(
                    font_path=font_path,
                    width=800, height=500,
                    background_color="white",
                    colormap="viridis",
                    random_state=42
                ).generate(text_for_wc)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            except:
                st.warning("تعذر إنشاء سحابة الكلمات (قد يكون الخط غير موجود)")

            # المخطط العمودي
            st.markdown("<h2 class='result-header'>📈 التكرار البصري</h2>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            top_words = freq.head(12)
            bars = ax2.bar(range(len(top_words)), top_words["التكرار"], color="#3b82f6")
            ax2.set_xticks(range(len(top_words)))
            ax2.set_xticklabels([fix_arabic(w) for w in top_words["الكلمة"]], fontproperties=font_prop, rotation=45, ha='right')
            ax2.set_ylabel("عدد التكرارات", fontproperties=font_prop, fontsize=14)
            ax2.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(height),
                        ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig2)
            plt.close(fig2)
            
            st.markdown('</div>', unsafe_allow_html=True)

    st.balloons()
    st.success("✅ تم التحليل بنجاح!")

# ------------------------------------------
# تذييل
# ------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>تم التطوير بواسطة 💙 مع حب للغة العربية</p>", unsafe_allow_html=True)
