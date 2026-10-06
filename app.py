import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io

from extractor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_contact_info,
    extract_sections,
    extract_certifications_list,
    extract_trainings_and_internships,
    estimate_years_of_experience,
    extract_education_details
)
from analyzer import (
    calculate_ats_score,
    match_cv_to_job_description,
    extract_skills_from_text,
    detect_quantifiable_metrics,
    detect_action_verbs_and_weaknesses,
    evaluate_candidate_custom_criteria
)
from ai_advisor import (
    generate_ai_review,
    rewrite_bullet_point_ai
)
from sample_cv import SAMPLE_CV_TEXT, SAMPLE_JOB_DESCRIPTION


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CV Analyzer Pro | welcome to ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Styling */
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #252b3b 100%);
        border: 1px solid #333d52;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .verdict-card-green {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.3) 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .verdict-card-amber {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(120, 53, 15, 0.3) 100%);
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .verdict-card-red {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.3) 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .audit-section-card {
        background: #181f2f;
        border: 1px solid #2e3a50;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 3px;
    }
    .badge-green {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-blue {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-amber {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-red {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .rec-box {
        background-color: #1a2233;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    .audit-tag {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- PRESET CRITERIA PROFILES ---
PRESET_PROFILES = {
    "مهندس برمجيات (Backend / Full Stack)": {
        "role": "Senior Backend / Full Stack Engineer",
        "mandatory": "Python, FastAPI, PostgreSQL, SQL",
        "preferred": "Docker, Kafka, Redis, React, CI/CD",
        "certs": "AWS Certified, Meta",
        "trainings": "Internship, ITI, Bootcamp",
        "exp": 3,
        "degree": "Bachelor"
    },
    "عالم / مهندس بيانات (Data Scientist / AI)": {
        "role": "Data Scientist / Machine Learning Engineer",
        "mandatory": "Python, Machine Learning, SQL, Pandas, PyTorch",
        "preferred": "Docker, NLP, LLM, Git, Cloud",
        "certs": "TensorFlow, Deep Learning, AWS, Google",
        "trainings": "Internship, Nanodegree, ITI",
        "exp": 2,
        "degree": "Bachelor"
    },
    "مهندس سحابي و DevOps (Cloud & DevOps)": {
        "role": "DevOps / Cloud Platform Engineer",
        "mandatory": "Linux, Docker, Kubernetes, CI/CD, AWS",
        "preferred": "Terraform, Python, Jenkins, Ansible, Helm",
        "certs": "AWS Certified, CKA, Azure",
        "trainings": "Internship, Bootcamp",
        "exp": 3,
        "degree": "Bachelor"
    },
    "تخصيص يدوي مخصص (Custom Criteria)": {
        "role": "",
        "mandatory": "",
        "preferred": "",
        "certs": "",
        "trainings": "",
        "exp": 0,
        "degree": ""
    }
}


# --- HELPER: ATS SCORE DONUT CHART ---
def create_score_gauge(score: int, title: str = "ATS Compatibility"):
    if score >= 80:
        color = "#10b981"  # Emerald
        status_text = "مطابق ومتميز / Excellent"
    elif score >= 65:
        color = "#3b82f6"  # Blue
        status_text = "جيد جداً / Good"
    elif score >= 50:
        color = "#f59e0b"  # Amber
        status_text = "متوسط / Moderate"
    else:
        color = "#ef4444"  # Red
        status_text = "ضعيف / Needs Review"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{title}</b><br><span style='font-size:0.85em;color:{color}'>{status_text}</span>", 'font': {'size': 17}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ]
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#f8fafc"}
    )
    return fig


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=60)
    st.title("CV Analyzer Pro")
    st.caption("محلل ومطابق السير الذاتية لمعايير التوظيف")
    st.markdown("---")

    st.subheader("1. تحميل ملف السيرة الذاتية (CV)")
    uploaded_file = st.file_uploader("ارفع ملف المرشح (PDF أو DOCX)", type=["pdf", "docx"])
    use_sample = st.checkbox("أو تجربة نموذج CV جاهز (Demo CV)", value=False)

    st.markdown("---")
    st.subheader("2. شروط واشتراطات مسؤول التوظيف (Criteria)")
    
    preset_choice = st.selectbox(
        "اختر قالباً وظيفياً جاهزاً أو خصص بنفسك:",
        list(PRESET_PROFILES.keys()),
        index=0
    )
    preset_data = PRESET_PROFILES[preset_choice]

    target_role = st.text_input(
        "المسمى الوظيفي المستهدف:",
        value=preset_data["role"],
        placeholder="e.g. Senior Backend Engineer"
    )

    with st.expander("🛠️ تحديد المهارات والاشتراطات بدقة", expanded=True):
        mandatory_input = st.text_input(
            "المهارات الإلزامية (Mandatory Skills) *",
            value=preset_data["mandatory"],
            help="افصل بين المهارات بفاصلة (,). مثل: Python, SQL, Docker",
            placeholder="Python, FastAPI, SQL"
        )
        preferred_input = st.text_input(
            "المهارات الإضافية المفضلة (Preferred):",
            value=preset_data["preferred"],
            placeholder="Kafka, Redis, React"
        )
        certs_input = st.text_input(
            "الشهادات المطلوبة (Certifications):",
            value=preset_data["certs"],
            placeholder="AWS Certified, PMP, CCNA"
        )
        trainings_input = st.text_input(
            "التدريبات / الكورسات المفضلة (Trainings):",
            value=preset_data["trainings"],
            placeholder="Internship, ITI, Bootcamp, NTI"
        )
        exp_input = st.number_input(
            "الحد الأدنى لسنوات الخبرة (بالسنوات):",
            min_value=0,
            max_value=25,
            value=preset_data["exp"]
        )
        degree_input = st.text_input(
            "المؤهل الدراسي المطلوب (Degree):",
            value=preset_data["degree"],
            placeholder="Bachelor / بكالوريوس"
        )

    with st.expander("📝 إعلان الوظيفة الكامل (Job Description - اختياري)", expanded=False):
        jd_text = st.text_area(
            "الصق نص إعلان الوظيفة هنا:",
            value=SAMPLE_JOB_DESCRIPTION if use_sample else "",
            height=130,
            placeholder="الصق نص إعلان الوظيفة لمطابقة النص الكامل..."
        )

    st.markdown("---")
    st.subheader("3. (AI Coach)")
    api_key_input = st.text_input(
        "Gemini API Key (اختياري)",
        type="password",
        placeholder="AIzaSy...",
        help="send me message gmail : ahmedsoph248@gmail.com for a free API key from Google AI Studio"
    )


# --- EXTRACT CV TEXT ---
raw_text = ""
page_count = 1

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    if uploaded_file.name.endswith(".pdf"):
        raw_text, page_count = extract_text_from_pdf(io.BytesIO(file_bytes))
    elif uploaded_file.name.endswith(".docx"):
        raw_text, page_count = extract_text_from_docx(io.BytesIO(file_bytes))
elif use_sample:
    raw_text = SAMPLE_CV_TEXT
    page_count = 2


# If no CV uploaded or selected
if not raw_text.strip():
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 2.5rem; margin-bottom: 10px;">🎯 welcome to <span style="color: #3b82f6;">CV Analyzer Pro</span></h1>
        <p style="font-size: 1.2rem; color: #94a3b8; max-width: 800px; margin: 0 auto 30px auto;">
            النظام الاحترافي لمسؤولي التوظيف والباحثين عن عمل: حدد <b>المهارات الإلزامية، الشهادات المطلوبة، والتدريبات</b> وسيقوم النظام بفحص السيرة الذاتية 
            وتفسير مطابقة كل محور على حدة مع قرار فوري للمقابلة!
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 8px;">🏢</div>
            <div class="metric-title">Recruiter Criteria Match</div>
            <div style="font-size: 0.95rem; color: #cbd5e1;">فحص وتفسير المهارات والشهادات والتدريبات المطلوبة بدقة</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 8px;">🎯</div>
            <div class="metric-title">ATS Compatibility</div>
            <div style="font-size: 0.95rem; color: #cbd5e1;">قياس قابلية القراءة واجتياز فلاتر أنظمة التوظيف العالمية</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
            <div class="metric-title">STAR & Impact Metrics</div>
            <div style="font-size: 0.95rem; color: #cbd5e1;">رصد الإنجازات الرقمية والنسب واستبدال العبارات الضعيفة</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 8px;">🤖</div>
            <div class="metric-title">AI Career Coach</div>
            <div style="font-size: 0.95rem; color: #cbd5e1;">مستشار ذكي لتحليل SWOT وأسئلة المقابلات الشخصية المتوقعة</div>
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 للبدء، قم برفع ملف CV من القائمة الجانبية أو فعّل خيار **'تجربة نموذج CV جاهز (Demo CV)'** لتجربة الفحص الفوري!")
    st.stop()


# --- BUILD CRITERIA DICT ---
criteria_dict = {
    "target_role": target_role,
    "mandatory_skills": [s.strip() for s in mandatory_input.split(",") if s.strip()],
    "preferred_skills": [s.strip() for s in preferred_input.split(",") if s.strip()],
    "required_certifications": [c.strip() for c in certs_input.split(",") if c.strip()],
    "required_trainings": [t.strip() for t in trainings_input.split(",") if t.strip()],
    "min_years_experience": exp_input,
    "required_degree": degree_input
}


# --- EXECUTE CORE AUDIT ---
with st.spinner("جاري فحص السيرة الذاتية ومطابقة الشروط بدقة..."):
    ats_results = calculate_ats_score(raw_text, page_count)
    criteria_audit = evaluate_candidate_custom_criteria(raw_text, criteria_dict)
    jd_match_results = match_cv_to_job_description(raw_text, jd_text) if jd_text.strip() else None

score = ats_results["overall_score"]
metrics = ats_results["metrics"]
contacts = ats_results["contacts"]
v_score = criteria_audit["verdict_score"]
v_badge = criteria_audit["verdict_badge"]


# --- HEADER DASHBOARD SUMMARY ---
st.markdown("## 📊 نتائج فحص وتدقيق المرشح")

col_gauge, col_stats = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(create_score_gauge(v_score, title="نسبة استيفاء شروط الوظيفة"), use_container_width=True)

with col_stats:
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">قرار الفرز الأولي</div>
            <div class="metric-value" style="font-size: 1.3rem; color: {'#10b981' if v_badge == 'green' else ('#f59e0b' if v_badge == 'amber' else '#ef4444')};">
                {criteria_audit['decision_label']}
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">للوظيفة: {target_role or 'غير محدد'}</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">الخبرة العملية المقدرة</div>
            <div class="metric-value" style="color: {'#10b981' if criteria_audit['experience_audit']['meets_experience'] else '#f59e0b'};">
                {criteria_audit['experience_audit']['candidate_years']} سنوات
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">المطلوب: {exp_input}+ سنوات</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">توافق نظام الـ ATS</div>
            <div class="metric-value" style="color: #60a5fa;">{score}/100</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">{metrics['word_count']} كلمة (~{metrics['page_count']} صفحة)</div>
        </div>
        """, unsafe_allow_html=True)

    # Contact Details row
    st.markdown("##### 📇 بيانات التواصل المكتشفة في الـ CV:")
    contact_badges = []
    if contacts.get("email"):
        contact_badges.append(f"<span class='badge badge-green'>✉️ {contacts['email']}</span>")
    else:
        contact_badges.append("<span class='badge badge-red'>❌ إيميل مفقود</span>")

    if contacts.get("phone"):
        contact_badges.append(f"<span class='badge badge-green'>📞 {contacts['phone']}</span>")
    else:
        contact_badges.append("<span class='badge badge-red'>❌ رقم هاتف مفقود</span>")

    if contacts.get("linkedin"):
        contact_badges.append("<span class='badge badge-blue'>🔗 LinkedIn متوفر</span>")
    else:
        contact_badges.append("<span class='badge badge-amber'>⚠️ رابط LinkedIn مفقود</span>")

    if contacts.get("github"):
        contact_badges.append("<span class='badge badge-blue'>🐙 GitHub متوفر</span>")

    st.markdown(" ".join(contact_badges), unsafe_allow_html=True)


st.markdown("---")


# --- TABS FOR IN-DEPTH ANALYSIS ---
tab_criteria, tab_ats, tab_jd, tab_skills, tab_star, tab_ai, tab_raw = st.tabs([
    "🎯 تقرير فحص ومطابقة الشروط (Recruiter Audit)",
    "📈 توافق ATS والتوصيات",
    "📝 مطابقة إعلان الوظيفة (JD Match)",
    "🛠️ تصنيف المهارات الشامل",
    "✍️ صياغة الإنجازات (STAR)",
    "🤖 استشارة الذكاء الاصطناعي (AI Coach)",
    "📄 استعراض النص والأقسام"
])


# ==========================================
# --- TAB 1: RECRUITER CRITERIA AUDIT (NEW!) ---
# ==========================================
with tab_criteria:
    # 1. VERDICT CARD
    card_class = f"verdict-card-{v_badge}"
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.9rem; text-transform: uppercase; color: #cbd5e1; letter-spacing: 0.05em;">نتيجة التقييم النهائي لشروط الوظيفة</span>
                <h2 style="margin: 4px 0 8px 0; color: #f8fafc;">{criteria_audit['verdict_status']}</h2>
                <div style="font-size: 1.05rem; color: #e2e8f0;">المسمى الوظيفي: <b>{target_role or 'غير محدد'}</b> | نسبة التطابق: <b>{v_score}%</b></div>
            </div>
            <div style="font-size: 3rem; font-weight: 800; color: #f8fafc; padding: 10px 20px;">
                {v_score}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. FOUR DETAILED AUDIT SECTIONS
    st.markdown("### 📋 التفسير والتدقيق المنفصل لكل محور:")

    # --- SECTION A: SKILLS AUDIT ---
    skills_audit = criteria_audit["skills_audit"]
    st.markdown("""
    <div class="audit-section-card">
        <h4 style="color: #60a5fa; margin-bottom: 12px;">🛠️ 1. فحص وتفسير المهارات (Skills Audit)</h4>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**المهارات الإلزامية المطلوبة (Mandatory Skills):**")
        if criteria_dict["mandatory_skills"]:
            m_badges = []
            for m in skills_audit["matched_mandatory"]:
                m_badges.append(f"<span class='badge badge-green'>✓ {m} (متوفر)</span>")
            for mis in skills_audit["missing_mandatory"]:
                m_badges.append(f"<span class='badge badge-red'>✗ {mis} (مفقود)</span>")
            st.markdown(" ".join(m_badges), unsafe_allow_html=True)
        else:
            st.info("لم تُحدد مهارات إلزامية في القائمة الجانبية.")

    with col_s2:
        st.markdown("**المهارات الإضافية المفضلة (Preferred Skills):**")
        if criteria_dict["preferred_skills"]:
            p_badges = []
            for p in skills_audit["matched_preferred"]:
                p_badges.append(f"<span class='badge badge-green'>✓ {p}</span>")
            for pmis in skills_audit["missing_preferred"]:
                p_badges.append(f"<span class='badge badge-amber'>- {pmis}</span>")
            st.markdown(" ".join(p_badges), unsafe_allow_html=True)
        else:
            st.caption("لم تُحدد مهارات إضافية.")

    st.markdown(f"""
        <div style="margin-top: 14px; padding: 10px 14px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border-left: 3px solid #3b82f6;">
            <b>💡 التفسير والتحليل:</b> {skills_audit['explanation']} (نسبة تغطية المهارات: <b>{skills_audit['score_pct']}%</b>)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION B: CERTIFICATIONS AUDIT ---
    certs_audit = criteria_audit["certifications_audit"]
    st.markdown(f"""
    <div class="audit-section-card">
        <h4 style="color: #10b981; margin-bottom: 12px;">📜 2. فحص وتفسير الشهادات والاعتمادات (Certifications Audit)</h4>
        <div style="margin-bottom: 10px;">
            <b>حالة استيفاء شرط الشهادات:</b> 
            <span class="badge {'badge-green' if certs_audit['status'] == 'مستوفى بالكامل (100%)' else ('badge-amber' if 'جزئياً' in certs_audit['status'] else 'badge-red')}">
                {certs_audit['status']}
            </span>
        </div>
    """, unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**الشهادات التي حددها مسؤول التوظيف:**")
        if criteria_dict["required_certifications"]:
            rc_badges = []
            for c in certs_audit["matched"]:
                rc_badges.append(f"<span class='badge badge-green'>✓ {c} (حاصل عليها)</span>")
            for cmis in certs_audit["missing"]:
                rc_badges.append(f"<span class='badge badge-red'>✗ {cmis} (غير متوفرة)</span>")
            st.markdown(" ".join(rc_badges), unsafe_allow_html=True)
        else:
            st.caption("لم تُحدد شهادات معينة كشرط إلزامي.")

    with col_c2:
        st.markdown("**الشهادات المعتمدة المكتشفة في السيرة الذاتية:**")
        if certs_audit["all_detected_in_cv"]:
            for found_c in certs_audit["all_detected_in_cv"]:
                st.markdown(f"- 🎖️ **{found_c}**")
        else:
            st.warning("لم يتم العثور على شهادات رسمية مسجلة في السيرة الذاتية.")

    st.markdown(f"""
        <div style="margin-top: 14px; padding: 10px 14px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 3px solid #10b981;">
            <b>💡 التفسير والتحليل:</b> {certs_audit['explanation']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION C: TRAININGS & INTERNSHIPS AUDIT ---
    trainings_audit = criteria_audit["trainings_audit"]
    st.markdown(f"""
    <div class="audit-section-card">
        <h4 style="color: #f59e0b; margin-bottom: 12px;">🎓 3. فحص وتفسير التدريبات والكورسات الصيفية (Trainings & Internships Audit)</h4>
        <div style="margin-bottom: 10px;">
            <b>حالة استيفاء البرامج التدريبية:</b> 
            <span class="badge {'badge-green' if trainings_audit['status'] == 'مستوفى بالكامل' else 'badge-amber'}">
                {trainings_audit['status']}
            </span>
        </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**التدريبات والكورسات المطلوبة من جهة العمل:**")
        if criteria_dict["required_trainings"]:
            rt_badges = []
            for t in trainings_audit["matched"]:
                rt_badges.append(f"<span class='badge badge-green'>✓ {t} (تم اجتيازه)</span>")
            for tmis in trainings_audit["missing"]:
                rt_badges.append(f"<span class='badge badge-amber'>- {tmis} (غير مذكور)</span>")
            st.markdown(" ".join(rt_badges), unsafe_allow_html=True)
        else:
            st.caption("لم تُحدد برامج تدريبية خاصة.")

    with col_t2:
        st.markdown("**سجل التدريبات والـ Internships والبرامج المرصودة في الـ CV:**")
        if trainings_audit["all_detected_in_cv"]:
            for tr_item in trainings_audit["all_detected_in_cv"]:
                st.markdown(f"- 💼 **{tr_item['title']}**")
                if tr_item.get("details"):
                    st.caption(f"  {tr_item['details'][:140]}...")
        else:
            st.info("لم يذكر المرشح فترات تدريب صيفي أو برامج تدريبية منفصلة (قد يركز مباشرة على الخبرات الوظيفية).")

    st.markdown(f"""
        <div style="margin-top: 14px; padding: 10px 14px; background: rgba(245, 158, 11, 0.1); border-radius: 8px; border-left: 3px solid #f59e0b;">
            <b>💡 التفسير والتحليل:</b> {trainings_audit['explanation']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION D: EXPERIENCE & EDUCATION AUDIT ---
    exp_audit = criteria_audit["experience_audit"]
    edu_audit = criteria_audit["education_audit"]
    st.markdown(f"""
    <div class="audit-section-card">
        <h4 style="color: #a855f7; margin-bottom: 12px;">💼 4. فحص سنوات الخبرة والمؤهل الدراسي (Experience & Education)</h4>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("**تقييم سنوات الخبرة العملية:**")
        st.write(f"- خبرة المرشح المقدرة: **{exp_audit['candidate_years']} سنوات**")
        st.write(f"- الحد الأدنى المطلوب: **{exp_audit['required_years']} سنوات**")
        if exp_audit["meets_experience"]:
            st.success(f"✓ {exp_audit['explanation']}")
        else:
            st.error(f"✗ {exp_audit['explanation']}")

    with col_e2:
        st.markdown("**المؤهل الدراسي والتخصص:**")
        st.write(f"- المؤهل المطلوب: **{criteria_dict['required_degree'] or 'غير محدد'}**")
        st.write(f"- الدرجات العلمية المرصودة: **{', '.join(edu_audit['detected_degrees']) if edu_audit['detected_degrees'] else 'مؤهل جامعي'}**")
        if edu_audit["meets_education"]:
            st.success(f"✓ {edu_audit['explanation']}")
        else:
            st.warning(f"⚠️ {edu_audit['explanation']}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Download Candidate Screening Summary Button
    screening_summary = f"""# تقرير قرار الفرز النهائي للمرشح
المسمى الوظيفي المستهدف: {target_role}
القرار النهائي: {criteria_audit['decision_label']} ({criteria_audit['verdict_status']})
نسبة مطابقة المعايير: {v_score}%

1. فحص المهارات:
- المهارات الإلزامية المتوفرة: {', '.join(skills_audit['matched_mandatory'])}
- المهارات الإلزامية المفقودة: {', '.join(skills_audit['missing_mandatory']) if skills_audit['missing_mandatory'] else 'لا يوجد'}
- التحليل: {skills_audit['explanation']}

2. فحص الشهادات:
- الحالة: {certs_audit['status']}
- الشهادات المتوفرة: {', '.join(certs_audit['all_detected_in_cv'])}
- التحليل: {certs_audit['explanation']}

3. فحص التدريبات والكورسات:
- الحالة: {trainings_audit['status']}
- التحليل: {trainings_audit['explanation']}

4. فحص الخبرة والمؤهل:
- سنوات الخبرة: {exp_audit['candidate_years']} (المطلوب: {exp_audit['required_years']})
- تحليل الخبرة: {exp_audit['explanation']}
- تحليل المؤهل: {edu_audit['explanation']}
"""
    st.download_button(
        label="📥 تحميل تقرير قرار الفرز (Screening Report)",
        data=screening_summary,
        file_name="Candidate_Screening_Report.md",
        mime="text/markdown"
    )


# ==========================================
# --- TAB 2: ATS SCORE & GENERAL RECOMMENDATIONS ---
# ==========================================
with tab_ats:
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("تفصيل نقاط تقييم ATS")
        breakdown_data = []
        for cat, val in ats_results["breakdown"].items():
            breakdown_data.append({
                "Category": cat,
                "Score": val["score"],
                "Max": val["max"],
                "Percentage": round((val["score"] / val["max"]) * 100)
            })
        df_breakdown = pd.DataFrame(breakdown_data)

        fig_bar = px.bar(
            df_breakdown,
            x="Score",
            y="Category",
            orientation="h",
            text="Score",
            range_x=[0, 25],
            color="Percentage",
            color_continuous_scale="Blues",
            labels={"Score": "النقاط المحققة", "Category": "المعيار"}
        )
        fig_bar.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': '#f8fafc'},
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.subheader("قائمة الإجراءات لتحسين الـ ATS")
        if ats_results["recommendations"]:
            for idx, rec in enumerate(ats_results["recommendations"], 1):
                st.markdown(f"""
                <div class="rec-box">
                    <b>{idx}.</b> {rec}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 تهانينا! السيرة الذاتية متوافقة بدرجة ممتازة مع المعايير القياسية للـ ATS.")

        st.markdown("##### 📑 فحص الأقسام الأساسية:")
        p_sections = ats_results["sections"]["present_sections"]
        sec_badges = []
        for essential in ["Summary", "Experience", "Education", "Skills", "Projects", "Certifications"]:
            if essential in p_sections:
                sec_badges.append(f"<span class='badge badge-green'>✓ {essential}</span>")
            else:
                sec_badges.append(f"<span class='badge badge-amber'>✗ {essential} (غير مؤكد)</span>")
        st.markdown(" ".join(sec_badges), unsafe_allow_html=True)


# ==========================================
# --- TAB 3: FULL JOB DESCRIPTION MATCH ---
# ==========================================
with tab_jd:
    if jd_match_results:
        m_score = jd_match_results["match_score"]
        col_m1, col_m2 = st.columns([1, 2])

        with col_m1:
            st.markdown(f"""
            <div class="metric-card" style="padding: 26px;">
                <div class="metric-title">نسبة مطابقة نص الوظيفة</div>
                <div class="metric-value" style="font-size: 2.8rem; color: {'#10b981' if m_score >= 70 else ('#f59e0b' if m_score >= 50 else '#ef4444')};">
                    {m_score}%
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                    تشابه المحتوى: {jd_match_results['similarity_percent']}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"#### التوافق مع إعلان: **{target_role or 'الوظيفة المحددة'}**")
            st.progress(m_score / 100)
            if m_score >= 75:
                st.success("سيرتك الذاتية متوافقة بشكل قوي جداً مع نص إعلان الوظيفة!")
            elif m_score >= 55:
                st.warning("توافق جيد، لكن هناك مصطلحات هامة في إعلان الوظيفة غير مذكورة في سيرتك الذاتية.")
            else:
                st.error("نسبة المطابقة منخفضة. يوصى بإضافة الكلمات المفتاحية الناقصة أدناه.")

        st.markdown("---")
        c_match_l, c_match_r = st.columns(2)

        with c_match_l:
            st.markdown("#### ✅ الكلمات والمهارات المتطابقة:")
            if jd_match_results["matching_skills"]:
                matching_html = " ".join([f"<span class='badge badge-green'>{s}</span>" for s in jd_match_results["matching_skills"]])
                st.markdown(matching_html, unsafe_allow_html=True)
            else:
                st.write("لم يتم العثور على مهارات مشتركة واضحة.")

        with c_match_r:
            st.markdown("#### ⚠️ مهارات في الإعلان وغير موجودة في الـ CV:")
            if jd_match_results["missing_skills"]:
                missing_html = " ".join([f"<span class='badge badge-red'>{s}</span>" for s in jd_match_results["missing_skills"]])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.success("سيرتك الذاتية تغطي جميع المهارات الأساسية المذكورة في نص الوظيفة!")
    else:
        st.info("💡 لم تقم بلصق نص كامل لإعلان الوظيفة (Job Description). يمكنك لصقه في القائمة الجانبية لحساب نسبة التطابق الدقيقة.")


# ==========================================
# --- TAB 4: COMPREHENSIVE SKILLS TAXONOMY ---
# ==========================================
with tab_skills:
    st.subheader("تصنيف المهارات المستخرجة من السيرة الذاتية")
    skills_data = ats_results["skills"]

    if skills_data:
        cat_counts = [{"Category": cat, "Count": len(items)} for cat, items in skills_data.items()]
        df_cats = pd.DataFrame(cat_counts)
        fig_skills = px.pie(
            df_cats,
            names="Category",
            values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_skills.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': '#f8fafc'},
            margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_skills, use_container_width=True)

        cols = st.columns(3)
        cat_items = list(skills_data.items())
        for idx, (category, items) in enumerate(cat_items):
            col_target = cols[idx % 3]
            with col_target:
                st.markdown(f"##### {category} ({len(items)})")
                badges_html = " ".join([f"<span class='badge badge-blue'>{item}</span>" for item in items])
                st.markdown(badges_html, unsafe_allow_html=True)
                st.write("")
    else:
        st.warning("لم يتم العثور على مهارات تقنية واضحة من القائمة القياسية.")


# ==========================================
# --- TAB 5: STAR & BULLET REWRITER ---
# ==========================================
with tab_star:
    st.subheader("فحص صياغة الإنجازات (STAR Method & Action Verbs)")

    col_star1, col_star2 = st.columns(2)

    with col_star1:
        st.markdown("#### 🎯 الأرقام والنتائج المقاسة (Quantifiable Metrics)")
        if ats_results["quant_metrics"]:
            st.success(f"تم رصد {len(ats_results['quant_metrics'])} قياسات رقمية / نسب مئوية:")
            for m in ats_results["quant_metrics"]:
                st.markdown(f"- 📌 `{m}`")
        else:
            st.warning("⚠️ لم يتم رصد نتائج رقمية كافية (مثل: زيادة بنسبة 30%، إدارة ميزانية 50 ألف دولار، 100 ألف مستخدم).")

    with col_star2:
        st.markdown("#### ⚡ الأفعال القوية والعبارات الضعيفة")
        p_verbs = ats_results["power_verbs"]
        w_phrases = ats_results["weak_phrases"]

        if p_verbs:
            st.markdown(f"**أفعال إنجاز قوية مستخدمة ({len(p_verbs)}):**")
            p_html = " ".join([f"<span class='badge badge-green'>{v}</span>" for v in p_verbs[:12]])
            st.markdown(p_html, unsafe_allow_html=True)

        if w_phrases:
            st.markdown(f"<br>**⚠️ عبارات ضعيفة/سلبية يفضل استبدالها ({len(w_phrases)}):**", unsafe_allow_html=True)
            w_html = " ".join([f"<span class='badge badge-red'>{p}</span>" for p in w_phrases])
            st.markdown(w_html, unsafe_allow_html=True)
        else:
            st.markdown("<br><span class='badge badge-green'>✓ لا توجد عبارات تقليدية ضعيفة مرصودة</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🛠️ أداة إعادة صياغة النقاط (Interactive Bullet Point Rewriter)")
    st.caption("الصق أي نقطة خبرة عادية وسيقوم المحلل بتحويلها إلى 3 صياغات قوية وفق معيار STAR:")

    sample_bullet = "Responsible for managing payment system and fixing bugs."
    bullet_input = st.text_input("نقطة الخبرة المراد تحسينها:", value=sample_bullet)

    if st.button("🚀 إعادة صياغة النقطة بنماذج احترافية", key="btn_rewrite"):
        with st.spinner("جاري صياغة الخيارات الاحترافية..."):
            rewrite_res = rewrite_bullet_point_ai(
                bullet_input,
                target_role=target_role,
                api_key=api_key_input
            )
            st.markdown(f"**المصدر:** `{rewrite_res['source']}`")
            st.markdown(rewrite_res["result"])


# ==========================================
# --- TAB 6: AI CAREER COACH ---
# ==========================================
with tab_ai:
    st.subheader("🤖 استشارة المستشار المهني بالذكاء الاصطناعي")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_ai_btn = st.button("✨ توليد التقرير الاستشاري الشامل", type="primary")
    with col_info:
        if api_key_input:
            st.success("متصل بنموذج **Gemini 2.5 Flash** المباشر.")
        else:
            st.info("يعمل حالياً عبر المستشار الخوارزمي المدمج. يمكنك إدخال مفتاح Gemini API في القائمة الجانبية للتحليل الحي.")

    if run_ai_btn or "ai_report_cache" in st.session_state:
        if run_ai_btn or "ai_report_cache" not in st.session_state:
            with st.spinner("جاري إعداد التحليل الشامل والأسئلة المتوقعة..."):
                report_data = generate_ai_review(
                    raw_text,
                    target_role=target_role,
                    job_description=jd_text,
                    api_key=api_key_input
                )
                st.session_state["ai_report_cache"] = report_data

        cached = st.session_state["ai_report_cache"]
        st.markdown(f"**محرك الاستشارة:** `{cached['source']}`")
        st.markdown(cached["content"])


# ==========================================
# --- TAB 7: RAW TEXT & PARSED SECTIONS ---
# ==========================================
with tab_raw:
    st.subheader("استعراض المحتوى المستخرج وتصدير التقرير")

    c_raw1, c_raw2 = st.columns([2, 1])

    with c_raw1:
        st.markdown("##### النص المستخرج:")
        st.text_area("الـ CV الكامل بعد المعالجة", raw_text, height=350)

    with c_raw2:
        st.markdown("##### 📥 تصدير تقرير التدقيق:")
        report_summary = f"""# تقرير تدقيق السيرة الذاتية - CV Analyzer Pro
المسمى الوظيفي المستهدف: {target_role or 'Not specified'}

## التقييم الإجمالي
- نسبة توافق الشروط المطلوبة: {v_score}/100 ({criteria_audit['decision_label']})
- نسبة توافق ATS: {score}/100
- عدد الكلمات: {metrics['word_count']}
- سنوات الخبرة المقدرة: {criteria_audit['experience_audit']['candidate_years']} سنوات
- الشهادات المكتشفة: {len(criteria_audit['certifications_audit']['all_detected_in_cv'])}
- البرامج التدريبية المكتشفة: {len(criteria_audit['trainings_audit']['all_detected_in_cv'])}
"""
        st.download_button(
            label="📄 تحميل التقرير الكامل بصيغة Markdown",
            data=report_summary,
            file_name="CV_Full_Audit_Report.md",
            mime="text/markdown"
        )
st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; padding: 20px 0;">
        <h4>Developed by Ahmed Sobh</h4>
        <p>AI & Data Science Student</p>
    </div>
    """,
    unsafe_allow_html=True
)