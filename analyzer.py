import re
from typing import Dict, List, Any, Set, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from extractor import (
    extract_contact_info,
    extract_sections,
    extract_certifications_list,
    extract_trainings_and_internships,
    estimate_years_of_experience,
    extract_education_details
)


# --- SKILLS TAXONOMIES ---
SKILLS_TAXONOMY = {
    "Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "sql", "bash", "shell", "r", "scala", "dart", "html", "css"
    ],
    "Frameworks & Web": [
        "react", "react.js", "next.js", "vue", "vue.js", "angular", "node.js", "express",
        "django", "fastapi", "flask", "spring boot", "asp.net", ".net", "flutter",
        "react native", "tailwind", "tailwind css", "bootstrap", "graphql", "rest api"
    ],
    "Data & AI": [
        "machine learning", "deep learning", "nlp", "computer vision", "llm", "llms",
        "generative ai", "langchain", "rag", "pytorch", "tensorflow", "keras",
        "scikit-learn", "pandas", "numpy", "power bi", "tableau", "data analysis",
        "data engineering", "spark", "hadoop", "etl"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "ci/cd",
        "github actions", "terraform", "jenkins", "linux", "nginx", "ansible",
        "helm", "serverless", "microservices"
    ],
    "Databases": [
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "sqlite", "cassandra", "dynamodb", "oracle", "firebase", "supabase"
    ],
    "Tools & Practices": [
        "git", "github", "gitlab", "jira", "confluence", "agile", "scrum",
        "kanban", "tdd", "unit testing", "system design", "postman", "kafka", "rabbitmq"
    ],
    "Soft Skills": [
        "leadership", "communication", "problem solving", "critical thinking",
        "teamwork", "collaboration", "time management", "adaptability",
        "mentorship", "negotiation", "ownership", "analytical skills", "presentation"
    ]
}

POWER_ACTION_VERBS = [
    "spearheaded", "orchestrated", "engineered", "architected", "accelerated",
    "amplified", "streamlined", "revamped", "championed", "maximized",
    "minimized", "reduced", "increased", "boosted", "generated", "surpassed",
    "automated", "delivered", "negotiated", "cultivated", "pioneered",
    "transformed", "optimized", "implemented", "developed", "launched",
    "designed", "executed", "established", "led", "mentored", "consolidated"
]

WEAK_PHRASES = [
    "responsible for", "duties included", "helped with", "worked on",
    "assisted in", "tasked with", "participated in", "involved with",
    "did", "handled daily"
]


def extract_skills_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extracts categorized skills from text.
    """
    text_lower = f" {text.lower()} "
    detected = {}
    
    for category, skill_list in SKILLS_TAXONOMY.items():
        found = []
        for skill in skill_list:
            # Match whole words or phrases
            pattern = r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])'
            if re.search(pattern, text_lower):
                found.append(skill.title() if len(skill) > 3 and not skill.isupper() else skill.upper())
        if found:
            detected[category] = list(set(found))
            
    return detected


def detect_quantifiable_metrics(text: str) -> List[str]:
    """
    Detects metrics like percentages, revenue, numbers + units, indicating STAR impact.
    """
    patterns = [
        r'\b\d+(?:\.\d+)?%\b',  # percentages: 40%, 15.5%
        r'\$\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*[kKmMbB])?\b',  # currency: $50k, $1.2M
        r'\b\d+(?:,\d+)*\+\s*(?:users|clients|customers|requests|transactions|records|engineers|projects|downloads|stars)\b',
        r'\b(?:reduced|increased|boosted|saved|grew|cut)\b[^\n,.]*?\b\d+(?:\.\d+)?(?:%|x|\s*hours|\s*days|\s*dollars)?\b'
    ]
    
    findings = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            if isinstance(m, str):
                cleaned = m.strip()
                if cleaned and cleaned not in findings:
                    findings.append(cleaned)
    return findings


def detect_action_verbs_and_weaknesses(text: str) -> Tuple[List[str], List[str]]:
    """
    Identifies strong action verbs and weak phrases used.
    """
    text_lower = text.lower()
    
    found_power_verbs = []
    for verb in POWER_ACTION_VERBS:
        pattern = r'\b' + re.escape(verb) + r'\b'
        if re.search(pattern, text_lower):
            found_power_verbs.append(verb.title())
            
    found_weak_phrases = []
    for phrase in WEAK_PHRASES:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text_lower):
            found_weak_phrases.append(phrase)
            
    return found_power_verbs, found_weak_phrases


def calculate_ats_score(text: str, page_count: int = 1) -> Dict[str, Any]:
    """
    Calculates a comprehensive, transparent ATS score out of 100 with category breakdowns.
    """
    words = text.split()
    word_count = len(words)
    
    contacts = extract_contact_info(text)
    sections_data = extract_sections(text)
    detected_sections = sections_data["detected_sections"]
    skills_data = extract_skills_from_text(text)
    quant_metrics = detect_quantifiable_metrics(text)
    power_verbs, weak_phrases = detect_action_verbs_and_weaknesses(text)
    
    # 1. Section Structure (Max 20 pts)
    section_score = 0
    essential_sections = ["Experience", "Education", "Skills", "Summary"]
    for s in essential_sections:
        if s in detected_sections:
            section_score += 5
    if "Projects" in detected_sections or "Certifications" in detected_sections:
        section_score = min(20, section_score + 2)
        
    # 2. Contact Information (Max 15 pts)
    contact_score = 0
    if contacts.get("email"):
        contact_score += 5
    if contacts.get("phone"):
        contact_score += 5
    if contacts.get("linkedin") or contacts.get("github") or contacts.get("portfolio"):
        contact_score += 5
        
    # 3. Quantifiable Achievements / STAR (Max 20 pts)
    # At least 4 quantifiable metrics for full points
    metric_count = len(quant_metrics)
    metric_score = min(20, metric_count * 5)
    
    # 4. Action Verbs & Tone (Max 15 pts)
    # Rewards strong power verbs and penalizes weak passive phrases
    verb_count = len(power_verbs)
    weak_count = len(weak_phrases)
    base_verb_score = min(15, verb_count * 3)
    verb_score = max(0, base_verb_score - (weak_count * 3))
    
    # 5. Skills Breadth & Technical Keywords (Max 15 pts)
    total_skills_count = sum(len(skills) for skills in skills_data.values())
    skill_score = min(15, total_skills_count * 2)
    
    # 6. Length & Formatting (Max 15 pts)
    # Ideal CV length: 350 to 900 words, 1-2 pages
    length_score = 15
    if word_count < 250:
        length_score = 5
    elif word_count < 350:
        length_score = 10
    elif word_count > 1200:
        length_score = 8
    
    if page_count > 3:
        length_score = max(5, length_score - 5)
        
    # Total Score
    total_ats_score = section_score + contact_score + metric_score + verb_score + skill_score + length_score
    total_ats_score = min(100, max(0, total_ats_score))
    
    # Generate actionable checklist
    recommendations = []
    if section_score < 20:
        missing = [s for s in essential_sections if s not in detected_sections]
        recommendations.append(f"Add missing standard sections: **{', '.join(missing)}**.")
    if not contacts.get("linkedin"):
        recommendations.append("Include your **LinkedIn profile URL** in the header.")
    if metric_count < 3:
        recommendations.append("Quantify your achievements using numbers (e.g., *'Increased sales by 25%'*, *'Led team of 6'*).")
    if weak_phrases:
        recommendations.append(f"Replace passive phrases like *'{weak_phrases[0]}'* with strong impact verbs.")
    if word_count < 350:
        recommendations.append("Your CV is too brief. Elaborate on project deliverables and responsibilities.")
    elif word_count > 1000 and page_count > 2:
        recommendations.append("Consider trimming your CV down to 1-2 concise pages.")
    if total_skills_count < 6:
        recommendations.append("List more specific technical tools, frameworks, and methodologies.")

    return {
        "overall_score": total_ats_score,
        "breakdown": {
            "Sections Completeness": {"score": section_score, "max": 20},
            "Contact Details": {"score": contact_score, "max": 15},
            "Quantifiable Impact (STAR)": {"score": metric_score, "max": 20},
            "Action Verbs Strength": {"score": verb_score, "max": 15},
            "Skills Breadth": {"score": skill_score, "max": 15},
            "Length & Formatting": {"score": length_score, "max": 15}
        },
        "metrics": {
            "word_count": word_count,
            "page_count": page_count,
            "reading_time_min": max(1, round(word_count / 200, 1)),
            "quantifiable_metrics_count": metric_count,
            "power_verbs_count": verb_count,
            "weak_phrases_count": weak_count,
            "total_skills_count": total_skills_count
        },
        "contacts": contacts,
        "sections": sections_data,
        "skills": skills_data,
        "quant_metrics": quant_metrics,
        "power_verbs": power_verbs,
        "weak_phrases": weak_phrases,
        "recommendations": recommendations
    }


def match_cv_to_job_description(cv_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Computes semantic and keyword overlap between CV and Job Description.
    """
    if not jd_text.strip() or not cv_text.strip():
        return {
            "match_score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "similarity_percent": 0
        }
        
    # TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
    try:
        tfidf_matrix = vectorizer.fit_transform([cv_text, jd_text])
        similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        sim_percentage = round(similarity * 100, 1)
    except Exception:
        sim_percentage = 0.0

    # Skills comparison
    cv_skills_dict = extract_skills_from_text(cv_text)
    jd_skills_dict = extract_skills_from_text(jd_text)
    
    cv_all_skills = set(s.lower() for skills in cv_skills_dict.values() for s in skills)
    jd_all_skills = set(s.lower() for skills in jd_skills_dict.values() for s in skills)
    
    matching_skills = [s.title() for s in jd_all_skills if s in cv_all_skills]
    missing_skills = [s.title() for s in jd_all_skills if s not in cv_all_skills]
    
    # Combined Job Fit Score
    if jd_all_skills:
        skill_coverage_ratio = len(matching_skills) / len(jd_all_skills)
        combined_score = round((sim_percentage * 0.4) + (skill_coverage_ratio * 100 * 0.6))
    else:
        combined_score = round(sim_percentage)
        
    combined_score = min(100, max(0, combined_score))

    return {
        "match_score": combined_score,
        "similarity_percent": sim_percentage,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "jd_skills_total": len(jd_all_skills),
        "cv_skills_total": len(cv_all_skills)
    }


def evaluate_candidate_custom_criteria(cv_text: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a candidate's CV against explicit employer/recruiter criteria:
    - Mandatory & Preferred Skills
    - Required Certifications
    - Required Trainings & Internships
    - Min Experience Years
    - Education / Degree
    Provides a separate, deep audit for each dimension.
    """
    sections_data = extract_sections(cv_text)
    detected_sections = sections_data["detected_sections"]
    cv_lower = cv_text.lower()

    # 1. EXTRACT DATA DIMENSIONS
    certs_found = extract_certifications_list(cv_text, detected_sections)
    trainings_found = extract_trainings_and_internships(cv_text, detected_sections)
    exp_data = estimate_years_of_experience(cv_text)
    edu_data = extract_education_details(cv_text, detected_sections)
    cv_skills_dict = extract_skills_from_text(cv_text)
    cv_all_skills_flat = set(s.lower() for cat in cv_skills_dict.values() for s in cat)

    # 2. EVALUATE SKILLS
    mandatory_reqs = [s.strip() for s in criteria.get("mandatory_skills", []) if s.strip()]
    preferred_reqs = [s.strip() for s in criteria.get("preferred_skills", []) if s.strip()]

    matched_mandatory = []
    missing_mandatory = []
    for req in mandatory_reqs:
        req_clean = req.lower()
        if req_clean in cv_all_skills_flat or re.search(r'\b' + re.escape(req_clean) + r'\b', cv_lower):
            matched_mandatory.append(req)
        else:
            missing_mandatory.append(req)

    matched_preferred = []
    missing_preferred = []
    for req in preferred_reqs:
        req_clean = req.lower()
        if req_clean in cv_all_skills_flat or re.search(r'\b' + re.escape(req_clean) + r'\b', cv_lower):
            matched_preferred.append(req)
        else:
            missing_preferred.append(req)

    total_req_skills = len(mandatory_reqs) + len(preferred_reqs)
    total_matched_skills = len(matched_mandatory) + len(matched_preferred)
    skills_pct = round((total_matched_skills / total_req_skills * 100)) if total_req_skills > 0 else 100

    skills_explanation = ""
    if missing_mandatory:
        skills_explanation = f"المرشح يفتقر إلى {len(missing_mandatory)} مهارات إلزامية هامة: ({', '.join(missing_mandatory)})."
    elif mandatory_reqs:
        skills_explanation = f"المرشح استوفى جميع المهارات الإلزامية المطلوبة بنسبة 100% ({', '.join(matched_mandatory)})."
    else:
        skills_explanation = f"تم رصد {len(cv_all_skills_flat)} مهارة متنوعة في السيرة الذاتية."

    # 3. EVALUATE CERTIFICATIONS
    required_certs = [c.strip() for c in criteria.get("required_certifications", []) if c.strip()]
    matched_certs = []
    missing_certs = []
    for req_c in required_certs:
        req_c_lower = req_c.lower()
        if any(req_c_lower in f.lower() for f in certs_found) or req_c_lower in cv_lower:
            matched_certs.append(req_c)
        else:
            missing_certs.append(req_c)

    certs_status = "غير محدد"
    if required_certs:
        if len(matched_certs) == len(required_certs):
            certs_status = "مستوفى بالكامل (100%)"
            certs_explanation = f"المرشح حاصل على جميع الشهادات المطلوبة: {', '.join(matched_certs)}."
        elif matched_certs:
            certs_status = "مستوفى جزئياً"
            certs_explanation = f"المرشح حاصل على ({', '.join(matched_certs)})، لكن ينقصه ({', '.join(missing_certs)})."
        else:
            certs_status = "غير متوفرة"
            certs_explanation = f"المرشح لا يمتلك أي من الشهادات المحددة ({', '.join(missing_certs)})."
    else:
        certs_status = "اختياري"
        certs_explanation = f"تم رصد {len(certs_found)} شهادات معتمدة في السيرة الذاتية." if certs_found else "لم تُحدد شهادات كشرط إلزامي، ولم يذكر المرشح شهادات رسمية."

    # 4. EVALUATE TRAININGS & INTERNSHIPS
    required_trainings = [t.strip() for t in criteria.get("required_trainings", []) if t.strip()]
    matched_trainings = []
    missing_trainings = []
    for req_t in required_trainings:
        req_t_lower = req_t.lower()
        if any(req_t_lower in tf["title"].lower() or req_t_lower in tf["details"].lower() for tf in trainings_found) or req_t_lower in cv_lower:
            matched_trainings.append(req_t)
        else:
            missing_trainings.append(req_t)

    trainings_status = "غير محدد"
    if required_trainings:
        if len(matched_trainings) == len(required_trainings):
            trainings_status = "مستوفى بالكامل"
            trainings_explanation = f"المرشح خاض جميع التدريبات والبرامج المطلوبة بنجاح ({', '.join(matched_trainings)})."
        elif matched_trainings:
            trainings_status = "مستوفى جزئياً"
            trainings_explanation = f"المرشح خاض ({', '.join(matched_trainings)})، ولكن لم يُعثر على ({', '.join(missing_trainings)})."
        else:
            trainings_status = "غير مستوفى"
            trainings_explanation = f"لم يتم العثور على البرامج التدريبية المحددة في سيرة المرشح ({', '.join(missing_trainings)})."
    else:
        trainings_status = "مكتشف"
        trainings_explanation = f"تم رصد {len(trainings_found)} فترات تدريب/برامج تدريبية في السيرة الذاتية." if trainings_found else "لم يذكر المرشح برامج تدريبية أو تدريب صيفي صريح."

    # 5. EVALUATE EXPERIENCE
    min_years = criteria.get("min_years_experience", 0)
    candidate_years = exp_data["estimated_years"]
    meets_experience = candidate_years >= min_years if min_years > 0 else True
    if min_years > 0:
        if meets_experience:
            exp_explanation = f"يستوفي شرط الخبرة: المرشح لديه حوالي {candidate_years} سنوات خبرة (المطلوب {min_years}+ سنوات)."
        else:
            exp_explanation = f"أقل من الخبرة المطلوبة: المرشح لديه حوالي {candidate_years} سنوات خبرة بينما المطلوب {min_years}+ سنوات."
    else:
        exp_explanation = f"تم تقدير خبرة المرشح بـ {candidate_years} سنوات."

    # 6. EVALUATE EDUCATION
    req_degree = criteria.get("required_degree", "").strip()
    meets_education = True
    if req_degree:
        meets_education = any(req_degree.lower() in d.lower() for d in edu_data["degrees"]) or req_degree.lower() in cv_lower
        if meets_education:
            edu_explanation = f"مستوفى: المرشح يحمل مؤهل مطابق ({req_degree})."
        else:
            edu_explanation = f"المؤهل المطلوب ({req_degree}) غير مذكور بوضوح. المؤهلات المرصودة: {', '.join(edu_data['degrees']) if edu_data['degrees'] else 'غير محددة'}."
    else:
        edu_explanation = f"المؤهلات المرصودة: {', '.join(edu_data['degrees']) if edu_data['degrees'] else 'مؤهل جامعي'}"

    # 7. OVERALL VERDICT SCORE
    # Weight distribution: Mandatory Skills (40%), Experience (20%), Certifications (15%), Trainings (15%), Education (10%)
    verdict_score = 0
    
    # Skills contribution (40 pts)
    if mandatory_reqs:
        verdict_score += (len(matched_mandatory) / len(mandatory_reqs)) * 40
    else:
        verdict_score += min(40, len(cv_all_skills_flat) * 4)

    # Experience contribution (20 pts)
    if min_years > 0:
        if candidate_years >= min_years:
            verdict_score += 20
        elif candidate_years > 0:
            verdict_score += (candidate_years / min_years) * 20
    else:
        verdict_score += 20

    # Certifications contribution (15 pts)
    if required_certs:
        verdict_score += (len(matched_certs) / len(required_certs)) * 15
    else:
        verdict_score += 15 if certs_found else 10

    # Trainings contribution (15 pts)
    if required_trainings:
        verdict_score += (len(matched_trainings) / len(required_trainings)) * 15
    else:
        verdict_score += 15 if trainings_found else 10

    # Education contribution (10 pts)
    if req_degree:
        verdict_score += 10 if meets_education else 0
    else:
        verdict_score += 10

    verdict_score = round(min(100, max(0, verdict_score)))

    # Decision Badge
    if missing_mandatory or (min_years > 0 and candidate_years < min_years * 0.7):
        verdict_status = "غير مطابق لبعض الشروط الأساسية (Does Not Meet Key Criteria)"
        verdict_badge = "red"
        decision_label = "غير مطابق (Not Matched)"
    elif verdict_score >= 80:
        verdict_status = "مرشح متميز ومطابق تماماً للمتطلبات (Highly Recommended)"
        verdict_badge = "green"
        decision_label = "مؤهل للمقابلة (Recommended)"
    else:
        verdict_status = "مرشح مؤهل جزئياً مع بعض الملاحظات (Potential Match)"
        verdict_badge = "amber"
        decision_label = "مؤهل جزئياً (Review Needed)"

    return {
        "verdict_score": verdict_score,
        "verdict_status": verdict_status,
        "verdict_badge": verdict_badge,
        "decision_label": decision_label,
        "skills_audit": {
            "score_pct": skills_pct,
            "matched_mandatory": matched_mandatory,
            "missing_mandatory": missing_mandatory,
            "matched_preferred": matched_preferred,
            "missing_preferred": missing_preferred,
            "explanation": skills_explanation
        },
        "certifications_audit": {
            "status": certs_status,
            "matched": matched_certs,
            "missing": missing_certs,
            "all_detected_in_cv": certs_found,
            "explanation": certs_explanation
        },
        "trainings_audit": {
            "status": trainings_status,
            "matched": matched_trainings,
            "missing": missing_trainings,
            "all_detected_in_cv": trainings_found,
            "explanation": trainings_explanation
        },
        "experience_audit": {
            "meets_experience": meets_experience,
            "candidate_years": candidate_years,
            "required_years": min_years,
            "explanation": exp_explanation
        },
        "education_audit": {
            "meets_education": meets_education,
            "detected_degrees": edu_data["degrees"],
            "explanation": edu_explanation
        }
    }

