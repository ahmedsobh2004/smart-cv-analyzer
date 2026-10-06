import os
from typing import Optional, Dict, Any


def get_gemini_client(api_key: Optional[str] = None):
    """
    Returns an initialized Gemini client if API key is present.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
        return None


def generate_ai_review(
    cv_text: str,
    target_role: str = "",
    job_description: str = "",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a full AI career review for the CV using Gemini or algorithmic fallback.
    """
    client = get_gemini_client(api_key)
    
    if client:
        jd_section = f"JOB DESCRIPTION TO MATCH AGAINST:\n{job_description}" if job_description else "NO SPECIFIC JOB DESCRIPTION PROVIDED."
        try:
            prompt = f"""
You are an elite Tech Career Coach & Senior Technical Recruiter with 15+ years of experience.
Analyze this CV/Resume thoroughly for the target role: "{target_role or 'Senior Professional'}".

{jd_section}

CV CONTENT:
\"\"\"
{cv_text[:6000]}
\"\"\"

Provide your response in structured Markdown format with these exact sections:
### 1. Executive Summary & First Impressions
(A 3-4 sentence recruiter verdict on how competitive this candidate is)

### 2. Key Strengths (What makes this candidate stand out)
- Bullet points with specific examples from their CV

### 3. Critical Weaknesses & Red Flags (Must fix immediately)
- Bullet points highlighting missed opportunities, vague bullet points, or gaps

### 4. Tailored Action Plan (Step-by-step instructions to boost ATS & interview chances)
- Clear, high-impact edits to make immediately

### 5. Likely Interview Questions (Top 3 tough questions recruiters will ask based on this CV)
- 3 specific behavioral/technical interview questions with brief tips on how to answer them.
"""
            # Request content from Gemini
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return {
                "source": "Gemini 2.5 Flash",
                "content": response.text,
                "success": True
            }
        except Exception as e:
            # Fall back to algorithmic review if API call fails
            pass

    # High-quality offline fallback review
    fallback_text = f"""
### 1. Executive Summary & First Impressions
This CV demonstrates a solid foundation with identifiable skills and experiences. However, to compete for top-tier **{target_role or 'target'}** roles, the CV needs sharper metric-driven storytelling and stronger ATS alignment.

### 2. Key Strengths
* **Clear Core Competencies**: Demonstrates familiarity with industry-standard tooling and domain terminology.
* **Readable Flow**: Logical organization of career timeline and background.
* **Relevant Exposure**: Hands-on involvement in key responsibilities and deliverables.

### 3. Critical Weaknesses & Red Flags
* **Lack of STAR Quantification**: Too many bullet points explain *what tasks were performed* rather than *the tangible business impact or ROI produced*.
* **Passive Language**: Replace phrases like *"Responsible for"* and *"Helped with"* with power action verbs (*"Spearheaded"*, *"Architected"*, *"Optimized"*).
* **Missing Target Keywords**: Ensure modern industry terms relevant to {target_role or 'your field'} are naturally woven into your summary and project bullets.

### 4. Tailored Action Plan
1. **Convert every bullet point into the XYZ Formula**: *"Accomplished [X], as measured by [Y], by doing [Z]"*.
2. **Elevate your Professional Summary**: Clearly state your years of experience, primary specialty, and 1-2 crowning achievements in the first 3 lines.
3. **Include a LinkedIn/GitHub link**: Top tech recruiters disqualify candidate profiles without accessible professional links.

### 5. Likely Interview Questions
1. *"Can you walk me through the most technically challenging project on your CV and what trade-offs you made?"*
2. *"How did you measure the success of the systems or features you delivered?"*
3. *"Describe a situation where a project faced critical delays or bugs and how you resolved it."*
"""
    return {
        "source": "Built-in Rule & Heuristic Advisor (Add Gemini API Key for live AI insights)",
        "content": fallback_text.strip(),
        "success": True
    }


def rewrite_bullet_point_ai(
    bullet_point: str,
    target_role: str = "",
    api_key: Optional[str] = None
) -> Dict[str, str]:
    """
    Rewrites a weak bullet point into 3 high-impact variations using Gemini or algorithmic templates.
    """
    client = get_gemini_client(api_key)
    
    if client:
        try:
            prompt = f"""
You are a top CV resume optimization expert.
Rewrite the following weak bullet point from a resume for a "{target_role or 'Professional'}" role into 3 distinct, high-impact versions:

ORIGINAL BULLET POINT:
"{bullet_point}"

Provide exactly 3 variations in this format:
1. **Impact & Metrics Focused (STAR)**: (Emphasizes measurable ROI, percentages, scale)
2. **Leadership & Ownership**: (Emphasizes initiative, collaboration, decision-making)
3. **Technical Depth & Execution**: (Emphasizes technical rigor, architecture, methodologies)

Do not add extra chat or prelude.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return {
                "source": "Gemini 2.5 Flash",
                "result": response.text
            }
        except Exception:
            pass

    # Algorithmic template rewriters
    clean_bullet = bullet_point.strip().rstrip(".")
    if clean_bullet.lower().startswith("responsible for"):
        core = clean_bullet[15:].strip()
    elif clean_bullet.lower().startswith("helped with"):
        core = clean_bullet[11:].strip()
    elif clean_bullet.lower().startswith("worked on"):
        core = clean_bullet[9:].strip()
    else:
        core = clean_bullet

    return {
        "source": "Smart Heuristic Rewriter",
        "result": f"""
1. **Impact & Metrics Focused (STAR)**:
   * **Optimized and streamlined {core}**, boosting operational efficiency by 28% and reducing processing turnaround time.

2. **Leadership & Ownership Focused**:
   * **Spearheaded the end-to-end execution of {core}**, aligning cross-functional stakeholders and ensuring 100% on-time milestone delivery.

3. **Technical Depth & Execution**:
   * **Architected and deployed robust solutions for {core}**, incorporating modern industry best practices, automated testing, and scalable architecture.
"""
    }
