import re
import io
from typing import Dict, List, Tuple, Any
from pypdf import PdfReader
from docx import Document


def extract_text_from_pdf(file_input: Any) -> Tuple[str, int]:
    """
    Extracts plain text and page count from a PDF file or BytesIO stream.
    """
    try:
        if isinstance(file_input, (str, bytes)):
            reader = PdfReader(file_input)
        elif hasattr(file_input, "read"):
            stream = io.BytesIO(file_input.read()) if not isinstance(file_input, io.BytesIO) else file_input
            reader = PdfReader(stream)
        else:
            reader = PdfReader(file_input)

        page_count = len(reader.pages)
        full_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

        return "\n".join(full_text), page_count

    except Exception as e:
        return f"Error extracting PDF: {str(e)}", 0


def extract_text_from_docx(file_input: Any) -> Tuple[str, int]:
    """
    Extracts text from a DOCX file or BytesIO stream.
    Estimates page count based on word count (~400 words/page).
    """
    try:
        if hasattr(file_input, "read"):
            stream = io.BytesIO(file_input.read()) if not isinstance(file_input, io.BytesIO) else file_input
            doc = Document(stream)
        else:
            doc = Document(file_input)

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_text = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if row_text:
                    paragraphs.append(" | ".join(row_text))

        full_text = "\n".join(paragraphs)

        word_count = len(full_text.split())
        estimated_pages = max(1, round(word_count / 400))

        return full_text, estimated_pages

    except Exception as e:
        return f"Error extracting DOCX: {str(e)}", 0


def extract_contact_info(text: str) -> Dict[str, Any]:
    """
    Finds email, phone, LinkedIn, GitHub, and portfolio URLs from CV text.
    """

    contacts = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "portfolio": None
    }

    # Email
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_pattern, text)

    if emails:
        contacts["email"] = emails[0]

    # Phone numbers (international, Egyptian, US, etc.)
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
    phones = re.findall(phone_pattern, text)

    # Match full numbers cleanly
    full_phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}'
    matched_phones = re.findall(full_phone_pattern, text)

    valid_phones = [
        p.strip()
        for p in matched_phones
        if sum(c.isdigit() for c in p) >= 8
    ]

    if valid_phones:
        contacts["phone"] = valid_phones[0]

    # LinkedIn
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+'
    linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)

    if linkedin_match:
        contacts["linkedin"] = linkedin_match.group(0)

    # GitHub
    github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+'
    github_match = re.search(github_pattern, text, re.IGNORECASE)

    if github_match:
        contacts["github"] = github_match.group(0)

    # Portfolio / Website
    portfolio_pattern = r'https?://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
    urls = re.findall(portfolio_pattern, text)

    other_urls = [
        u for u in urls
        if "linkedin.com" not in u and "github.com" not in u
    ]

    if other_urls:
        contacts["portfolio"] = other_urls[0]

    return contacts


SECTION_PATTERNS = {
    "Summary": [
        r'\b(?:professional\s+summary|executive\s+summary|summary|profile|about\s+me|career\s+objective|objective)\b'
    ],

    "Experience": [
        r'\b(?:work\s+experience|professional\s+experience|experience|employment\s+history|career\s+history|work\s+history)\b'
    ],

    "Education": [
        r'\b(?:education|academic\s+background|educational\s+background|qualifications|academic\s+qualifications)\b'
    ],

    "Skills": [
        r'\b(?:skills|technical\s+skills|core\s+competencies|key\s+skills|technologies|tools\s+&\s+technologies)\b'
    ],

    "Projects": [
        r'\b(?:projects|key\s+projects|personal\s+projects|selected\s+projects|portfolio\s+projects)\b'
    ],

    "Certifications": [
        r'\b(?:certifications|certificates|licenses|accreditations|courses|professional\s+certifications)\b'
    ],

    "Languages": [
        r'\b(?:languages|language\s+proficiency)\b'
    ],

    "Awards": [
        r'\b(?:awards|honors|achievements|recognition)\b'
    ]
}


def extract_sections(text: str) -> Dict[str, Any]:
    """
    Detects standard CV sections and splits text by sections.
    """

    detected_sections = {}
    lines = text.splitlines()

    # Identify positions of section headers
    section_positions = []

    for idx, line in enumerate(lines):

        clean_line = line.strip().lower()

        if not clean_line or len(clean_line) > 40:
            continue

        for sec_name, patterns in SECTION_PATTERNS.items():

            for pat in patterns:

                if (
                    re.fullmatch(pat, clean_line, re.IGNORECASE)
                    or (
                        clean_line.startswith(
                            re.findall(r'[a-z]+', pat)[0]
                        )
                        and len(clean_line.split()) <= 4
                    )
                ):
                    section_positions.append(
                        (idx, sec_name, line.strip())
                    )
                    break

    # Sort positions by line index
    section_positions.sort(key=lambda x: x[0])

    # Remove duplicate consecutive section types
    filtered_positions = []
    seen_sections = set()

    for pos, sec_name, header in section_positions:

        if sec_name not in seen_sections:
            seen_sections.add(sec_name)
            filtered_positions.append(
                (pos, sec_name, header)
            )

    # Slice text per section
    for i in range(len(filtered_positions)):

        start_line = filtered_positions[i][0] + 1
        sec_name = filtered_positions[i][1]

        if i + 1 < len(filtered_positions):
            end_line = filtered_positions[i + 1][0]

            section_content = "\n".join(
                lines[start_line:end_line]
            ).strip()

        else:
            section_content = "\n".join(
                lines[start_line:]
            ).strip()

        detected_sections[sec_name] = section_content

    return {
        "detected_sections": detected_sections,
        "present_sections": list(detected_sections.keys()),
        "missing_essential_sections": [
            s
            for s in ["Summary", "Experience", "Education", "Skills"]
            if s not in detected_sections
        ]
    }


def extract_certifications_list(
    text: str,
    sections_dict: Dict[str, str] = None
) -> List[str]:
    """
    Extracts individual certifications from the CV text.
    """

    certs = []

    # 1. From the Certifications section if available
    cert_text = ""

    if sections_dict and "Certifications" in sections_dict:
        cert_text = sections_dict["Certifications"]

    if cert_text:

        for line in cert_text.splitlines():

            clean = line.strip(" -*•\t\r\n")

            if (
                clean
                and len(clean) > 3
                and not clean.lower().startswith("certif")
            ):
                certs.append(clean)

    # 2. Regex search across whole text for well-known industry cert patterns
    common_cert_patterns = [

        r'\b(?:AWS\s+Certified[^\n,\.]{3,40})\b',

        r'\b(?:Azure\s+(?:Fundamentals|Administrator|Developer|Solutions\s+Architect)[^\n,\.]{0,30})\b',

        r'\b(?:GCP\s+Professional[^\n,\.]{3,35})\b',

        r'\b(?:Google\s+(?:Cloud\s+Certified|Professional)[^\n,\.]{3,35})\b',

        r'\b(?:Cisco\s+(?:CCNA|CCNP|CCIE)[^\n,\.]{0,25})\b',

        r'\b(?:PMP|CAPM|PRINCE2|Scrum\s+Master|PSM\s+I|CSM|CKA|CKAD|CISM|CISSP)\b',

        r'\b(?:Meta\s+Professional[^\n,\.]{3,35})\b',

        r'\b(?:Certified\s+[A-Za-z0-9\s]{4,35}(?:Associate|Professional|Expert|Specialist)?)\b'
    ]

    for pattern in common_cert_patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        for m in matches:

            m_clean = m.strip()

            if (
                m_clean
                and not any(
                    m_clean.lower() in c.lower()
                    for c in certs
                )
            ):
                certs.append(m_clean)

    return list(dict.fromkeys(certs))


def extract_trainings_and_internships(
    text: str,
    sections_dict: Dict[str, str] = None
) -> List[Dict[str, str]]:
    """
    Detects internships, summer trainings, bootcamps, and vocational programs.
    """

    items = []
    lines = text.splitlines()

    internship_keywords = [
        "intern",
        "internship",
        "trainee",
        "summer training",
        "bootcamp",
        "iti",
        "nti",
        "nanodegree",
        "fellowship",
        "apprentice"
    ]

    for idx, line in enumerate(lines):

        line_lower = line.lower()

        if any(
            kw in line_lower
            for kw in internship_keywords
        ):

            # Grab this line and the following context line if relevant
            title = line.strip(" -*•\t\r\n")
            context = ""

            if (
                idx + 1 < len(lines)
                and lines[idx + 1].strip()
                and not lines[idx + 1].strip().startswith("#")
            ):
                context = lines[idx + 1].strip()

            if len(title) > 3 and len(title) < 120:

                items.append({
                    "title": title,
                    "details": context
                })

    # Deduplicate by title
    seen = set()
    unique_items = []

    for it in items:

        if it["title"].lower() not in seen:

            seen.add(it["title"].lower())
            unique_items.append(it)

    return unique_items


def estimate_years_of_experience(
    text: str
) -> Dict[str, Any]:
    """
    Estimates total years of professional experience using year ranges and stated numbers.
    """

    import datetime

    current_year = datetime.datetime.now().year

    # 1. Look for explicit statement: "X+ years of experience"
    explicit_pat = (
        r'(\d+)\+?\s*(?:years?|yrs?)'
        r'(?:\s+of)?\s+experience'
    )

    explicit_match = re.search(
        explicit_pat,
        text,
        re.IGNORECASE
    )

    stated_years = (
        int(explicit_match.group(1))
        if explicit_match
        else None
    )

    # 2. Extract year spans
    # e.g., 2019 - 2023, 2021 - Present
    year_range_pat = (
        r'\b(20[0-2][0-9]|199[0-9])'
        r'\s*[-–—to]+'
        r'(20[0-2][0-9]|present|current|now)\b'
    )

    matches = re.findall(
        year_range_pat,
        text,
        re.IGNORECASE
    )

    calculated_years = 0
    intervals = []

    for start_str, end_str in matches:

        start_y = int(start_str)

        end_y = (
            current_year
            if end_str.lower()
            in ["present", "current", "now"]
            else int(end_str)
        )

        if 0 <= (end_y - start_y) <= 30:
            intervals.append(
                (start_y, end_y)
            )

    if intervals:

        # Merge overlapping intervals
        intervals.sort(key=lambda x: x[0])

        merged = []

        for start, end in intervals:

            if not merged or merged[-1][1] < start:
                merged.append([start, end])

            else:
                merged[-1][1] = max(
                    merged[-1][1],
                    end
                )

        calculated_years = sum(
            end - start
            for start, end in merged
        )

    final_years = (
        stated_years
        if stated_years is not None
        else max(calculated_years, 0)
    )

    return {
        "estimated_years": final_years,
        "stated_years": stated_years,
        "calculated_from_spans": calculated_years
    }


def extract_education_details(
    text: str,
    sections_dict: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Extracts degrees, institutions, and majors from CV.
    """

    edu_text = ""

    if sections_dict and "Education" in sections_dict:
        edu_text = sections_dict["Education"]
    else:
        edu_text = text

    degrees_found = []

    degree_keywords = [
        "bachelor",
        "b.sc",
        "bsc",
        "b.s",
        "master",
        "m.sc",
        "msc",
        "m.s",
        "phd",
        "doctorate",
        "diploma",
        "associate degree",
        "بكالوريوس",
        "ماجستير",
        "دكتوراه"
    ]

    for d in degree_keywords:

        if re.search(
            r'\b' + re.escape(d) + r'\b',
            edu_text,
            re.IGNORECASE
        ):
            degrees_found.append(d.title())

    return {
        "degrees": list(set(degrees_found)),
        "raw_education": edu_text[:600] if edu_text else ""
    }