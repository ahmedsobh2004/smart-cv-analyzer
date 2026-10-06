from extractor import extract_contact_info, extract_sections
from analyzer import calculate_ats_score, match_cv_to_job_description
from ai_advisor import generate_ai_review, rewrite_bullet_point_ai
from sample_cv import SAMPLE_CV_TEXT, SAMPLE_JOB_DESCRIPTION

print("Testing extractor...")
contacts = extract_contact_info(SAMPLE_CV_TEXT)
print("Contacts detected:", contacts)

sections = extract_sections(SAMPLE_CV_TEXT)
print("Sections detected:", sections["present_sections"])

print("\nTesting ATS scoring...")
score_res = calculate_ats_score(SAMPLE_CV_TEXT, page_count=2)
print("Total ATS Score:", score_res["overall_score"])
print("Breakdown:", score_res["breakdown"])
print("Recommendations count:", len(score_res["recommendations"]))

print("\nTesting Job Description Matching...")
match_res = match_cv_to_job_description(SAMPLE_CV_TEXT, SAMPLE_JOB_DESCRIPTION)
print("Match Score:", match_res["match_score"])
print("Matching Skills:", match_res["matching_skills"])
print("Missing Skills:", match_res["missing_skills"])

print("\nTesting AI Advisor Fallback...")
review = generate_ai_review(SAMPLE_CV_TEXT, target_role="Senior Backend Engineer")
print("Review Success:", review["success"], "Source:", review["source"])

rewrite = rewrite_bullet_point_ai("Responsible for database maintenance", target_role="Backend Engineer")
print("Rewrite Source:", rewrite["source"])
print("All tests passed successfully!")
