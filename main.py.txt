from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os, json, re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="ATS Resume & JD Match Engine")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeScore(BaseModel):
    skills_score: int = Field(..., example=40)
    experience_score: int = Field(..., example=30)
    keywords_score: int = Field(..., example=20)
    formatting_score: int = Field(..., example=10)
    total: int = Field(..., example=100)


class JDMatchScore(BaseModel):
    skill_match: int = Field(..., example=40)
    experience_match: int = Field(..., example=40)
    keyword_match: int = Field(..., example=20)
    overall_match: int = Field(..., example=100)


class ATSResponse(BaseModel):
    resume_score: ResumeScore
    job_description_match_score: JDMatchScore
    skills_detected: list[str]
    skills_matched: list[str]
    skills_missing: list[str]
    message: str


class ATSRequest(BaseModel):
    resume: str
    job_description: str


PROMPT = """
Return ONLY this JSON structure:

{{
 "resume_score": {{
    "skills_score": 0,
    "experience_score": 0,
    "keywords_score": 0,
    "formatting_score": 0,
    "total": 0
 }},
 "job_description_match_score": {{
    "skill_match": 0,
    "experience_match": 0,
    "keyword_match": 0,
    "overall_match": 0
 }},
 "skills_detected": [],
 "skills_matched": [],
 "skills_missing": [],
 "message": ""
}}

Resume:
{resume}

Job Description:
{jd}
"""


def extract_json(text):
    match = re.search(r"(\{[\s\S]*\})", text)
    if not match:
        raise ValueError("JSON extraction failed")
    return json.loads(match.group(1))


@app.post("/ats", response_model=ATSResponse)
async def ats(req: ATSRequest):
    try:
        prompt = PROMPT.format(resume=req.resume, jd=req.job_description)

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800
        )

        raw = result.choices[0].message.content
        data = extract_json(raw)

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
