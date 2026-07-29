"""schemas.py — Request/response contracts, validated automatically by FastAPI/Pydantic."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ParseResumeRequest(BaseModel):
    resume_text: str = Field(..., min_length=1, description="Raw resume text to parse")

    @field_validator("resume_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("resume_text must not be blank")
        return v


class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


class Experience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None


class Project(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ParsedResume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = []
    education: list[Education] = []
    experience: list[Experience] = []
    certifications: list[str] = []
    projects: list[Project] = []
    links: list[str] = []


class ParseResumeResponse(BaseModel):
    success: bool
    data: Optional[ParsedResume] = None
    error: Optional[str] = None
    raw_model_output: Optional[str] = None  # useful for debugging schema-validation failures
