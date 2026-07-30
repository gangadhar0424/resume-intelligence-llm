"""schemas.py — Request/response contracts, validated automatically by FastAPI/Pydantic."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


def _coerce_scalar_to_str(v):
    """Small models are inconsistent about quoting numeric-looking values (e.g. emitting
    year: 2021 or duration: 3 instead of "2021"/"3") — coerce int/float to str before
    validation so a merely-unquoted number doesn't fail schema validation outright."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return v


def _coerce_none_to_list(v):
    """Small models sometimes emit null for an empty list field instead of omitting it
    or using [] — coerce None to [] before validation so it doesn't fail as a type error."""
    if v is None:
        return []
    return v


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

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v):
        return _coerce_scalar_to_str(v)


class Experience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None

    @field_validator("duration", mode="before")
    @classmethod
    def coerce_duration(cls, v):
        return _coerce_scalar_to_str(v)


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

    @field_validator("phone", mode="before")
    @classmethod
    def coerce_phone(cls, v):
        return _coerce_scalar_to_str(v)

    @field_validator("skills", "education", "experience", "certifications", "projects", "links",
                     mode="before")
    @classmethod
    def coerce_none_lists(cls, v):
        return _coerce_none_to_list(v)


class ParseResumeResponse(BaseModel):
    success: bool
    data: Optional[ParsedResume] = None
    error: Optional[str] = None
    raw_model_output: Optional[str] = None  # useful for debugging schema-validation failures
