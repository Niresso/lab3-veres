from typing import Literal
from pydantic import BaseModel, Field


class SpecOutput(BaseModel):
    title: str = Field(description="Short title of the task")
    requirements: list[str] = Field(description="Functional requirements for the implementation")
    acceptance_criteria: list[str] = Field(description="Specific, testable conditions for acceptance")
    tech_stack: list[str] = Field(description="Libraries and technologies to use")
    estimated_complexity: Literal["simple", "medium", "complex"] = Field(
        description="Estimated implementation complexity"
    )
    coding_standards: list[str] = Field(
        description="Coding conventions and standards to follow",
        default_factory=list,
    )


class CodeOutput(BaseModel):
    description: str = Field(description="Brief description of what was implemented")
    files_created: list[str] = Field(description="Relative paths of all files written to disk")
    source_code: str = Field(description="Main source code (entry point), full content")
    test_results: str = Field(description="Output from python_repl when the code was tested")


class ReviewOutput(BaseModel):
    verdict: Literal["APPROVED", "REVISION_NEEDED"]
    score: float = Field(ge=0.0, le=1.0, description="Quality score 0.0–1.0")
    issues: list[str] = Field(description="Specific problems found (file:line if possible)")
    suggestions: list[str] = Field(description="Concrete improvement suggestions")
    acceptance_criteria_met: list[str] = Field(description="Criteria that passed")
    acceptance_criteria_failed: list[str] = Field(description="Criteria that did not pass")
