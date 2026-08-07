"""
Domain model describing the fields we extract from a government tender
document. This is the single source of truth for:
  - the JSON schema we ask Azure OpenAI to fill in
  - validation of the LLM's response
  - the rows written into the output Excel file
"""

from typing import ClassVar, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class TenderData(BaseModel):
    tender_number: Optional[str] = Field(None, description="Tender / bid number")
    tender_title: Optional[str] = Field(None, description="Title of the tender")
    organization: Optional[str] = Field(None, description="Issuing organization")
    department: Optional[str] = Field(None, description="Issuing department")
    tender_value: Optional[str] = Field(None, description="Estimated tender value")
    emd_amount: Optional[str] = Field(None, description="Earnest Money Deposit amount")
    bid_submission_date: Optional[str] = Field(None, description="Bid / closing date")
    bid_opening_date: Optional[str] = Field(None, description="Bid opening date")
    eligibility: Optional[str] = Field(None, description="Eligibility criteria")
    experience_required: Optional[str] = Field(None, description="Experience required")
    required_certificates: Optional[str] = Field(None, description="Required certificates")
    contact_person: Optional[str] = Field(None, description="Contact person name")
    email: Optional[str] = Field(None, description="Contact email")
    phone: Optional[str] = Field(None, description="Contact phone number")
    location: Optional[str] = Field(None, description="Work / delivery location")
    work_description: Optional[str] = Field(None, description="Scope of work")
    duration: Optional[str] = Field(None, description="Contract duration")

    model_config = ConfigDict(extra="ignore")  # tolerate stray keys the LLM might add

    # Field order used for the Excel sheet and prompt, kept in one place
    # so the prompt and the spreadsheet never drift apart.
    FIELD_LABELS: ClassVar[Dict[str, str]] = {
        "tender_number": "Tender Number",
        "tender_title": "Tender Title",
        "organization": "Organization",
        "department": "Department",
        "tender_value": "Tender Value",
        "emd_amount": "EMD Amount",
        "bid_submission_date": "Bid Submission Date",
        "bid_opening_date": "Bid Opening Date",
        "eligibility": "Eligibility",
        "experience_required": "Experience Required",
        "required_certificates": "Required Certificates",
        "contact_person": "Contact Person",
        "email": "Email",
        "phone": "Phone",
        "location": "Location",
        "work_description": "Work Description",
        "duration": "Duration",
    }
