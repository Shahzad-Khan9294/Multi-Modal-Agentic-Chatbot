from enum import Enum
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict, Dict


class EmbedInput(BaseModel):
    """Basic user input for the agent."""
    resource_type: str = Field(
        ...,
        description="Resource type to process for embedding.",
        example= "document"
    )
    resources: list[str] = Field(
        ...,
        description="List of document IDs to process for embedding.",
        example= ["doc1", "doc2", "doc3"]
    )

class SummaryInput(BaseModel):
    doc_id: str = Field(
        ...,
        description="Document ID to generate summary for",
        example="doc1"
    )
