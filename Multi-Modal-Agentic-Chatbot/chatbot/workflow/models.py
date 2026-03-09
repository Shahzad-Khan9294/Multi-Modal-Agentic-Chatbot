from typing import Literal
from pydantic import BaseModel, Field
from typing import List, Optional

EntityType = Literal[
    "PRODUCT",
    "PERSON",
    "THING",
    "DATE",
    "LOCATION",
    "GOVERNMENT",
    "EVENT",
    "ORGANIZATION",
    "CONCEPT"
]


EntityRole = Literal[
    "SOURCE",   # Subject / initiator
    "TARGET"   # Receiver / object
]

class Entity(BaseModel):
    name: Optional[str] = Field(
        description="Exact phrase copied verbatim from the user query"
    )
    type: EntityType = Field(
        description="Type of the entity"
    )
    
    role: EntityRole = Field(
        description="Role of the entity in the user query"
    )

class Relationship(BaseModel):
    source_entity: Entity = Field(
        description="Source entity of the relationship"
    )
    target_entity: Optional[Entity] = Field(
        description="Target entity of the relationship"
    )
    relationship_type: Optional[str] = Field(
        description="Relationship type of the relationship"
    )

class RelationshipsOutput(BaseModel):
    relationships: List[Relationship] = Field(
        description="List of relationships between entities"
    )
