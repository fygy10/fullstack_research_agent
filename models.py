from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


#how to strcuture the analyst objects
class Analyst(BaseModel):
    affiliation: str = Field(
        description = 'Primary affiliation of the analyst'
    )

    name: str = Field(
        description = 'Name of the analyst'
    )

    role: str = Field(
        description = 'Role of the analyst'
    )

    description: str = Field(
        description = 'Description of the analyst'
    )

    @property
    def persona(self) -> str:
        return f'Name: {self.name}\nRole: {self.role}\nDescription: {self.description}\nAffiliation: {self.affiliation}\n'


#what there should be at a high level
class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description = 'List of analysts with their roles, affiliations, and descriptions'
    )


#analyst state that is passed through the graph
class GeneratedAnalystState(TypedDict):
    topic: str  
    number_analysts: int  
    analysts: Optional[List[Analyst]]  