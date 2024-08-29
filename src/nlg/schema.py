from typing import Optional, List
from langchain_core.pydantic_v1 import BaseModel, Field


class Disease(BaseModel):
    """
    A disease is a condition that negatively affects the structure or function of part or all of an organism, and is not due to any immediate external injury. Diseases are often construed as medical conditions associated with specific symptoms and signs. They may be caused by factors such as infections, genetics, or environmental factors, and can be chronic or acute. Diseases can affect humans, animals, and plants, and are studied in the field of medicine to understand their nature and provide appropriate treatments or prevention measures. 
    """
    names: Optional[List[str]] = Field(default=[], description="The names of diseases. Entity names included in parentheses should be extracted. For example, '(cancer)' should be considered 'cancer'. If a disease name occurs multiple times, it should be extracted multiple times.")


class Gene(BaseModel):
    """
    Proteins are large, complex molecules that play many critical roles in the body. They are essential for the structure, function, and regulation of the body’s tissues and organs. Proteins are made up of smaller units called amino acids, which are linked together in long chains. There are 20 different types of amino acids that can be combined to make a protein. The sequence of amino acids determines each protein’s unique 3-dimensional structure and its specific function.
    """
    names: Optional[List[str]] = Field(default=[], description="The names of the proteins related to gene.")


class POL(BaseModel):
    persons: Optional[List[str]] = Field(default=[], description="The names of persons.")
    organizations: Optional[List[str]] = Field(default=[], description="The names of organizations.")
    locations: Optional[List[str]] = Field(default=[], description="The names of locations.")
    miscs: Optional[List[str]] = Field(default=[], description="The names of miscellaneous entities. A miscellaneous entity is a catch-all category where the entity does not belong to the person, organization, or location category.")