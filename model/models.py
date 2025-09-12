from pydantic import BaseModel,Field,RootModel
from typing import Optional,List,Dict,Any,Union
from enum import Enum

class Metadata(BaseModel):
    summary: List[str] = Field(default_factory = list,description = "summary of documents" )
    Title: str
    Author: str
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int,str]
    SentimentTone: str
    
class ChangeFormat(BaseModel):
    page: str
    changes: str
    
class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass

class PromptType(str, Enum):
    document_analysis = "document_analysis"
    document_comparison = "document_comparison"
    contextualize_question = "contextualize_question"
    context_qa = "context_qa"
    
