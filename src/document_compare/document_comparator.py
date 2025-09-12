import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import SummaryResponse,PromptType
from prompt.prompt_library import PROMPT_REGISTRY
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from utils.model_loader import ModelLoader

class DocumentComparatorLLM:
    "Compares two documents using pretrained model"
    def __init__(self):
        load_dotenv(override=True)
        self.log =CustomLogger().get_logger(__name__)
        self.llm = ModelLoader().load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser,llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.document_comparison .value]
        self.chain = self.prompt | self.llm |self.parser 
        self.log.info("DocumentComparatorLLM initialized successfully")
    def compare_documents(self,combined_docs:str) ->pd.DataFrame:
        "compare two documents and highlight the differences"
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instructions": self.parser.get_format_instructions()}
            self.log.info("comparing documents",input_keys=inputs)
            
            response = self.chain.invoke(inputs)
            self.log.info("documents compared successfully",response=response)
            return self._format_response(response)
        except Exception as e:
            self.log.error("error in comparing documents",error=str(e))
            raise DocumentPortalException("error in comparing documents",e) from e
        
    def _format_response(self,response_parsed: list[dict]) -> pd.DataFrame:
        "format the response in required format"
        try:
            df = pd.DataFrame(response_parsed)
            self.log.info("response formatted successfully",dataframe = df)
            return df
        except Exception as e:
            self.log.error("error in formatting response",error=str(e))
            raise DocumentPortalException("error in formatting response",e) from e
    
    
    
    