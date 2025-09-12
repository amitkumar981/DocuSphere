import os
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY
import sys



class DocumentAnalyzer:
    "Analyzes documents using pretrained model"
    def __init__(self):
        try:
            self.loader = ModelLoader()
            self.log = CustomLogger().get_logger(__name__)
            self.llm = self.loader.load_llm()
            #define parser
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser,llm=self.llm)

            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("DocumentAnalyzer initialized Successfully")
        except Exception as e:
            self.log.info("error initializing DocumentAnalyzer:{e}")
            raise DocumentPortalException("error in DocumentAnalyzer initialization",sys)

    def analyze_document(self,document_text:str):
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Metadata chain initialized")

            response = chain.invoke({
                "format_instructions":self.parser.get_format_instructions(),
                "document_text":document_text
            })

            self.log.info("Metadata extraction successful",keys=list(response.keys()))
            return response
        except Exception as e:
            self.log.error("Metadata analysis failed",error=str(e))
            raise DocumentPortalException("Metadata extraction failed") from e
