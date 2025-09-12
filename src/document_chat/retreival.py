import os
import sys
from typing import List,Optional,Dict,Any
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

class ConversationalRAG:
    def __init__(self,session_id: Optional[str],retriever = None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id # type: ignore
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.contextualize_question.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.context_qa.value]
            self.retriever = retriever
            self.chain = None
            if self.retriever:
                self._build_chain()
            self.log.info("ConversationalRAG initialized successfully")
        except Exception as e:
            self.log.error("ConversationalRAG initialization failed", error=str(e))
            raise DocumentPortalException("error in initializing ConversationalRAG",sys)
        
    def load_retriever_from_faiss(self,index_path,k: int = 5,index_name: str = "index", search_type: str = 'similarity',
                                search_kwargs: Optional[Dict[str, Any]] = None):
        "load vectorstore form disk and convert into retriever"
        try:
            model_loader =ModelLoader()
            embedding = model_loader.load_embedding_model()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index path not found:{index_path}")
            vectorstore = FAISS.load_local(index_path,embedding,index_name,allow_dangerous_deserialization=True)
            if search_kwargs is None:
                search_kwargs = {"k": k}
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            self.log.info("retriever loaded from FAISS successfully", faiss_path=index_path)
            self._build_chain()
            return self.retriever
        except Exception as e:
            self.logger.error("loading retriever from FAISS failed", error=str(e))
            raise DocumentPortalException("error in loading retriever from faiss",sys)
    
    @staticmethod
    def _format_documents(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _load_llm(self):
        try:
            llm =ModelLoader().load_llm()
            if not llm:
                raise ValueError("llm could not loaded")
            self.log.info("llm loaded successfully",session_id=self.session_id)
            return llm           
        except Exception as e:
            self.log.error("loading LLM failed", error=str(e))
            raise DocumentPortalException("error in loading LLM", sys)
    
    def _build_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)
            question_rewriter = (
                {"input": itemgetter("input"),"chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            retrieve_docs=  question_rewriter | self.retriever | self._format_documents
            self.chain = (
                {
                "context":retrieve_docs,
                "input":itemgetter("input"),
                "chat_history":itemgetter("chat_history"),
            }
            |self.qa_prompt
            |self.llm
            |StrOutputParser()
            )
            self.log.info("chain built successfully",session_id=self.session_id)
        except Exception as e:
            self.log.error("building chain failed", error=str(e))
            raise DocumentPortalException("error in building chain", sys)
        
    def invoke(self,user_input:str,chat_history: Optional[List[BaseMessage]]= None)-> str:
        try:
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.warning("no answer generated from RAG chain",user_input=user_input,session_id=self.session_id)
                return "Sorry, I couldn't generate an answer."
            self.log.info("RAG chain invoked successfully", user_input=user_input, session_id=self.session_id,answer_preview= answer[:200])
            return answer
        except Exception as e :
            self.log.error("RAG chain failed", error=str(e))
            raise DocumentPortalException("error in RAG chain", sys)
    
        

            