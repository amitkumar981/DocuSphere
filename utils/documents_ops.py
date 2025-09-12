from __future__ import annotations
from pathlib import Path
from fastapi import UploadFile
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader,Docx2txtLoader,TextLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from typing import Iterable,Dict,List
import sys

SUPPORTED_EXTENSIONS = {".pdf",".docx",".txt",".md"}

log  = CustomLogger().get_logger(__name__)

def load_documents(paths: Iterable[Path]) -> List[Document]:
    "Load documents from paths"
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix
            if ext == ".pdf":
                loader = PyMuPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p),encoding = "utf-8")
            else:
                log.warning("Unsupported extensions skipped",path =str(p))
                continue
            docs.extend(loader.load())
        log.info("document loaded successfully", count =len(docs), path=str(p))
        return docs
    except Exception as e:
        log.error("failed to load documents", error=str(e))
        raise DocumentPortalException("failed to load documents", sys) from e

def concat_for_analysis(docs: List[Document]) ->str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "Unknown"
        parts.append(f"\n --SOURCE: {src}-- \n{d.page_content}")
    return "\n".join(parts)

def concat_for_comparison(ref_docs: List[Document],actual_docs: List[Document]) ->str:
    left  = concat_for_analysis(ref_docs)
    right = concat_for_analysis(actual_docs)
    return f"Reference Documents:\n{left}\n\nActual Documents:\n{right}"

class FastAPIFileAdaptor:
    "Adept FastApi UploadFile -> .name + getbuffer() API " 
    def __init__(self,uf: UploadFile):
        self._uf = uf
        self.name  = uf.filename
    def getbuffer(self):
        self._uf.file.seek(0)
        return self._uf.file.read()
def read_pdf_via_handler(handler,path: str) ->str:
    if hasattr(handler, "read_files"):
        return handler.read_files(path)
    elif hasattr(handler,"read_"):
        return handler.read_(path)
    raise RuntimeError("document handler has neither read_pdf not read_ method")
 

    
    

        
    
            
            
            
            
            