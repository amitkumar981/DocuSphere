from __future__ import annotations
import os
import sys
import shutil
import uuid
import hashlib
from pathlib import Path
from datetime import datetime,timezone
from typing import List, Dict,Optional,Iterable,Any

import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
import json
from utils.file_io import generate_session_id, save_uploaded_files
from utils.documents_ops import load_documents, concat_for_analysis, concat_for_comparison
from os import mkdir

SUPPORTED_EXTENSIONS = {".pdf",".docx",".txt"}

class FaissManager:
    def __init__(self,index_dir:str,model_loader: Optional[ModelLoader]= None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str,Any] = {"rows":{}}
        
        if self.meta_path.exists():
            try:
                self._meta = json.load(self.meta_path.read_text(encode="utf-8")) or {"rows":{}}
            except Exception:
                self._meta = {"rows":{}}
                
        self.model_loader = model_loader or ModelLoader()
        self.emd_model = self.model_loader.load_embedding_model()
        self.vs: Optional[FAISS] = None
        
    def _exist(self) -> bool:
        return (self.index_dir/"index.faiss").exists() or (self.index_dir/"index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str,md: dict[str,Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta,ensure_ascii=False,indent=2),encoding="utf-8")
    
    def add_documents(self,docs: List[Document]):
        if self.vs is None:
            raise RuntimeError("VectorStore is not initialized")
        
        new_docs: List[Document] = []
        for doc in docs:
            key = self._fingerprint(doc.page_content, doc.metadata)
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(doc)
        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)
    
    def load_or_create(self,texts: Optional[List[str]]= None,metadatas: Optional[List[Dict]]= None):
        if self._exist():
            self.vs= FAISS.load_local(str(self.index_dir),embeddings=self.emd_model,allow_dangerous_deserialization=True)
            return self.vs
        if not texts:
            raise DocumentPortalException("no text provided for vectorstore creation", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emd_model, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
class ChatIngestor:
    def __init__(self,temp_base: str ='data',faiss_base: str = 'faiss_index',use_session_dirs: bool=True,
                session_id: Optional[str]= None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.model_loader = ModelLoader()
            
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id() # type: ignore
            
            self.temp_base = Path(temp_base) 
            self.temp_base.mkdir(parents=True, exist_ok=True)
        
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)  
            
            
            self.temp_base = self._resolve_dir(self.temp_base)
            self.faiss_base  = self._resolve_dir(self.faiss_base)
            self.log.info("chat ingestor initialized", temp_base=self.temp_base, faiss_base=self.faiss_base)
        except Exception as e:
            self.log.error("chat ingestor initialization failed", error=str(e))
            raise DocumentPortalException("chat ingestor initialization failed", sys)
    
    def _resolve_dir(self,base: Path):
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents =True, exist_ok=True)
            return d
        return base
    
    def _split(self, docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(docs)
            self.log.info("text split successfully", chunks=len(chunks))
            return  chunks
        except Exception as e:
            self.log.error("Error splitting text", error=str(e))
            raise DocumentPortalException("text splitting failed", sys)
    
    def build_retriever(self,uploaded_files: Iterable,*,chunk_size: int =1000,chunk_overlap: int = 100,
                        k: int = 5):
        try:
            paths = save_uploaded_files(uploaded_files,self.temp_base)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents uploaded")
            
            chunks  = self._split(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            fm = FaissManager(self.faiss_base,self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts,metadatas=metas)
            except Exception as e:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            
            added =  fm.add_documents(chunks)
            self.log.info("retriever built successfully", added=added)
            return vs.as_retriever(search_type= 'similarity',search_kwargs={"k":k})
        except Exception as e:
            self.log.error("error in building retriever", error=str(e))
            raise DocumentPortalException("error in building retriever", sys)
        
            
class DocHandler:
    def __init__(self, data_dir: Optional[str] = None,session_id: Optional[str] = None):
        try:
            self.log = CustomLogger().get_logger()
            self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH",os.path.join(os.getcwd(),"data","document_analysis"))
            self.session_id  = session_id or generate_session_id("session")
            self.session_path = os.path.join(self.data_dir,self.session_id)
            os.makedirs(self.session_path, exist_ok=True)
            self.log.info("document handler initialized",session_id=self.session_id)
        except Exception as e:
            self.log.error("Error initializing document handler",error=str(e))
            raise DocumentPortalException("document handler initializing failed",sys)
    
    def save_files(self,uploaded_files) ->str:
        try:
            filename = os.path.basename(uploaded_files.name) 
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are allowed")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path,"wb") as f:
                if hasattr(uploaded_files, "read"):
                    f.write(uploaded_files.read())
                else:
                    f.write(uploaded_files.getbuffer())
            self.log.info("file saved successfully", filename=filename)
            return save_path
        except Exception as e:
            self.log.error("Error saving file", error=str(e))
            raise DocumentPortalException("file saving failed", sys)
        
    def read_files(self,pdf_path:str):
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n---Page {page_num + 1}---\n{page.get_text()}")
                text = "".join(text_chunks)
                self.log.info("PDF read successfully", pdf_path=pdf_path, pages=doc.page_count)
                return text
        except Exception as e:
            self.log.error(f"error in reading PDF: {e}")
            raise DocumentPortalException("error in reading document", e) from e
        
class DocumentComparator:
    def __init__(self,data_dir= "data/data_compare",session_id: Optional[str] = None):
        self.log  =  CustomLogger().get_logger(__name__)
        self.base_dir  =  Path(data_dir)
        self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.log.info("document comparator initialized",session_id=self.session_id)
    
    def save_uploaded_files(self,reference_file,actual_file) ->str:
        try:
            reference_path = self.session_path / reference_file.name
            actual_path  = self.session_path / actual_file.name
            
            for fobj, out in ((reference_file,reference_path),(actual_file,actual_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"): 
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
                self.log.info("file saved successfully", filename=fobj.name)
                return reference_path,actual_path
        except Exception as e:
            self.log.error("Error saving file", error=str(e))
            raise DocumentPortalException("file saving failed", sys)
      
    def read_files(self,pdf_path:str):
        try:
            with fitz.open(pdf_path) as docs:
                if docs.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported")
                parts = []
                for page_num in range(docs.page_count):
                    page = docs.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        parts.append(f"\n---Page {page_num + 1}---\n{text}")
            self.log.info("PDF read successfully", pdf_path=pdf_path, pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            self.log.error(f"error in reading PDF: {e}")
            raise DocumentPortalException("error in reading document", e) from e
    
    def combine_files(self):
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_files(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            self.log.info("Files combined successfully",count=len(doc_parts),session_id=self.session_id)
            return combined_text
        except Exception as e:
            self.log.error("Error combining files", error=str(e))
            raise DocumentPortalException("error in combining files", e) from e
    
    def clean_old_sessions(self,keep_latest:int =3):
        try:
            session = sorted([f for f in self.base_dir.iterdir() if f.is_dir()],reverse=True)
            for folder in session[keep_latest:]:
                shutil.rmtree(folder,ignore_errors=True)
                self.log.info("old session cleaned", path=str(folder))
        except Exception as e:
            self.log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("error in cleaning old sessions", e) from e
        
            
            
                                    
            
                
        
