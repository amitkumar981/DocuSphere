import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from src.data_ingestion.data_ingestion import DocHandler,DocumentComparator,ChatIngestor
from src.document_analyzer.document_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retreival import ConversationalRAG
from utils.documents_ops import FastAPIFileAdaptor,read_pdf_via_handler
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
import shutil

log = CustomLogger().get_logger(__name__)

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")


app = FastAPI(title = "Document Portal",version= "0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
async def health_check() -> Dict[str,str]:
    lof.info("Health check passed.")
    return {"status": "ok","service":"document portal"}

@app.post("/analyze")
async def analyze_document(file: UploadFile = (...)) -> Any:
    try:
        log.info(f"received file for analysis :{file.filename}")
        dh = DocHandler()
        saved_path = dh.save_files(FastAPIFileAdaptor(file))
        text = read_pdf_via_handler(dh,saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        log.info("document analysis completed", result=result)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...),actual: UploadFile=File(...)) -> Any:
    try:
        log.info("received files for comparison", reference=reference.filename, actual=actual.filename)
        dc = DocumentComparator()
        ref_path,actual_path =dc.save_uploaded_files(FastAPIFileAdaptor(reference),FastAPIFileAdaptor(actual))
        _ =ref_path,actual_path
        combined_text = dc.combine_files()
        comp = DocumentComparatorLLM()
        df  = comp.compare_documents(combined_text)
        log.info("document comparison completed", result=df)
        return {"rows":df.to_dict(orient="records"),"session_id": dc.session_id}
    except HTTPException:
        raise
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/index")
async def chat_build_index(files: List[UploadFile] = File(...), session_id: Optional[str] = Form(None),use_session_dirs: bool = Form(True),
                        chunk_size: int = Form(1000),chunk_overlap: int = Form(200),k: int = Form(3)) -> Any:
    try:
        log.info("received files for indexing", files=[file.filename for file in files])
        wrapped = [FastAPIFileAdaptor(f) for f in files]
        ci = ChatIngestor(temp_base= UPLOAD_BASE,faiss_base=FAISS_BASE,use_session_dirs=use_session_dirs,
                        session_id=session_id or None)
        ci.build_retriever(wrapped,chunk_size=chunk_size,chunk_overlap=chunk_overlap,k=k)
        log.info("document indexing completed", session_id=ci.session_id)
        return {"session_id": ci.session_id,"k":k,"use_session_dirs":use_session_dirs}
    except HTTPException:
        raise
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.")

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")