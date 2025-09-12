# #import os
# #from pathlib import Path
# #from src.document_analyzer.data_injection import DocumentHandler
# #from src.document_analyzer.data_analysis import DocumentAnalyzer

# #pdf_path = r"C:\Users\redhu\OneDrive\Desktop\document_portal\data\document_analysis\sample.pdf"

# class DummyFile:
#         def __init__(self,file_path):
#             self.name =  Path(file_path).name
#             self._file_path = file_path
#         def getbuffer(self):
#             return open(self._file_path,"rb").read()
# def main():
#     try:
#         #____data injection____
#         print("strating pdf injection....")
#         dummy_pdf = DummyFile(pdf_path)

#         handler = DocumentHandler(session_id="test_injection_analysis")
#         saved_path = handler.save_document(dummy_pdf)
#         print(f"pdf saved at:{saved_path}")

#         text_content = handler.read_document(saved_path)
#         print(f"Extracted text length: {len(text_content)} chars\n")

#         #### data analysis#####
#         print("starting Metadata analysis ....")
#         analyzer = DocumentAnalyzer()
#         analysis_result = analyzer.analyze_document(text_content) 

#         #### Display Results####
#         print("\n ===Metadata Analysis Results===")
#         for key,value  in analysis_result.items():
#             print(f"{key}:{value}")
#     except Exception as e:
#         print(f"test failed:{e}")
# import io
# from pathlib import Path 
# from src.document_compare.data_injection import DocumentInjection
# from src.document_compare.document_camparator import DocumentComparatorLLM

# def fake_upload(file_path:Path):
#     return io.BytesIO(file_path.read_bytes())

# def test_compare_documents():
#     ref_path = Path(r"C:\Users\redhu\OneDrive\Desktop\document_portal\data\document_compare\UserFile2.pdf")
#     actual_path = Path(r"C:\Users\redhu\OneDrive\Desktop\document_portal\data\document_compare\UserFile3.pdf")
    
#     class FakeUpload:
#         def __init__(self,file_path:Path):
#             self.name = file_path.name
#             self._buffer = file_path.read_bytes()
            
#         def get_buffer(self):
#             return (self._buffer)
#     comparator = DocumentInjection()

#     ref_upload = FakeUpload(ref_path)
#     actual_upload = FakeUpload(actual_path)
#     ref_file,actual_file = comparator.save_uploaded_documents(ref_upload,actual_upload)
#     combined_text = comparator.combine_documents()

#     print("/n===Combined Document Text Preview===")
#     print(combined_text[:500])

#     llm_comparator = DocumentComparatorLLM()
#     comparesion_df = llm_comparator.compare_documents(combined_text)
#     print()

#     print("\n===Comparison Result DataFrame===")
    
# if __name__=="__main__":
#     test_compare_documents()

# import sys
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from src.single_document_chat.data_injection import SingleDocsIngest
# from src.single_document_chat.data_retreival import ConversationalRAG
# from utils.model_loader import ModelLoader

# faiss_index_path=Path("index_path")

# def test_conversational_rag_on_pdf(pdf_path:str,question:str):
#     try:
#         model_loader =  ModelLoader()
#         if faiss_index_path.exists():
#             print("FAISS index found, loading retriever from index...")
#             embedding = model_loader.load_embedding_model()
#             vectorstore = FAISS.load_local(folder_path=str(faiss_index_path),embedding=embedding,allow_dangerous_deserialization=True)
#             retriever =  vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
#         else:
#             print('FAISS index not found, ingesting document and creating index...')
#             with open(pdf_path,"rb") as f:
#                 uploaded_file = [f]
#                 ingestor =SingleDocsIngest()
#                 retriever = ingestor.ingest_documents(uploaded_file)
#         print("Running Conversational RAG...")
#         session_id = "test_session_1"
#         rag = ConversationalRAG(retriever=retriever,session_id=session_id)
            
#         response = rag.invoke(user_input=question)
#         print(f"\nquestion: {question}\nAnswer: {response}")
#         print("test completed successfully")
#     except Exception as e:
#         print(f"test failed:{e}")
#         sys.exit
# if __name__=="__main__":
#     pdf_path = r"C:\Users\redhu\OneDrive\Desktop\document_portal\data\single_doument_chat\Gradient boosting_ Distance to target.pdf"
#     question = "What are working of gradient boosting explain in simple terms?"
    
#     if not Path(pdf_path).is_file():
#         print(f"pdf file not found at:{pdf_path}")
#         sys.exit(1)

# test_conversational_rag_on_pdf(pdf_path,question)
        
    
#### multi doc chat################
import sys
from pathlib import Path
from src.multi_document_chat.retrieval import ConversationalRAG
from src.multi_document_chat.data_ingestion import DocumentInegestor

def test_conversational_rag_on_multiple_pdfs():
    try:
        test_files = [
            r"C:\Users\redhu\OneDrive\Desktop\document_portal\data\multi_doc_chat\Gradient boosting_ Distance to target.pdf",
            r"C:\Users\redhu\OneDrive\Desktop\document_portal\data\multi_doc_chat\sample.pdf",
            r"C:\Users\redhu\OneDrive\Desktop\document_portal\data\multi_doc_chat\New Microsoft Word Document.docx"
            
        ]
        uploaded_files = []
        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"file not found at:{file_path}")
        
        if not uploaded_files:
            print("No valid files to upload")
            sys.exit(1)
            
        ingestor = DocumentInegestor()
        retreiver = ingestor.ingest_documents(uploaded_files)
        
        for f in uploaded_files:
            f.close()
        
        print("Running Coversational RAG...")
        
        sesssion_id ="test document ingest"
        rag= ConversationalRAG(session_id=sesssion_id, retriever=retreiver)
        question ="how we Create Kubernetes Secret for ECR Access"
        response = rag.invoke(user_input=question)
        print(f"\nquestion:{question}\nAnswer:{response}")
        
        if not uploaded_files:
            print("No valid files to upload")
            sys.exit(1)
    except Exception as e:
        print(f"test failed:{e}")
        sys.exit(1)
if __name__ =="__main__":
    test_conversational_rag_on_multiple_pdfs()
            
            
        

           
    



    
   
   

   
   
