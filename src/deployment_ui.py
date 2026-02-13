import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import PyPDF2
from io import BytesIO
import time
import traceback

from inference import PaperQASystem
from utils import setup_logging, load_config

logger = setup_logging()

class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = None
    min_length: Optional[int] = None

class QuestionRequest(BaseModel):
    question: str
    context: str

class MultiQuestionRequest(BaseModel):
    questions: List[str]
    context: str

app = FastAPI(
    title="AI Paper Analyzer",
    description="Professional AI Research Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

system = None

@app.on_event("startup")
async def startup_event():
    global system
    logger.info("Loading models...")
    try:
        system = PaperQASystem(
            summarizer_path="./models/summarizer_final",
            qa_path="./models/qa_final"
        )
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        try:
            system = PaperQASystem(
                summarizer_path="sshleifer/distilbart-cnn-12-6",
                qa_path="distilbert-base-uncased-distilled-squad"
            )
        except Exception as e2:
            logger.error(f"Failed to load base models: {e2}")
            raise

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": system is not None}

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    try:
        if not system:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        summary = system.summarizer.summarize(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "summary": summary,
            "input_length": len(request.text.split()),
            "summary_length": len(summary.split())
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def answer_question(request: QuestionRequest):
    try:
        if not system:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        result = system.qa.answer_question(
            question=request.question,
            context=request.context
        )
        return result
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
async def upload_file(
    file: UploadFile = File(...),
    questions: str = Form(default="")
):
    """Upload and process a file (PDF or TXT)"""
    try:
        if not system:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        logger.info(f"Questions parameter: {questions}")
        
        # Read file contents
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes")
        
        # Extract text based on file type
        text = ""
        filename_lower = file.filename.lower()
        
        try:
            if filename_lower.endswith('.pdf'):
                logger.info("Processing as PDF...")
                pdf_file = BytesIO(contents)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    logger.info(f"Page {i+1}: extracted {len(page_text)} chars")
                    
            elif filename_lower.endswith('.txt'):
                logger.info("Processing as TXT...")
                try:
                    text = contents.decode('utf-8')
                except UnicodeDecodeError:
                    text = contents.decode('latin-1')
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Please upload PDF or TXT."
                )
            
            logger.info(f"Total extracted text: {len(text)} characters")
            
            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="No text could be extracted from the file"
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text: {str(e)}"
            )
        
        # Process based on whether questions were provided
        start_time = time.time()
        
        if questions and questions.strip():
            # Q&A Mode
            question_list = [q.strip() for q in questions.split(',') if q.strip()]
            logger.info(f"Q&A mode: processing {len(question_list)} questions")
            
            try:
                qa_results = system.qa.answer_multiple_questions(question_list, text)
                processing_time = time.time() - start_time
                
                logger.info(f"Q&A completed: {len(qa_results)} answers in {processing_time:.2f}s")
                
                return {
                    "summary": "",
                    "qa_results": qa_results,
                    "processing_time": processing_time,
                    "filename": file.filename,
                    "text_length": len(text.split()),
                    "mode": "qa"
                }
            except Exception as e:
                logger.error(f"Q&A processing failed: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")
        else:
            # Summarize Mode
            logger.info("Summarize mode: generating summary")
            
            try:
                summary = system.summarizer.summarize(text)
                processing_time = time.time() - start_time
                
                logger.info(f"Summary completed in {processing_time:.2f}s")
                
                return {
                    "summary": summary,
                    "qa_results": [],
                    "processing_time": processing_time,
                    "filename": file.filename,
                    "text_length": len(text.split()),
                    "mode": "summarize"
                }
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
