from fastapi.responses import JSONResponse
from typing import List, Dict
from dotenv import load_dotenv
from invoice_system import InvoiceReimbursementSystem # Import InvoiceReimbursementSystem
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from typing import Annotated, Optional
import os
from rag_system import RAGInvoiceChatbot  # Import your RAG class

# Configuration
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Invoice Reimbursement API",
    description="API for processing invoices against HR policies",
    version="1.0.0"
)

# Initialize core system
system = InvoiceReimbursementSystem(
    groq_api_key=os.getenv("GROQ_API_KEY")  
)

# Initialize RAG chatbot
rag_chatbot = RAGInvoiceChatbot(
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Endpoint to analyze invoices
@app.post("/analyze-invoices/")
async def analyze_invoices(
    policy_file: Annotated[UploadFile, File(description="HR policy PDF/DOCX")],
    invoice_zip: Annotated[UploadFile, File(description="ZIP containing invoice PDFs")],
    employee_name: Annotated[str, Form(description="Default employee name")] = "Unknown"
):
    """Process invoices with enhanced error handling."""
    policy_temp_path = None
    zip_temp_path = None
    
    try:
        # Save policy file with proper extension
        policy_ext = os.path.splitext(policy_file.filename)[1].lower()
        if policy_ext not in ['.pdf', '.docx']:
            raise HTTPException(400, detail="Policy file must be PDF or DOCX")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=policy_ext) as policy_temp:
            policy_content = await policy_file.read()
            policy_temp.write(policy_content)
            policy_temp_path = policy_temp.name

        # Save ZIP file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as zip_temp:
            zip_content = await invoice_zip.read()
            zip_temp.write(zip_content)
            zip_temp_path = zip_temp.name

        # Verify policy file has content
        if os.path.getsize(policy_temp_path) == 0:
            raise HTTPException(400, detail="Policy file is empty")

        # Process invoices
        result = system.process_invoices(
            policy_path=policy_temp_path,
            zip_path=zip_temp_path,
            employee_name=employee_name
        )
        
        if not result.get('total_processed', 0):
            raise HTTPException(400, detail="No valid invoices processed")
            
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup temp files
        for path in [policy_temp_path, zip_temp_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

# Endpoint to query invoices using RAG system
@app.get("/query-invoices/")
async def query_invoices(
    query: Annotated[str, Query(description="Natural language query about invoices")],
    status: Annotated[Optional[str], Query(description="Filter by status")] = None,
    employee_name: Annotated[Optional[str], Query(description="Filter by employee")] = None,
):
    """Query processed invoices using RAG system."""
    try:
        # Build filters dictionary
        filters = {}
        if status:
            filters["status"] = status.title()
        if employee_name:
            filters["employee_name"] = employee_name.lower()


        # Query the RAG system
        response = rag_chatbot.query_invoices(
            query=query,
            filters=filters if filters else None,
        )
        
        return {"query": query, "response": response}
    
    except Exception as e:
        raise HTTPException(500, detail=f"Query error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)