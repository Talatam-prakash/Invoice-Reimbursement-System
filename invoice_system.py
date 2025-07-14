# Import necessary libraries
import os
import json
import zipfile
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PyPDF2 import PdfReader
import docx
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import fitz  

# Define the InvoiceReimbursementSystem class
# This class handles invoice processing, analysis, and storage in a vector database
class InvoiceReimbursementSystem:
    """Core system for invoice processing and analysis."""
    
    # Initialize the system with Groq API key and vector database
    def __init__(self, groq_api_key: str):
        """Initialize with Groq API key and vector database.
        
        Args:
            groq_api_key: API key for Groq LLM service
        """

        # Intialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Connect to ChromaDB (persistent) and create or get the collection
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
        
            self.collection = self.chroma_client.get_collection(
                name="invoices",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
        except chromadb.errors.NotFoundError:
            self.collection = self.chroma_client.create_collection(
                name="invoices",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )

    # Method to extract text from PDF or DOCX files
    def extract_text_from_file(self, file_path: str) -> Optional[str]:
        """Extract text from PDF or DOCX file with robust error handling."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file extension and extract text accordingly
            if file_path.lower().endswith('.pdf'):
                try:
                    text = ""
                    with fitz.open(file_path) as doc:
                        for page in doc:
                            text += page.get_text()
                    if text.strip():
                        return text.strip()
                except Exception:
                   # Fallback to PyPDF2 if fitz fails
                    with open(file_path, "rb") as f:
                        reader = PdfReader(f)
                        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                        return text if text.strip() else None
            # DOCX file handling
            elif file_path.lower().endswith('.docx'):
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs if para.text])
            
            raise ValueError(f"Unsupported file format: {file_path}")
            
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return None
        
    # def extract_text_from_pdf(self, file_path) -> Optional[str]:
    #     """Extract text from a PDF file using PyMuPDF (fitz)."""
    #     try:
    #         text = ""
    #         with fitz.open(file_path) as doc:
    #             for page in doc:
    #                 text += page.get_text()
    #         return text.strip() if text.strip() else None
    #     except Exception as e:
    #         print(f"Error reading PDF: {str(e)}")
    #         return None

    # Method to analyze invoice against HR policy using Groq LLM
    def analyze_invoice(self, policy_text: str, invoice_text: str, employee_name: str = None, max_retries: int = 3) -> Optional[Dict]:
        """Analyze invoice against policy using Groq LLM."""

        # Prepare the prompt for LLM
        prompt = f"""
        You are an expert invoice analyzer for our reimbursement system. Carefully examine this invoice 
    against the company's HR reimbursement policy and provide a detailed analysis. Return JSON with:
        - "employee_name": Name from invoice or '{employee_name}' if not found
        - "status": "Fully Reimbursed", "Partially Reimbursed", or "Declined"
         - "reason": "Must include: (1) Exact policy violation/approval reason, (2) Specific policy section reference (e.g., 'Section 4.2'), (3) If partial approval, state approved vs. claimed amounts"
        - "amount_approved": Approved amount
        - "invoice_amount": Total invoice amount
        - "policy_reference": Relevant policy section
        - "invoice_date": Date in YYYY-MM-DD format

        HR Policy:
        {policy_text}

        Invoice Details:
        {invoice_text}

Example Response:
        {{
            "employee_name": "Ajay Kumar",
            "status": "Partially Reimbursed",
            "reason": "Cab fare exceeds $100 daily limit (Policy Section 4.2)",
            "amount_approved": 100.00,
            "invoice_amount": 120.00,
            "policy_reference": "Section 4.2: Transportation Limits",
            "invoice_date": "2024-05-15"
        }}
         Important:
        - The reason must explicitly reference the policy section number
        - For partial approvals, clearly state both approved and claimed amounts
        - Avoid vague phrases like 'as per policy' - be specific
        """
        
        # Retry logic for robustness
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    response_format={"type": "json_object"},
                    temperature=0.3 # for detailed, factual responses
                )
                
                analysis = json.loads(response.choices[0].message.content)
                
                # Validate response
                required_keys = {
                    "status", "reason", "amount_approved", 
                    "invoice_amount", "policy_reference", "invoice_date"
                }
                if not all(k in analysis for k in required_keys):
                    missing = required_keys - analysis.keys()
                    raise ValueError(f"Missing fields: {missing}")
                
                # Use provided name if not found
                if "employee_name" not in analysis or not analysis["employee_name"].strip():
                    analysis["employee_name"] = employee_name
                
                # Validate status
                valid_statuses = {"Fully Reimbursed", "Partially Reimbursed", "Declined"}
                if analysis["status"] not in valid_statuses:
                    raise ValueError(f"Invalid status: {analysis['status']}")
                
                # Convert amounts to float
                try:
                    analysis["amount_approved"] = float(analysis["amount_approved"])
                    analysis["invoice_amount"] = float(analysis["invoice_amount"])
                except ValueError:
                    raise ValueError("Amounts must be numeric")
                
                # Validate date
                try:
                    datetime.strptime(analysis["invoice_date"], "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Invalid date format: {analysis['invoice_date']}")
                
                return analysis
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
    
    # Method to process a batch of invoices and store them in the vector database
    def process_invoices(self, policy_path: str, zip_path: str, employee_name: str = "Unknown") -> Dict:
        """Process invoice batch and store in vector database."""
        
        # Extract policy text
        policy_text = self.extract_text_from_file(policy_path)
        if not policy_text:
            raise ValueError(f"Failed to extract policy text from: {policy_path}")
        
        # Verify policy format
        if not (policy_path.lower().endswith('.pdf') or policy_path.lower().endswith('.docx')):
            raise ValueError("Policy must be PDF or DOCX")
        
        # Extract invoices
        extraction_path = 'extracted_invoices'
        os.makedirs(extraction_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)

        processed_invoices = []
        for root, _, files in os.walk(extraction_path):
            for file in files:
                file_path = os.path.join(root, file)
                if not file.lower().endswith('.pdf'):
                    continue
                
                try:
                    # Process invoice
                    invoice_text = self.extract_text_from_file(file_path)
                    if not invoice_text:
                        print(f"Skipping empty PDF: {file}")
                        continue
                    
                    analysis = self.analyze_invoice(policy_text, invoice_text, employee_name)
                    if not analysis:
                        print(f"Failed to analyze: {file}")
                        continue
                    
                    # Store in vector DB 
                    embedding = self.embedding_model.encode(invoice_text)
                    metadata = {
                        "employee_name": analysis["employee_name"].lower(),
                        "status": analysis["status"].title(),
                        "reason": analysis["reason"],
                        "amount_approved": str(analysis["amount_approved"]),
                        "invoice_amount": str(analysis["invoice_amount"]),
                        "policy_reference": analysis["policy_reference"],
                        "invoice_date": analysis["invoice_date"],
                        "file_name": file,
                        "source_employee_name": "extracted" if analysis["employee_name"] != employee_name else "provided"
                    }
                    
                    self.collection.add(
                        documents=[invoice_text],
                        embeddings=[embedding.tolist()],
                        metadatas=[metadata],
                        ids=[f"{analysis['employee_name']}_{file}"]
                    )
                    
                    processed_invoices.append({
                        "file": file,
                        "status": analysis["status"],
                        "employee": analysis["employee_name"],
                        "amount_approved": analysis["amount_approved"],
                        "invoice_amount": analysis["invoice_amount"]
                    })
                    
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    continue
        
        # Cleanup
        for root, dirs, files in os.walk(extraction_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(extraction_path)
        # self.chroma_client.persist()
        return {
            "total_processed": len(processed_invoices),
            "results": processed_invoices,
            "summary": {
                "fully_reimbursed": sum(1 for i in processed_invoices if i["status"] == "Fully Reimbursed"),
                "partially_reimbursed": sum(1 for i in processed_invoices if i["status"] == "Partially Reimbursed"),
                "declined": sum(1 for i in processed_invoices if i["status"] == "Declined")
            },
            "vector_db_status": "Data stored successfully"
        }