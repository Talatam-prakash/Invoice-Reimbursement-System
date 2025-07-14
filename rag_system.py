# Import the necessary libraries
import os
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
# Disable parallelism in tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the RAGInvoiceChatbot class
# This class implements a Retrieval-Augmented Generation (RAG) chatbot for querying invoice analyses
class RAGInvoiceChatbot:
    """RAG chatbot for querying invoice analyses from ChromaDB."""
    # Initialize the chatbot with Groq API key and ChromaDB path
    def __init__(self, groq_api_key: str, chroma_persist_path: str = "./chroma_db"):
        """
        Args:
            groq_api_key: API key for Groq's LLM service.
            chroma_persist_path: Path to ChromaDB data (default: ./chroma_db).
        """
        # Initialize LLM client
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm_model = "llama-3.1-8b-instant"  # Default model

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Connect to ChromaDB (persistent) and create or get the collection
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
       
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

    # Define the main method to query invoices using the RAG pipeline
    def query_invoices(self, query: str, filters: Optional[Dict] = None) -> str:
        """
        End-to-end RAG pipeline: Retrieve invoices → Generate LLM response.
        
        Args:
            query: Natural language question (e.g., "Show declined invoices").
            filters: Metadata filters (e.g., {"status": "Declined"}).
            n_results: Number of invoices to retrieve.
        Returns:
            Markdown-formatted answer.
        """
        original_filters = filters.copy() if filters else None
        # Retrieve relevant invoices
        invoices = self._retrieve_invoices(query, filters)
        if not invoices:
            # Check if an employee_name filter was specifically used
            if original_filters and "employee_name" in original_filters:
                employee_name = original_filters["employee_name"]
                return f"No data provided for the employee named '{employee_name}'."
            else:
                return "No matching invoices found."

       
        # Generate LLM response
        return self._generate_response(query, invoices)

    def _retrieve_invoices(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Hybrid search (vector + metadata) with ChromaDB-compatible filters."""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        chroma_filters = None
        employee_name_filter_value = None
       # Flag to track if filters other than employee_name are present

        if filters:
            normalized_filters = {}
            for k, v in filters.items():
                if k.lower() == 'status':
                    normalized_filters['status'] = v.title()
    
                elif k.lower() == 'employee_name':
                    employee_name_filter_value = str(v).lower() # Store employee name separately
                    normalized_filters['employee_name'] = employee_name_filter_value
                else:
                    normalized_filters[k] = v
    
            # Construct the base conditions for the 'where' clause
            conditions = []
            if employee_name_filter_value:
                conditions.append({"employee_name": {"$eq": employee_name_filter_value}})
            
            for k, v in normalized_filters.items():
                if k.lower() != 'employee_name': # Add other filters if they exist
                    conditions.append({k: {"$eq": v}})
            
            if conditions:
                chroma_filters = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            else:
                chroma_filters = None # No filters applied if conditions is empty

        results = self.collection.query(
            query_embeddings=[query_embedding], # Always include query_embeddings
            where=chroma_filters, # Use the correctly combined filters
            n_results=100000, # Still retrieve a large number for this specific employee
            include=["metadatas", "documents"]
        )
        
        # If no results are found after filtering, return empty
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return []
        
        # If an employee name filter was specifically applied and no results were found,
        if employee_name_filter_value and not any(
            m.get('employee_name') == employee_name_filter_value
            for m in results.get('metadatas', [[]])[0]
        ):
            return []

        return [
            {"metadata": m, "document": d}
            for m, d in zip(results["metadatas"][0], results["documents"][0])
        ]
    def _generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate a structured response using LLM with retrieved invoice context."""
        # Prepare the context for LLM
        context_str = "\n\n".join([
            f"Invoice {i+1} Details:\n"
            f"- Employee: {inv['metadata']['employee_name']}\n"
            f"- Status: {inv['metadata']['status']}\n"
            f"- Amount: {inv['metadata']['invoice_amount']}\n"
            f"- Approved Amount: {inv['metadata']['amount_approved']}\n"
            f"- Reason: {inv['metadata']['reason']}\n"
            f"- Policy Reference: {inv['metadata'].get('policy_reference', 'N/A')}\n"
            for i, inv in enumerate(context)
        ])

        prompt = f"""
        You are an invoice reimbursement assistant. Based on the following invoice details, 
        answer the user's question in a clear, structured format. Follow these guidelines:
        
        1. Start with a brief summary of findings
        2. List each invoice with its key details
        3. Format amounts consistently (e.g., $1,000.00)
        4. Include policy references when available
        5. End with a summary of reimbursement statuses
        
        User Question: {query}
        
        Invoice Details:
        {context_str}
        
        Respond in this format:
        Summary: [Brief overview of what was found]
        
        [Number]. [Employee Name] - [Status]
        • Amount: [Amount] (Approved: [Approved Amount])
        • Reason: [Reason with policy reference]
        
        Summary:
        - Fully Reimbursed: [count]
        - Partially Reimbursed: [count]
        - Declined: [count]
        """

        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.llm_model,
            temperature=0.1,  # Keep responses factual
            response_format={"type": "text"}
        )
        
        return response.choices[0].message.content


# if __name__ == "__main__":
#     # Initialize system
#     groq_api_key = os.getenv("GROQ_API_KEY")
#     if not groq_api_key:
#         raise ValueError("GROQ_API_KEY environment variable not set")
#     # Initialize chatbot
#     chatbot = RAGInvoiceChatbot(groq_api_key)

#     # Query 1: Semantic search + metadata filter
#     response = chatbot.query_invoices(
#         query="show me all invoices",
#         n_results=10
#     )
#     print(response)