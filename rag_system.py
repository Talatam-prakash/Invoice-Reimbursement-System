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
        except ValueError:
            self.collection = self.chroma_client.create_collection(
                name="invoices",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )

    # Define the main method to query invoices using the RAG pipeline
    def query_invoices(self, query: str, filters: Optional[Dict] = None, n_results: int = 3) -> str:
        """
        End-to-end RAG pipeline: Retrieve invoices → Generate LLM response.
        
        Args:
            query: Natural language question (e.g., "Show declined invoices").
            filters: Metadata filters (e.g., {"status": "Declined"}).
            n_results: Number of invoices to retrieve.
        Returns:
            Markdown-formatted answer.
        """
        # Retrieve relevant invoices
        invoices = self._retrieve_invoices(query, filters, n_results)
        if not invoices:
            return "No matching invoices found."

        # Generate LLM response
        return self._generate_response(query, invoices)

    # Private method to retrieve invoices using hybrid search (vector + metadata)
    def _retrieve_invoices(self, query: str, filters: Optional[Dict] = None, n_results: int = 3) -> List[Dict]:
        """Hybrid search (vector + metadata) with ChromaDB-compatible filters."""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Handle filters
        chroma_filters = None
        if filters:
            # Convert each filter to ChromaDB format
            conditions = [{k: {"$eq": v}} for k, v in filters.items()]
            
            # Use $and only if we have multiple conditions
            if len(conditions) > 1:
                chroma_filters = {"$and": conditions}
            else:
                chroma_filters = conditions[0]  # Single condition doesn't need $and
        
        # Perform the query with ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=chroma_filters,
            n_results=n_results,
            include=["metadatas", "documents"]
        )
        
        return [
            {"metadata": m, "document": d}
            for m, d in zip(results["metadatas"][0], results["documents"][0])
        ]
    
    # Private method to generate a structured response using LLM
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

        # Construct the prompt for LLM
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