#!/usr/bin/env python3
"""
Migration script to upload GitLab handbook data to Supabase with Google Gemini embeddings.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseMigrator:
    """Migrate GitLab handbook data to Supabase vector database."""
    
    def __init__(
        self, 
        supabase_url: str, 
        supabase_key: str, 
        gemini_api_key: str,
        batch_size: int = 50,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize the Supabase migrator with Google Gemini embeddings.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service key
            gemini_api_key: Google Gemini API key
            batch_size: Number of documents to upload per batch
            rate_limit_delay: Delay between API calls to respect rate limits
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.gemini_api_key = gemini_api_key
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        logger.info("✅ Supabase migrator initialized with Google Gemini embeddings")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embeddings using Google Gemini embedding API via REST.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List of float values representing the embedding, or None if error
        """
        try:
            import requests
            
            # Use Google Gemini REST API for embeddings
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.gemini_api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "models/gemini-embedding-001",
                "content": {
                    "parts": [{"text": text}]
                },
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 1536  # Match existing Supabase schema
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            # Add rate limiting
            time.sleep(self.rate_limit_delay)
            
            result = response.json()
            if 'embedding' in result and 'values' in result['embedding']:
                return result['embedding']['values']
            else:
                logger.error("❌ Unexpected response format from Google Gemini API")
                return None
        
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding for text: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test both Supabase and Google Gemini API connections."""
        try:
            # Test Supabase connection
            logger.info("🔍 Testing Supabase connection...")
            result = self.supabase.table('gitlab_documents').select('count').execute()
            doc_count = len(result.data) if result.data else 0
            logger.info(f"✅ Supabase connected! Found {doc_count} existing documents")
            
            # Test Google Gemini embedding API
            logger.info("🔍 Testing Google Gemini embedding API...")
            test_embedding = self.generate_embedding("Hello, world!")
            
            if test_embedding and len(test_embedding) > 0:
                logger.info(f"✅ Google Gemini embedding API connected! Embedding dimension: {len(test_embedding)}")
                return True
            else:
                logger.error("❌ Google Gemini embedding API test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

    def upload_document(self, document: Dict[str, Any], generate_embeddings: bool = True) -> bool:
        """Upload a single document to Supabase with optional embedding generation."""
        try:
            # Create document data
            doc_data = {
                "document_id": document.get("id", f"doc_{hash(str(document))}"),
                "title": document.get("title", ""),
                "content": document.get("content", ""),
                "url": document.get("url", ""),
                "section": document.get("section", ""),
                "metadata": document.get("metadata", {}),
                "created_at": "now()",
                "updated_at": "now()"
            }
            
            # Generate embedding if requested
            if generate_embeddings and doc_data["content"]:
                logger.info(f"🔄 Generating embedding for: {doc_data['title'][:50]}...")
                embedding = self.generate_embedding(doc_data["content"])
                if embedding:
                    doc_data["embedding"] = embedding
                    logger.info("✅ Embedding generated successfully")
                else:
                    logger.warning("⚠️ Failed to generate embedding, uploading without it")
            
            # Upload to Supabase (upsert to handle duplicates)
            result = self.supabase.table('gitlab_documents').upsert(doc_data).execute()
            
            if result.data:
                logger.info(f"✅ Uploaded: {doc_data['title'][:50]}")
                return True
            else:
                logger.error(f"❌ Failed to upload: {doc_data['title'][:50]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error uploading document: {e}")
            return False

    def upload_documents_batch(self, documents: List[Dict[str, Any]], generate_embeddings: bool = True, batch_size: Optional[int] = None) -> int:
        """Upload multiple documents in batches."""
        if not documents:
            logger.warning("No documents to upload")
            return 0
        
        if batch_size is None:
            batch_size = self.batch_size
            
        total_documents = len(documents)
        successful_uploads = 0
        
        logger.info(f"📤 Uploading {total_documents} documents in batches of {batch_size}")
        
        # Process documents in batches
        for i in range(0, total_documents, batch_size):
            batch = documents[i:i + batch_size]
            batch_data = []
            
            logger.info(f"🔄 Processing batch {i//batch_size + 1}/{(total_documents + batch_size - 1)//batch_size}")
            
            for doc in batch:
                # Prepare document data
                doc_data = {
                    "document_id": doc.get("id", f"doc_{hash(str(doc))}"),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "url": doc.get("url", ""),
                    "section": doc.get("section", ""),
                    "metadata": doc.get("metadata", {}),
                    "created_at": "now()",
                    "updated_at": "now()"
                }
                
                # Generate embedding if requested
                if generate_embeddings and doc_data["content"]:
                    embedding = self.generate_embedding(doc_data["content"])
                    if embedding:
                        doc_data["embedding"] = embedding
                
                batch_data.append(doc_data)
            
            try:
                # Upload batch to Supabase
                result = self.supabase.table('gitlab_documents').upsert(batch_data).execute()
                batch_success = len(result.data) if result.data else 0
                successful_uploads += batch_success
                
                logger.info(f"✅ Batch uploaded: {batch_success}/{len(batch)} documents")
                
            except Exception as e:
                logger.error(f"❌ Batch upload failed: {e}")
                continue
        
        logger.info(f"🎉 Upload completed: {successful_uploads}/{total_documents} documents uploaded successfully")
        return successful_uploads

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about uploaded documents."""
        try:
            # Get total count
            all_docs = self.supabase.table('gitlab_documents').select("*").execute()
            total_docs = len(all_docs.data) if all_docs.data else 0
            
            # Get documents with embeddings
            docs_with_embeddings = [doc for doc in all_docs.data if doc.get('embedding')] if all_docs.data else []
            
            return {
                "total_documents": total_docs,
                "documents_with_embeddings": len(docs_with_embeddings),
                "documents_without_embeddings": total_docs - len(docs_with_embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def create_vector_index(self) -> bool:
        """Create vector similarity index for efficient searching."""
        try:
            # Calculate appropriate lists parameter (rule of thumb: sqrt(rows))
            stats = self.get_stats()
            total_docs = stats.get("total_documents", 100)
            lists = max(1, int(total_docs ** 0.5))
            
            logger.info(f"Creating vector index with lists={lists} for {total_docs} documents...")
            
            # Create the index using raw SQL
            index_sql = f"""
                CREATE INDEX IF NOT EXISTS gitlab_documents_embedding_idx 
                ON gitlab_documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """
            
            # Execute the SQL directly
            result = self.supabase.postgrest.rpc('exec_sql', {'query': index_sql}).execute()
            
            if result:
                logger.info("✅ Vector index created successfully")
                return True
            else:
                logger.error("❌ Failed to create vector index")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating vector index: {e}")
            return False

def main():
    """Main function to run the migration."""
    load_dotenv()
    
    # Get configuration from environment
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not supabase_url:
        logger.error("❌ SUPABASE_URL not found in environment variables")
        logger.info("Please add SUPABASE_URL=https://your-project.supabase.co to your .env file")
        return
    
    if not supabase_key:
        logger.error("❌ SUPABASE_KEY not found in environment variables")
        logger.info("Please add SUPABASE_KEY=your_service_key_here to your .env file")
        return
    
    if not gemini_api_key:
        logger.error("❌ GEMINI_API_KEY not found in environment variables")
        logger.info("Please add GEMINI_API_KEY=your_gemini_key_here to your .env file")
        return
    
    # Initialize migrator
    migrator = SupabaseMigrator(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        gemini_api_key=gemini_api_key
    )
    
    # Test connection
    logger.info("🚀 Starting Supabase migration with Google Gemini embeddings...")
    if not migrator.test_connection():
        logger.error("❌ Connection test failed. Please check your credentials.")
        return
    
    # Ask user which data file to use
    data_files = [
        "data/gitlab_two_pages.json",
        "data/gitlab_specific_policies.json", 
        "data/gitlab_comprehensive_handbook.json"
    ]
    
    print("\nAvailable data files:")
    for i, file in enumerate(data_files, 1):
        if os.path.exists(file):
            with open(file, 'r') as f:
                data = json.load(f)
                print(f"{i}. {file} ({len(data.get('documents', []))} documents)")
        else:
            print(f"{i}. {file} (not found)")
    
    try:
        choice = int(input("\nSelect data file (1-3): ")) - 1
        if choice < 0 or choice >= len(data_files):
            logger.error("Invalid choice")
            return
            
        data_file = data_files[choice]
        if not os.path.exists(data_file):
            logger.error(f"❌ Data file not found: {data_file}")
            return
    except ValueError:
        logger.error("Invalid input")
        return
    
    # Ask about generating embeddings
    generate_embeddings = input("Generate embeddings using Google Gemini? (y/N): ").lower().strip() == 'y'
    
    # Load and upload data
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        if not documents:
            logger.error("❌ No documents found in data file")
            return
        
        uploaded_count = migrator.upload_documents_batch(
            documents=documents, 
            generate_embeddings=generate_embeddings
        )
        
        logger.info(f"🎉 Migration completed! Uploaded {uploaded_count} documents to Supabase")
        
        if generate_embeddings and uploaded_count > 0:
            logger.info("🎯 Creating vector similarity index...")
            migrator.create_vector_index()
            logger.info("✅ Vector index created successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    main() 