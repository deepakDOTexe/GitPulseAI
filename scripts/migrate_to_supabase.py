#!/usr/bin/env python3
"""
Migrate GitLab Handbook Data to Supabase Vector Database

This script uploads your local GitLab handbook data to Supabase,
generating embeddings and storing them for cloud deployment.

Usage:
    python scripts/migrate_to_supabase.py
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    print("supabase-py not installed. Run: pip install supabase")
    HAS_SUPABASE = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    print("openai not installed. Run: pip install openai")
    HAS_OPENAI = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseMigrator:
    """Migrate GitLab handbook data to Supabase vector database."""
    
    def __init__(self, 
                 supabase_url: str,
                 supabase_key: str,
                 openai_api_key: str,
                 table_name: str = "gitlab_documents"):
        """Initialize the migrator."""
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.openai_api_key = openai_api_key
        self.table_name = table_name
        
        # Initialize clients
        self.supabase: Client = create_client(supabase_url, supabase_key)
        openai.api_key = openai_api_key
        
        logger.info("Initialized Supabase migrator")
    
    def test_connections(self) -> bool:
        """Test connections to Supabase and OpenAI."""
        try:
            # Test Supabase connection
            result = self.supabase.table(self.table_name).select("count", count="exact").limit(1).execute()
            logger.info("✅ Supabase connection successful")
            
            # Test OpenAI connection
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input="test"
            )
            logger.info("✅ OpenAI connection successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI."""
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text.replace('\n', ' ')
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def upload_document(self, document: Dict[str, Any], generate_embeddings: bool = True) -> bool:
        """Upload a single document to Supabase."""
        try:
            doc_id = document.get('id')
            content = document.get('content', '')
            
            if not doc_id or not content.strip():
                logger.warning(f"Skipping document with missing id or content: {doc_id}")
                return False
            
            # Generate embedding if requested
            embedding = None
            if generate_embeddings:
                embedding = self.generate_embedding(content)
                if not embedding:
                    logger.warning(f"Failed to generate embedding for document: {doc_id}")
            
            # Prepare document data
            doc_data = {
                'document_id': doc_id,
                'title': document.get('title', ''),
                'content': content,
                'url': document.get('url', ''),
                'section': document.get('section', ''),
                'keywords': json.dumps(document.get('keywords', [])),
                'metadata': json.dumps({
                    'source': document.get('source', ''),
                    'chunk_index': document.get('chunk_index', 0),
                    'migrated_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }),
                'embedding': embedding
            }
            
            # Upload to Supabase (upsert to handle duplicates)
            result = self.supabase.table(self.table_name).upsert(doc_data).execute()
            
            if result.data:
                logger.debug(f"✅ Uploaded document: {doc_id}")
                return True
            else:
                logger.error(f"❌ Failed to upload document: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading document {document.get('id', 'unknown')}: {e}")
            return False
    
    def upload_documents_batch(self, 
                              documents: List[Dict[str, Any]], 
                              generate_embeddings: bool = True,
                              batch_size: int = 50) -> int:
        """Upload multiple documents in batches."""
        successful_uploads = 0
        total_docs = len(documents)
        
        logger.info(f"Uploading {total_docs} documents in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_data = []
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
            
            for doc in batch:
                doc_id = doc.get('id')
                content = doc.get('content', '')
                
                if not doc_id or not content.strip():
                    continue
                
                # Generate embedding if requested
                embedding = None
                if generate_embeddings:
                    embedding = self.generate_embedding(content)
                    if embedding:
                        logger.debug(f"Generated embedding for {doc_id}")
                    else:
                        logger.warning(f"Failed embedding for {doc_id}")
                
                doc_data = {
                    'document_id': doc_id,
                    'title': doc.get('title', ''),
                    'content': content,
                    'url': doc.get('url', ''),
                    'section': doc.get('section', ''),
                    'keywords': json.dumps(doc.get('keywords', [])),
                    'metadata': json.dumps({
                        'source': doc.get('source', ''),
                        'chunk_index': doc.get('chunk_index', 0),
                        'migrated_at': time.strftime("%Y-%m-%d %H:%M:%S")
                    }),
                    'embedding': embedding
                }
                batch_data.append(doc_data)
                
                # Rate limiting for OpenAI API
                if generate_embeddings:
                    time.sleep(0.1)  # 10 requests per second max
            
            try:
                # Upload batch to Supabase
                result = self.supabase.table(self.table_name).upsert(batch_data).execute()
                batch_success = len(result.data) if result.data else 0
                successful_uploads += batch_success
                
                logger.info(f"✅ Batch {i//batch_size + 1}: Uploaded {batch_success}/{len(batch_data)} documents")
                
                # Small delay between batches
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
        
        logger.info(f"🎉 Migration complete: {successful_uploads}/{total_docs} documents uploaded")
        return successful_uploads
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about uploaded documents."""
        try:
            # Get total count
            count_result = self.supabase.table(self.table_name).select("count", count="exact").execute()
            total_docs = count_result.count if count_result.count is not None else 0
            
            # Get documents with embeddings
            embedding_result = self.supabase.table(self.table_name).select("count", count="exact").not_.is_("embedding", "null").execute()
            docs_with_embeddings = embedding_result.count if embedding_result.count is not None else 0
            
            return {
                'total_documents': total_docs,
                'documents_with_embeddings': docs_with_embeddings,
                'embedding_coverage': docs_with_embeddings / total_docs * 100 if total_docs > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def create_vector_index(self):
        """Create vector index for better performance."""
        try:
            stats = self.get_stats()
            total_docs = stats.get('total_documents', 0)
            
            if total_docs == 0:
                logger.warning("No documents found, skipping index creation")
                return
            
            # Calculate optimal lists parameter (4 * sqrt(rows))
            import math
            lists = max(1, int(4 * math.sqrt(total_docs)))
            
            logger.info(f"Creating vector index for {total_docs} documents with lists={lists}")
            
            # Create index using SQL
            index_sql = f"""
                CREATE INDEX IF NOT EXISTS gitlab_documents_embedding_idx 
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """
            
            result = self.supabase.rpc('exec_sql', {'sql': index_sql}).execute()
            logger.info("✅ Vector index created successfully")
            
        except Exception as e:
            logger.warning(f"Could not create vector index automatically: {e}")
            logger.info("You can create it manually in Supabase SQL editor after migration")


def main():
    """Main migration function."""
    
    if not HAS_SUPABASE or not HAS_OPENAI:
        print("❌ Missing required packages. Install with:")
        print("   pip install supabase openai")
        return
    
    print("🚀 GitLab Handbook to Supabase Migration Tool")
    print("This will upload your local GitLab data to Supabase with embeddings")
    print()
    
    # Get environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY") 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials. Please set:")
        print("   SUPABASE_URL=your_supabase_project_url")
        print("   SUPABASE_KEY=your_supabase_anon_key")
        return
    
    if not openai_api_key:
        print("❌ Missing OpenAI API key. Please set:")
        print("   OPENAI_API_KEY=your_openai_api_key")
        return
    
    # Available data files
    data_files = {
        "1": ("data/sample_gitlab_data.json", "Sample GitLab data (12 docs)"),
        "2": ("data/gitlab_specific_policies.json", "Specific policies (5 docs)"), 
        "3": ("data/gitlab_comprehensive_handbook.json", "Comprehensive handbook (25 docs)"),
        "4": ("data/gitlab_two_pages.json", "Two pages only (2 docs)")
    }
    
    print("📁 Available data files:")
    for key, (file_path, description) in data_files.items():
        file_exists = Path(file_path).exists()
        status = "✅" if file_exists else "❌"
        print(f"   {key}. {description} {status}")
    
    print()
    choice = input("Select data file (1-4, default=3): ").strip() or "3"
    
    if choice not in data_files:
        print(f"❌ Invalid choice: {choice}")
        return
    
    input_file, description = data_files[choice]
    
    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        return
    
    # Ask about embeddings
    print()
    generate_embeddings = input("Generate embeddings? (y/N): ").strip().lower() == 'y'
    
    print(f"\n📤 Configuration:")
    print(f"   Data file: {input_file}")
    print(f"   Generate embeddings: {'Yes' if generate_embeddings else 'No'}")
    print(f"   Supabase URL: {supabase_url}")
    
    print()
    confirm = input("Continue with migration? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Migration cancelled")
        return
    
    try:
        # Initialize migrator
        print("\n🔧 Initializing migration...")
        migrator = SupabaseMigrator(supabase_url, supabase_key, openai_api_key)
        
        # Test connections
        if not migrator.test_connections():
            print("❌ Connection test failed")
            return
        
        # Load data
        print(f"📥 Loading data from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        if not documents:
            print("❌ No documents found in data file")
            return
        
        print(f"Found {len(documents)} documents")
        
        # Upload documents
        print(f"\n⬆️ Starting migration...")
        successful_uploads = migrator.upload_documents_batch(
            documents, 
            generate_embeddings=generate_embeddings,
            batch_size=25  # Smaller batches for embedding generation
        )
        
        # Show final stats
        print(f"\n📊 Final Statistics:")
        stats = migrator.get_stats()
        for key, value in stats.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Create vector index if embeddings were generated
        if generate_embeddings and stats.get('documents_with_embeddings', 0) > 0:
            print(f"\n🔧 Creating vector index...")
            migrator.create_vector_index()
        
        print(f"\n🎉 Migration Complete!")
        print(f"   Uploaded: {successful_uploads}/{len(documents)} documents")
        print(f"   Your Supabase vector database is ready for GitPulseAI!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logger.exception("Detailed migration error:")


if __name__ == "__main__":
    main() 