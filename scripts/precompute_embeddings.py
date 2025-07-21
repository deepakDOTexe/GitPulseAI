#!/usr/bin/env python3
"""
Pre-compute Embeddings for Streamlit Cloud Deployment

This script generates embeddings offline and saves them to JSON files
for fast loading on Streamlit Community Cloud.

Usage:
    python scripts/precompute_embeddings.py
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    print("sentence-transformers not installed. Run: pip install sentence-transformers")
    HAS_SENTENCE_TRANSFORMERS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingPrecomputer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding precomputer."""
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        if HAS_SENTENCE_TRANSFORMERS:
            self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get actual embedding dimension
            sample_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = sample_embedding.shape[1]
            
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.extend(batch_embeddings.tolist())
            
            # Small delay to be nice to the system
            time.sleep(0.1)
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def process_handbook_data(self, data_file: str, output_file: str):
        """Process GitLab handbook data and generate embeddings."""
        
        # Load the handbook data
        logger.info(f"Loading handbook data from: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        if not documents:
            raise ValueError("No documents found in data file")
        
        logger.info(f"Found {len(documents)} documents")
        
        # Extract content for embedding generation
        contents = []
        for doc in documents:
            content = doc.get('content', '').strip()
            if content:
                contents.append(content)
            else:
                logger.warning(f"Document {doc.get('id', 'unknown')} has no content")
        
        logger.info(f"Processing {len(contents)} documents with content")
        
        # Generate embeddings
        if not self.model:
            raise RuntimeError("Cannot generate embeddings without model")
        
        embeddings = self.generate_embeddings(contents)
        
        # Create the output data structure
        output_data = {
            "documents": [],
            "embeddings": embeddings,
            "metadata": {
                **data.get('metadata', {}),
                "embedding_model": self.model_name,
                "embedding_dimension": self.embedding_dimension,
                "precomputed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_embeddings": len(embeddings),
                "note": "Pre-computed embeddings for Streamlit Cloud deployment"
            }
        }
        
        # Add documents with their corresponding embedding indices
        embedding_idx = 0
        for i, doc in enumerate(documents):
            if doc.get('content', '').strip():
                doc_with_embedding = doc.copy()
                doc_with_embedding['embedding_index'] = embedding_idx
                output_data['documents'].append(doc_with_embedding)
                embedding_idx += 1
            else:
                logger.warning(f"Skipping document {doc.get('id', 'unknown')} - no content")
        
        # Save to output file
        logger.info(f"Saving precomputed embeddings to: {output_file}")
        
        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Calculate file size
        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        
        logger.info(f"Precomputed embeddings saved successfully:")
        logger.info(f"  - Documents: {len(output_data['documents'])}")
        logger.info(f"  - Embeddings: {len(embeddings)}")
        logger.info(f"  - Dimensions: {self.embedding_dimension}")
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        
        return output_data


def main():
    """Main function to precompute embeddings."""
    
    if not HAS_SENTENCE_TRANSFORMERS:
        print("❌ sentence-transformers not available. Install with:")
        print("   pip install sentence-transformers")
        return
    
    print("🚀 GitLab Handbook Embedding Precomputer")
    print("This will generate embeddings offline for fast Streamlit Cloud deployment")
    print()
    
    # Available data files
    data_dir = Path("data")
    available_files = {
        "1": ("data/sample_gitlab_data.json", "Sample GitLab data (12 docs)"),
        "2": ("data/gitlab_specific_policies.json", "Specific policies (5 docs)"),
        "3": ("data/gitlab_comprehensive_handbook.json", "Comprehensive handbook (25 docs)"),
        "4": ("data/gitlab_two_pages.json", "Two pages only (2 docs)")
    }
    
    print("📁 Available data files:")
    for key, (file_path, description) in available_files.items():
        file_exists = Path(file_path).exists()
        status = "✅" if file_exists else "❌"
        print(f"   {key}. {description} {status}")
    
    print()
    choice = input("Select data file (1-4, default=3): ").strip() or "3"
    
    if choice not in available_files:
        print(f"❌ Invalid choice: {choice}")
        return
    
    input_file, description = available_files[choice]
    
    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        return
    
    # Generate output filename
    input_path = Path(input_file)
    output_file = f"data/precomputed_{input_path.stem}.json"
    
    print(f"📥 Input: {input_file}")
    print(f"📤 Output: {output_file}")
    print()
    
    # Initialize precomputer
    print("🔧 Initializing embedding model...")
    precomputer = EmbeddingPrecomputer()
    
    if not precomputer.model:
        print("❌ Failed to load embedding model")
        return
    
    try:
        # Process the data
        print("⚡ Processing handbook data and generating embeddings...")
        result = precomputer.process_handbook_data(input_file, output_file)
        
        print(f"\n🎉 Success! Precomputed embeddings saved to:")
        print(f"   {output_file}")
        print(f"\n📋 Next steps for Streamlit Cloud deployment:")
        print(f"1. Use the precomputed embeddings file in your app")
        print(f"2. Update your vector store to load precomputed embeddings")
        print(f"3. Add st.cache_data decorators for fast loading")
        print(f"4. Deploy to Streamlit Cloud!")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        logger.exception("Detailed error:")


if __name__ == "__main__":
    main() 