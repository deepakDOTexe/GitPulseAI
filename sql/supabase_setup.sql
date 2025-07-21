-- Supabase Vector Database Setup for GitPulseAI
-- Run these commands in your Supabase SQL Editor

-- 1. Enable the vector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the gitlab_documents table
CREATE TABLE IF NOT EXISTS gitlab_documents (
    id SERIAL PRIMARY KEY,
    document_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    url TEXT DEFAULT '',
    section TEXT DEFAULT '',
    keywords JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(1536), -- OpenAI ada-002 embedding dimension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_gitlab_documents_document_id 
    ON gitlab_documents (document_id);

CREATE INDEX IF NOT EXISTS idx_gitlab_documents_title 
    ON gitlab_documents USING GIN (to_tsvector('english', title));

CREATE INDEX IF NOT EXISTS idx_gitlab_documents_content 
    ON gitlab_documents USING GIN (to_tsvector('english', content));

-- 4. Create vector similarity index (for faster vector searches)
-- Note: This requires documents to be added first, then create index with appropriate lists parameter
-- CREATE INDEX gitlab_documents_embedding_idx 
--     ON gitlab_documents USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- 5. Create the vector similarity search function
CREATE OR REPLACE FUNCTION match_gitlab_documents(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5
)
RETURNS TABLE(
    document_id TEXT,
    title TEXT,
    content TEXT,
    url TEXT,
    section TEXT,
    keywords JSONB,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        gitlab_documents.document_id,
        gitlab_documents.title,
        gitlab_documents.content,
        gitlab_documents.url,
        gitlab_documents.section,
        gitlab_documents.keywords,
        1 - (gitlab_documents.embedding <=> query_embedding) AS similarity
    FROM gitlab_documents
    WHERE gitlab_documents.embedding IS NOT NULL
      AND 1 - (gitlab_documents.embedding <=> query_embedding) > match_threshold
    ORDER BY gitlab_documents.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- 6. Create a function to get document statistics
CREATE OR REPLACE FUNCTION get_gitlab_documents_stats()
RETURNS TABLE(
    total_documents BIGINT,
    documents_with_embeddings BIGINT,
    avg_content_length NUMERIC,
    latest_update TIMESTAMP WITH TIME ZONE
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        COUNT(*) as total_documents,
        COUNT(embedding) as documents_with_embeddings,
        AVG(LENGTH(content)) as avg_content_length,
        MAX(updated_at) as latest_update
    FROM gitlab_documents;
$$;

-- 7. Create a trigger to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_gitlab_documents_updated_at 
    BEFORE UPDATE ON gitlab_documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- 8. Row Level Security (RLS) setup (optional, for production)
-- ALTER TABLE gitlab_documents ENABLE ROW LEVEL SECURITY;

-- Create policy for read access (adjust as needed)
-- CREATE POLICY "Allow read access to gitlab_documents" 
--     ON gitlab_documents FOR SELECT 
--     USING (true);

-- Create policy for insert/update (adjust as needed for your auth requirements)
-- CREATE POLICY "Allow insert/update to gitlab_documents" 
--     ON gitlab_documents FOR ALL 
--     USING (true);

-- 9. Sample data insertion function (for testing)
CREATE OR REPLACE FUNCTION insert_sample_gitlab_document(
    doc_id TEXT,
    doc_title TEXT,
    doc_content TEXT,
    doc_url TEXT DEFAULT '',
    doc_section TEXT DEFAULT ''
)
RETURNS BOOLEAN
LANGUAGE SQL
AS $$
    INSERT INTO gitlab_documents (document_id, title, content, url, section, keywords, metadata)
    VALUES (
        doc_id,
        doc_title,
        doc_content,
        doc_url,
        doc_section,
        '["gitlab", "handbook"]'::jsonb,
        '{"source": "manual", "test": true}'::jsonb
    )
    ON CONFLICT (document_id) DO UPDATE SET
        title = EXCLUDED.title,
        content = EXCLUDED.content,
        url = EXCLUDED.url,
        section = EXCLUDED.section,
        updated_at = NOW();
    
    SELECT true;
$$;

-- 10. Test the setup with sample data
SELECT insert_sample_gitlab_document(
    'test-doc-1',
    'GitLab Values',
    'GitLab values include collaboration, results, efficiency, diversity, iteration, and transparency. These values guide everything we do at GitLab.',
    'https://handbook.gitlab.com/handbook/values/',
    'company-culture'
);

-- 11. Verify the setup
SELECT 
    'Setup verification:' as status,
    (SELECT COUNT(*) FROM gitlab_documents) as document_count,
    (SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')) as vector_extension_enabled,
    (SELECT COUNT(*) FROM information_schema.routines WHERE routine_name = 'match_gitlab_documents') as search_function_exists;

-- 12. Performance optimization queries (run after adding documents)
-- Analyze table for query planner
-- ANALYZE gitlab_documents;

-- Create the vector index after you have documents (replace 100 with 4*sqrt(row_count))
-- Example for 625 documents: lists = 100
-- CREATE INDEX CONCURRENTLY gitlab_documents_embedding_idx 
--     ON gitlab_documents USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- Example search query (uncomment to test)
-- SELECT document_id, title, similarity 
-- FROM match_gitlab_documents(
--     (SELECT embedding FROM gitlab_documents LIMIT 1),  -- Use first doc's embedding as test
--     0.1,  -- Low threshold for testing
--     3     -- Return top 3 matches
-- );

COMMENT ON TABLE gitlab_documents IS 'GitLab handbook documents with vector embeddings for semantic search';
COMMENT ON FUNCTION match_gitlab_documents IS 'Semantic similarity search function using cosine distance';
COMMENT ON FUNCTION get_gitlab_documents_stats IS 'Get statistics about the document collection'; 