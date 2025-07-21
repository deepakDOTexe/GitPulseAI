# Create Vector Index in Supabase

After successfully migrating your documents to Supabase, you need to create a vector index for fast similarity search.

## 📋 Prerequisites

- ✅ Documents uploaded to Supabase with embeddings
- ✅ Admin access to your Supabase project dashboard

## 🚀 Steps to Create Vector Index

### 1. Open Supabase Dashboard
- Go to [https://supabase.com/dashboard](https://supabase.com/dashboard)
- Select your GitPulseAI project

### 2. Navigate to SQL Editor
- Click **"SQL Editor"** in the left sidebar
- Click **"New query"**

### 3. Run the Index Creation SQL (with Memory Fix)

**If you get a memory error**, run these commands in sequence:

**Step 3a: Temporarily increase memory**
```sql
-- Increase memory temporarily for index creation
SET maintenance_work_mem = '256MB';
```

**Step 3b: Create the index**
```sql
-- Create the vector index
CREATE INDEX IF NOT EXISTS gitlab_documents_embedding_idx 
ON gitlab_documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 5);
```

**Step 3c: Reset memory to default**
```sql
-- Reset memory back to default
RESET maintenance_work_mem;
```

### Alternative: Create Index with Lower Memory Requirements

If the above still fails, use a smaller lists parameter:

```sql
-- Set memory
SET maintenance_work_mem = '128MB';

-- Create index with fewer lists (uses less memory)
CREATE INDEX IF NOT EXISTS gitlab_documents_embedding_idx 
ON gitlab_documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 2);

-- Reset memory
RESET maintenance_work_mem;
```

### 4. Execute the Queries
- **Run each SQL block separately** (don't run all at once)
- Click **"Run"** for each command
- You should see: `SET` → `CREATE INDEX` → `RESET`

## ✅ Verification

To verify the index was created successfully, run:

```sql
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'gitlab_documents' 
AND indexname = 'gitlab_documents_embedding_idx';
```

You should see your new index in the results.

## 🎯 What This Enables

With the vector index created:
- ⚡ **Fast similarity search** (milliseconds instead of seconds)
- 🔍 **Efficient embedding queries** for your RAG system  
- 📈 **Scalable performance** as your document collection grows

## 🚨 Important Notes

- **One-time setup**: You only need to do this once per database
- **Background process**: Index creation runs in the background
- **No downtime**: Your app remains available during index creation
- **Automatic usage**: Your RAG system will automatically use the index
- **Memory setting**: The memory increase is temporary and session-specific

## 🆘 Troubleshooting

**If you get memory error (54000):**
- Use the 3-step approach above (SET → CREATE → RESET)
- Try smaller lists parameter (lists = 1 or lists = 2)

**If you get an error about pgvector:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Memory requirements by document count:**
- **25 documents**: lists = 2 (lower memory)
- **100 documents**: lists = 5 (medium memory)  
- **1000+ documents**: lists = 10+ (higher memory, may need more RAM)

**If Supabase free tier has memory limits:**
```sql
-- Minimal memory approach
SET maintenance_work_mem = '64MB';
CREATE INDEX IF NOT EXISTS gitlab_documents_embedding_idx 
ON gitlab_documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1);
RESET maintenance_work_mem;
```

---

🎉 **Once complete, your GitPulseAI is ready for cloud deployment!** 