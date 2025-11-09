"""
Integration tests for database schema.
Verifies that all required tables exist after database initialization.
"""
import pytest
import os
from dotenv import load_dotenv
from retrieval.db_utils import connect

load_dotenv()


class TestDatabaseSchema:
    """Tests for database schema initialization."""
    
    @pytest.mark.skipif(
        not all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"),
            os.getenv("DB_PASS"),
            os.getenv("DB_NAME")
        ]),
        reason="Database connection environment variables not set"
    )
    def test_documents_table_exists(self):
        """Test that the documents table exists with expected structure."""
        try:
            with connect() as conn, conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'documents'
                """)
                result = cur.fetchone()
                assert result is not None, "documents table does not exist"
                
                # Check expected columns
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'documents'
                    ORDER BY ordinal_position
                """)
                columns = {row[0]: row[1] for row in cur.fetchall()}
                
                # Verify required columns exist
                assert 'doc_id' in columns, "documents table missing doc_id column"
                assert 'title' in columns, "documents table missing title column"
                assert 'source_path' in columns, "documents table missing source_path column"
                assert 'meta' in columns, "documents table missing meta column"
                assert 'created_at' in columns, "documents table missing created_at column"
                
                print("✅ documents table exists with expected structure")
                
        except Exception as e:
            pytest.fail(f"documents table check failed: {e}")
    
    @pytest.mark.skipif(
        not all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"),
            os.getenv("DB_PASS"),
            os.getenv("DB_NAME")
        ]),
        reason="Database connection environment variables not set"
    )
    def test_chunks_table_exists(self):
        """Test that the chunks table exists with expected structure for multi-modal embeddings."""
        try:
            with connect() as conn, conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'chunks'
                """)
                result = cur.fetchone()
                assert result is not None, "chunks table does not exist"
                
                # Check expected columns
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'chunks'
                    ORDER BY ordinal_position
                """)
                columns = {row[0]: row[1] for row in cur.fetchall()}
                
                # Verify required columns exist
                assert 'chunk_id' in columns, "chunks table missing chunk_id column"
                assert 'doc_id' in columns, "chunks table missing doc_id column"
                assert 'text' in columns, "chunks table missing text column"
                assert 'emb' in columns, "chunks table missing emb (embedding) column"
                assert 'lex' in columns, "chunks table missing lex (lexical) column"
                assert 'content_type' in columns, "chunks table missing content_type column"
                assert 'image_path' in columns, "chunks table missing image_path column"
                
                # Verify embedding column is vector type
                cur.execute("""
                    SELECT udt_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'chunks' 
                    AND column_name = 'emb'
                """)
                emb_type = cur.fetchone()
                assert emb_type is not None, "emb column type not found"
                assert emb_type[0] == 'vector', f"emb column should be vector type, got {emb_type[0]}"
                
                print("✅ chunks table exists with expected structure for multi-modal embeddings")
                
        except Exception as e:
            pytest.fail(f"chunks table check failed: {e}")
    
    @pytest.mark.skipif(
        not all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"),
            os.getenv("DB_PASS"),
            os.getenv("DB_NAME")
        ]),
        reason="Database connection environment variables not set"
    )
    def test_thread_tracking_table_exists(self):
        """Test that the thread_tracking table exists with expected structure."""
        try:
            with connect() as conn, conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'thread_tracking'
                """)
                result = cur.fetchone()
                assert result is not None, "thread_tracking table does not exist"
                
                # Check expected columns
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'thread_tracking'
                    ORDER BY ordinal_position
                """)
                columns = {row[0]: row[1] for row in cur.fetchall()}
                
                # Verify required columns exist
                assert 'id' in columns, "thread_tracking table missing id column"
                assert 'user_id' in columns, "thread_tracking table missing user_id column"
                assert 'thread_id' in columns, "thread_tracking table missing thread_id column"
                assert 'doc_ids' in columns, "thread_tracking table missing doc_ids column"
                assert 'query_text' in columns, "thread_tracking table missing query_text column"
                assert 'final_answer' in columns, "thread_tracking table missing final_answer column"
                assert 'graphstate' in columns, "thread_tracking table missing graphstate column"
                assert 'ingestion_meta' in columns, "thread_tracking table missing ingestion_meta column"
                assert 'created_at' in columns, "thread_tracking table missing created_at column"
                assert 'entry_point' in columns, "thread_tracking table missing entry_point column"
                assert 'pipeline_type' in columns, "thread_tracking table missing pipeline_type column"
                assert 'cross_doc' in columns, "thread_tracking table missing cross_doc column"
                assert 'metadata' in columns, "thread_tracking table missing metadata column"
                
                # Verify doc_ids is array type
                cur.execute("""
                    SELECT udt_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'thread_tracking' 
                    AND column_name = 'doc_ids'
                """)
                doc_ids_type = cur.fetchone()
                assert doc_ids_type is not None, "doc_ids column type not found"
                assert '_text' in doc_ids_type[0] or 'array' in doc_ids_type[0].lower(), \
                    f"doc_ids column should be array type, got {doc_ids_type[0]}"
                
                print("✅ thread_tracking table exists with expected structure")
                
        except Exception as e:
            pytest.fail(f"thread_tracking table check failed: {e}")
    
    @pytest.mark.skipif(
        not all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"),
            os.getenv("DB_PASS"),
            os.getenv("DB_NAME")
        ]),
        reason="Database connection environment variables not set"
    )
    def test_all_required_tables_exist(self):
        """Test that all three required tables exist after database initialization."""
        required_tables = ['documents', 'chunks', 'thread_tracking']
        
        try:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = ANY(%s)
                    ORDER BY table_name
                """, (required_tables,))
                
                existing_tables = [row[0] for row in cur.fetchall()]
                
                missing_tables = set(required_tables) - set(existing_tables)
                assert not missing_tables, \
                    f"Missing required tables: {', '.join(missing_tables)}"
                
                print(f"✅ All required tables exist: {', '.join(existing_tables)}")
                
        except Exception as e:
            pytest.fail(f"Required tables check failed: {e}")
    
    @pytest.mark.skipif(
        not all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"),
            os.getenv("DB_PASS"),
            os.getenv("DB_NAME")
        ]),
        reason="Database connection environment variables not set"
    )
    def test_chunks_table_indexes_exist(self):
        """Test that required indexes exist on the chunks table for efficient retrieval."""
        required_indexes = [
            'idx_chunks_lex',
            'idx_chunks_emb_hnsw',
            'idx_chunks_content_type'
        ]
        
        try:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'public' 
                    AND tablename = 'chunks'
                """)
                existing_indexes = [row[0] for row in cur.fetchall()]
                
                missing_indexes = set(required_indexes) - set(existing_indexes)
                assert not missing_indexes, \
                    f"Missing required indexes on chunks table: {', '.join(missing_indexes)}"
                
                print(f"✅ All required indexes exist on chunks table: {', '.join(existing_indexes)}")
                
        except Exception as e:
            pytest.fail(f"Chunks table indexes check failed: {e}")
    
    @pytest.mark.skipif(
        not all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"),
            os.getenv("DB_PASS"),
            os.getenv("DB_NAME")
        ]),
        reason="Database connection environment variables not set"
    )
    def test_thread_tracking_table_indexes_exist(self):
        """Test that required indexes exist on the thread_tracking table."""
        required_indexes = [
            'idx_thread_tracking_user_id',
            'idx_thread_tracking_thread_id',
            'idx_thread_tracking_doc_ids',
            'idx_thread_tracking_created_at',
            'idx_thread_tracking_entry_point',
            'idx_thread_tracking_pipeline_type',
            'idx_thread_tracking_user_thread'
        ]
        
        try:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'public' 
                    AND tablename = 'thread_tracking'
                """)
                existing_indexes = [row[0] for row in cur.fetchall()]
                
                missing_indexes = set(required_indexes) - set(existing_indexes)
                assert not missing_indexes, \
                    f"Missing required indexes on thread_tracking table: {', '.join(missing_indexes)}"
                
                print(f"✅ All required indexes exist on thread_tracking table: {', '.join(existing_indexes)}")
                
        except Exception as e:
            pytest.fail(f"Thread tracking table indexes check failed: {e}")

