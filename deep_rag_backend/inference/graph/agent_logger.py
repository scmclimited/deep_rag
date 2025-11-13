# agent_logger.py
# Comprehensive logging system for agentic reasoning loop
# Logs queries, reasoning steps, decisions, and retrieval results
# Saves to CSV/TXT for future model training (SFT) and presentation

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import os

logger = logging.getLogger(__name__)

class AgentLogger:
    """
    Logger for agentic reasoning steps.
    
    Logs to both CSV (structured data) and TXT (human-readable)
    for different use cases:
    - CSV: For future model training, data analysis
    - TXT: For presentations, debugging, human review
    """
    
    def __init__(self, log_dir: Optional[Union[str, Path]] = None):
        # Detect if running in test environment
        # Check multiple ways to detect test environment:
        # 1. AGENT_LOG_TEST_MODE environment variable (explicit override)
        # 2. PYTEST_CURRENT_TEST environment variable (set by pytest)
        # 3. pytest module loaded in sys.modules
        # 4. Check if we're in a test directory or test file
        explicit_test_mode = os.getenv("AGENT_LOG_TEST_MODE", "").lower() in ("true", "1", "yes")
        pytest_env = os.getenv("PYTEST_CURRENT_TEST") is not None
        pytest_module = "pytest" in sys.modules
        cwd_has_test = "test" in str(Path.cwd()).lower()
        path_has_test = any("test" in str(p).lower() for p in Path.cwd().parts)
        
        is_test = explicit_test_mode or pytest_env or pytest_module or cwd_has_test or path_has_test
        self.is_test = is_test
        
        if log_dir is None:
            # Get the base directory (where this file is located)
            # This file is at: deep_rag_backend/inference/graph/agent_logger.py
            # So Path(__file__).parent gives us inference/graph/
            base_dir = Path(__file__).parent  # inference/graph/
            
            if is_test:
                # Use test logs directory when running tests
                log_dir_path = base_dir / "logs" / "test"
                logger.info(f"Test mode detected - using test logs directory: {log_dir_path}")
                logger.debug(f"Test detection: explicit={explicit_test_mode}, pytest_env={pytest_env}, pytest_module={pytest_module}, cwd_has_test={cwd_has_test}, path_has_test={path_has_test}")
            else:
                # Use dev logs directory for production/dev
                log_dir_path = base_dir / "logs" / "dev"
                logger.debug(f"Production/dev mode - using dev logs directory: {log_dir_path}")
        else:
            # If log_dir is provided, convert to Path
            log_dir_path = Path(log_dir) if isinstance(log_dir, str) else log_dir
        
        self.log_dir = log_dir_path.resolve()  # Resolve to absolute path
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Agent logger initialized - log_dir: {self.log_dir} (test_mode: {is_test})")
        
        # Create session-specific log files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"agent_log_{timestamp}.csv"
        self.txt_path = self.log_dir / f"agent_log_{timestamp}.txt"
        
        # Track if files have been initialized
        self._csv_initialized = False
        self._txt_initialized = False
        
        # Only initialize files immediately if not in test mode
        # In test mode, files will be initialized on first write to prevent empty files
        if not is_test:
            # Initialize CSV with headers
            self._initialize_csv()
            
            # Initialize TXT with header
            self._initialize_txt()
            
            logger.info(f"Agent logger initialized:")
            logger.info(f"  CSV: {self.csv_path}")
            logger.info(f"  TXT: {self.txt_path}")
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        self._csv_initialized = True
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'session_id',
                'node',
                'action',
                'question',
                'plan',
                'query',
                'num_chunks_retrieved',
                'pages_retrieved',
                'confidence',
                'iterations',
                'refinements',
                'answer',
                'metadata'
            ])
    
    def _initialize_txt(self):
        """Initialize TXT file with header."""
        self._txt_initialized = True
        with open(self.txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AGENT REASONING LOG\n")
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log_step(
        self,
        node: str,
        action: str,
        session_id: Optional[str] = None,
        question: Optional[str] = None,
        plan: Optional[str] = None,
        query: Optional[str] = None,
        num_chunks: Optional[int] = None,
        pages: Optional[List[int]] = None,
        confidence: Optional[float] = None,
        iterations: Optional[int] = None,
        refinements: Optional[List[str]] = None,
        answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a reasoning step to both CSV and TXT files.
        
        Args:
            node: Graph node name (planner, retriever, compressor, critic, synthesizer)
            action: Action taken (plan, retrieve, compress, evaluate, synthesize)
            session_id: Session/thread ID for tracking conversations
            question: User's question
            plan: Generated plan
            query: Query used for retrieval
            num_chunks: Number of chunks retrieved
            pages: List of page numbers retrieved
            confidence: Confidence score
            iterations: Current iteration count
            refinements: List of refinement queries
            answer: Final answer
            metadata: Additional metadata (scores, timings, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize CSV file if not already initialized (lazy initialization for tests)
        if not self._csv_initialized:
            self._initialize_csv()
        
        # Log to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                session_id or '',
                node,
                action,
                question or '',
                plan or '',
                query or '',
                num_chunks or 0,
                json.dumps(pages) if pages else '',
                confidence or 0.0,
                iterations or 0,
                json.dumps(refinements) if refinements else '',
                answer or '',
                json.dumps(metadata) if metadata else ''
            ])
        
        # Initialize TXT file if not already initialized (lazy initialization for tests)
        if not self._txt_initialized:
            self._initialize_txt()
        
        # Log to TXT (human-readable)
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {node.upper()} - {action}\n")
            f.write("-" * 80 + "\n")
            
            if session_id:
                f.write(f"Session ID: {session_id}\n")
            
            if question:
                f.write(f"Question: {question}\n\n")
            
            if plan:
                f.write(f"Plan:\n{plan}\n\n")
            
            if query:
                f.write(f"Query: {query}\n")
            
            if num_chunks is not None:
                f.write(f"Chunks Retrieved: {num_chunks}\n")
            
            if pages:
                f.write(f"Pages Retrieved: {sorted(set(pages))}\n")
            
            if confidence is not None:
                f.write(f"Confidence: {confidence:.2f}\n")
            
            if iterations is not None:
                f.write(f"Iterations: {iterations}\n")
            
            if refinements:
                f.write(f"Refinements:\n")
                for i, ref in enumerate(refinements, 1):
                    f.write(f"  {i}. {ref}\n")
                f.write("\n")
            
            if answer:
                f.write(f"Answer:\n{answer}\n\n")
            
            if metadata:
                f.write(f"Metadata: {json.dumps(metadata, indent=2)}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    def log_retrieval_details(
        self,
        session_id: str,
        query: str,
        chunks: List[Dict[str, Any]]
    ):
        """
        Log detailed retrieval results.
        
        Args:
            session_id: Session ID
            query: Query used
            chunks: List of retrieved chunks with scores
        """
        # Initialize TXT file if not already initialized (lazy initialization for tests)
        if not self._txt_initialized:
            self._initialize_txt()
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] RETRIEVAL DETAILS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Query: {query}\n")
            f.write(f"Results: {len(chunks)} chunks\n\n")
            
            for i, chunk in enumerate(chunks[:10], 1):  # Top 10
                f.write(f"[{i}] Chunk ID: {chunk.get('chunk_id', 'N/A')[:8]}...\n")
                f.write(f"    Pages: {chunk.get('p0', 'N/A')}-{chunk.get('p1', 'N/A')}\n")
                f.write(f"    Content Type: {chunk.get('content_type', 'N/A')}\n")
                f.write(f"    Scores: lex={chunk.get('lex', 0):.4f}, vec={chunk.get('vec', 0):.4f}, ce={chunk.get('ce', 0):.4f}\n")
                text_preview = chunk.get('text', '')[:200] if chunk.get('text') else 'N/A'
                f.write(f"    Text: {text_preview}...\n\n")
            
            if len(chunks) > 10:
                f.write(f"... and {len(chunks) - 10} more chunks\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    def log_error(self, node: str, error: str, session_id: Optional[str] = None):
        """Log an error that occurred during reasoning."""
        timestamp = datetime.now().isoformat()
        
        # Initialize CSV file if not already initialized (lazy initialization for tests)
        if not self._csv_initialized:
            self._initialize_csv()
        
        # Log to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                session_id or '',
                node,
                'ERROR',
                '', '', '', 0, '', 0.0, 0, '', '',
                json.dumps({'error': error})
            ])
        
        # Initialize TXT file if not already initialized (lazy initialization for tests)
        if not self._txt_initialized:
            self._initialize_txt()
        
        # Log to TXT
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR in {node.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Error: {error}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    def close(self):
        """Finalize the log files."""
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Agent logger closed. Logs saved to:")
        logger.info(f"  CSV: {self.csv_path}")
        logger.info(f"  TXT: {self.txt_path}")


# Global logger instance
_agent_logger: Optional[AgentLogger] = None

def get_agent_logger() -> AgentLogger:
    """Get or create the global agent logger instance."""
    global _agent_logger
    if _agent_logger is None:
        _agent_logger = AgentLogger()
    return _agent_logger

def reset_agent_logger():
    """Reset the global agent logger (creates new log files)."""
    global _agent_logger
    if _agent_logger:
        _agent_logger.close()
    _agent_logger = AgentLogger()
    return _agent_logger

