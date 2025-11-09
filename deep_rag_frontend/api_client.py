"""
API Client for Deep RAG Backend
Centralized client for all API interactions
"""
import requests
import os
from typing import Optional, List, Dict, Any
from pathlib import Path


class DeepRAGClient:
    """Client for interacting with Deep RAG API"""
    
    def __init__(self, base_url: str = None, timeout: int = 30):
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.timeout = timeout
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            kwargs.setdefault("timeout", self.timeout)
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API Error: {str(e)}")
    
    # Health & Status
    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            result = self._request("GET", "/health")
            return result is not None
        except:
            return False
    
    # Thread Management
    def get_threads(self, user_id: str = None, limit: int = 100) -> List[Dict]:
        """Get list of all threads (requires GET /threads endpoint)"""
        # TODO: Implement when endpoint is available
        params = {}
        if user_id:
            params["user_id"] = user_id
        params["limit"] = limit
        return self._request("GET", "/threads", params=params) or []
    
    def get_thread(self, thread_id: str) -> Optional[Dict]:
        """Get thread details and messages (requires GET /threads/{thread_id} endpoint)"""
        # TODO: Implement when endpoint is available
        return self._request("GET", f"/threads/{thread_id}")
    
    def create_thread(self, title: str = None) -> Optional[Dict]:
        """Create a new thread (requires POST /threads endpoint)"""
        # TODO: Implement when endpoint is available
        data = {}
        if title:
            data["title"] = title
        return self._request("POST", "/threads", json=data)
    
    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread (requires DELETE /threads/{thread_id} endpoint)"""
        # TODO: Implement when endpoint is available
        try:
            self._request("DELETE", f"/threads/{thread_id}")
            return True
        except:
            return False
    
    # Document Management
    def get_documents(self, limit: int = 100) -> List[Dict]:
        """Get list of all documents (requires GET /documents endpoint)"""
        # TODO: Implement when endpoint is available
        # For now, try using diagnostics endpoint as fallback
        try:
            result = self._request("GET", "/documents", params={"limit": limit})
            return result.get("documents", [])
        except:
            # Fallback to diagnostics
            try:
                result = self._request("GET", "/diagnostics/document")
                if result and "documents" in result:
                    return result["documents"]
            except:
                pass
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document details (requires GET /documents/{doc_id} endpoint)"""
        # TODO: Implement when endpoint is available
        return self._request("GET", f"/documents/{doc_id}")
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document (requires DELETE /documents/{doc_id} endpoint)"""
        # TODO: Implement when endpoint is available
        try:
            self._request("DELETE", f"/documents/{doc_id}")
            return True
        except:
            return False
    
    # File Ingestion
    def ingest_file(self, file_path: str, title: str = None) -> Optional[Dict]:
        """Ingest a single file"""
        with open(file_path, "rb") as f:
            files = {"attachment": (Path(file_path).name, f, self._get_content_type(file_path))}
            data = {}
            if title:
                data["title"] = title
            return self._request("POST", "/ingest", files=files, data=data, timeout=60)
    
    def ingest_file_obj(self, file_obj, filename: str, title: str = None) -> Optional[Dict]:
        """Ingest a file object (from Streamlit upload)"""
        files = {"attachment": (filename, file_obj.read(), file_obj.type)}
        data = {}
        if title:
            data["title"] = title
        return self._request("POST", "/ingest", files=files, data=data, timeout=60)
    
    def ingest_batch(self, file_paths: List[str], titles: List[str] = None) -> List[Dict]:
        """Ingest multiple files (requires POST /ingest/batch endpoint)"""
        # TODO: Implement when endpoint is available
        # For now, use sequential ingestion
        results = []
        for idx, file_path in enumerate(file_paths):
            title = titles[idx] if titles and idx < len(titles) else None
            result = self.ingest_file(file_path, title)
            if result:
                results.append(result)
        return results
    
    # Querying
    def ask(self, question: str, doc_id: str = None, cross_doc: bool = False) -> Optional[Dict]:
        """Query using direct pipeline"""
        data = {
            "question": question,
            "cross_doc": cross_doc
        }
        if doc_id:
            data["doc_id"] = doc_id
        return self._request("POST", "/ask", json=data)
    
    def ask_graph(self, question: str, thread_id: str = "default", 
                  doc_id: str = None, cross_doc: bool = False) -> Optional[Dict]:
        """Query using LangGraph pipeline"""
        data = {
            "question": question,
            "thread_id": thread_id,
            "cross_doc": cross_doc
        }
        if doc_id:
            data["doc_id"] = doc_id
        return self._request("POST", "/ask-graph", json=data)
    
    def infer(self, question: str, file_path: str = None, title: str = None, 
              cross_doc: bool = False) -> Optional[Dict]:
        """Ingest + query using direct pipeline"""
        if file_path:
            with open(file_path, "rb") as f:
                files = {"attachment": (Path(file_path).name, f, self._get_content_type(file_path))}
                data = {
                    "question": question,
                    "cross_doc": cross_doc
                }
                if title:
                    data["title"] = title
                return self._request("POST", "/infer", files=files, data=data, timeout=120)
        else:
            data = {"question": question, "cross_doc": cross_doc}
            return self._request("POST", "/infer", data=data)
    
    def infer_graph(self, question: str, thread_id: str = "default",
                    file_path: str = None, file_obj=None, filename: str = None,
                    title: str = None, cross_doc: bool = False) -> Optional[Dict]:
        """Ingest + query using LangGraph pipeline"""
        if file_path:
            with open(file_path, "rb") as f:
                files = {"attachment": (Path(file_path).name, f, self._get_content_type(file_path))}
                data = {
                    "question": question,
                    "thread_id": thread_id,
                    "cross_doc": cross_doc
                }
                if title:
                    data["title"] = title
                return self._request("POST", "/infer-graph", files=files, data=data, timeout=120)
        elif file_obj:
            files = {"attachment": (filename, file_obj.read(), file_obj.type)}
            data = {
                "question": question,
                "thread_id": thread_id,
                "cross_doc": cross_doc
            }
            if title:
                data["title"] = title
            return self._request("POST", "/infer-graph", files=files, data=data, timeout=120)
        else:
            data = {
                "question": question,
                "thread_id": thread_id,
                "cross_doc": cross_doc
            }
            return self._request("POST", "/infer-graph", data=data)
    
    # Diagnostics
    def get_diagnostics(self, doc_title: str = None, doc_id: str = None) -> Optional[Dict]:
        """Get document diagnostics"""
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
        elif doc_title:
            params["doc_title"] = doc_title
        return self._request("GET", "/diagnostics/document", params=params)
    
    # Utilities
    def _get_content_type(self, file_path: str) -> str:
        """Get content type from file extension"""
        ext = Path(file_path).suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg"
        }
        return content_types.get(ext, "application/octet-stream")

