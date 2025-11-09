import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for file uploads
})

export const apiService = {
  // Health
  async healthCheck() {
    try {
      const response = await api.get('/health')
      return response.status === 200
    } catch {
      return false
    }
  },
  
  // Threads
  async getThreads(userId, limit = 100) {
    const response = await api.get('/threads', {
      params: { user_id: userId, limit }
    })
    return response.data
  },
  
  async getThread(threadId, userId) {
    const response = await api.get(`/threads/${threadId}`, {
      params: { user_id: userId }
    })
    return response.data
  },

  async seedThread(threadId, userId) {
    const response = await api.post('/threads', {
      thread_id: threadId,
      user_id: userId
    })
    return response.data
  },
  
  // Documents
  async getDocuments(limit = 100) {
    const response = await api.get('/documents', {
      params: { limit }
    })
    return response.data
  },
  
  async deleteDocument(docId) {
    const response = await api.delete(`/documents/${docId}`)
    return response.data
  },
  
  async getDiagnostics(docId) {
    const response = await api.get('/diagnostics/document', {
      params: { doc_id: docId }
    })
    return response.data
  },
  
  // Ingestion
  async ingestFile(file, title = null) {
    console.log('api.ingestFile: Starting ingestion for file:', file.name, 'size:', file.size)
    const formData = new FormData()
    formData.append('attachment', file)
    if (title) {
      formData.append('title', title)
    }
    // Use a longer timeout for ingestion (10 minutes) since it can take a while for large files
    const response = await api.post('/ingest', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 600000 // 10 minutes for ingestion
    })
    console.log('api.ingestFile: Ingestion completed successfully')
    return response.data
  },
  
  // Querying
  async askGraph(question, threadId, docId = null, crossDoc = false, selectedDocIds = [], userId = null) {
    console.log('api.askGraph: Called with userId=', userId)
    const data = {
      question,
      thread_id: threadId,
      cross_doc: crossDoc
    }
    // Add user_id if provided
    if (userId) {
      data.user_id = userId
      console.log('api.askGraph: Added user_id to request:', userId)
    } else {
      console.warn('api.askGraph: No userId provided, thread will be logged with default_user')
    }
    // Always send selected_doc_ids (even if empty) to make the user's intent explicit
    if (Array.isArray(selectedDocIds)) {
      data.selected_doc_ids = [...selectedDocIds]
    }
    // Only include doc_id fallback when explicitly provided (e.g. ingestion + query)
    if (docId) {
      data.doc_id = docId
    }
    const response = await api.post('/ask-graph', data)
    return response.data
  },
  
  async inferGraph(question, threadId, file = null, title = null, crossDoc = false, userId = null, selectedDocIds = []) {
    console.log('api.inferGraph: Called with userId=', userId)
    const formData = new FormData()
    formData.append('question', question)
    formData.append('thread_id', threadId)
    formData.append('cross_doc', crossDoc)
    if (userId) {
      formData.append('user_id', userId)
      console.log('api.inferGraph: Added user_id to request:', userId)
    } else {
      console.warn('api.inferGraph: No userId provided, thread will be logged with default_user')
    }
    // Send selected_doc_ids as JSON string (even if empty)
    if (selectedDocIds !== null && selectedDocIds !== undefined) {
      formData.append('selected_doc_ids', JSON.stringify(selectedDocIds))
    }
    if (file) {
      formData.append('attachment', file)
    }
    if (title) {
      formData.append('title', title)
    }
    const response = await api.post('/infer-graph', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  }
}

