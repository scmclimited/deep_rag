import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import axios from 'axios'
import { apiService } from '../services/api'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const useAppStore = defineStore('app', () => {
  // State
  const userId = ref(null)
  const userIdEntered = ref(false)
  const currentThreadId = ref(null)
  const threads = ref({}) // { threadId: { messages, created_at, active_doc_ids, cross_doc, selectionTouched } }
  const messages = ref([]) // Current thread messages
  const documents = ref([])
  const selectedDocIds = ref([]) // Multi-document selection
  const crossDocSearch = ref(false)
  const pendingAttachment = ref(null)
  const processedDocIds = ref(new Set())
  const activeQueries = ref({}) // { threadId: { status, query } }
  const ingesting = ref(false)
  
  function ensureThreadEntry(threadId) {
    if (!threadId) return null
    const existing = threads.value[threadId]
    if (!existing) {
      threads.value[threadId] = {
        messages: [],
        created_at: new Date().toISOString(),
        active_doc_ids: [],
        persisted: false,
        cross_doc: false,
        selectionTouched: false
      }
      return threads.value[threadId]
    }
    if (!Array.isArray(existing.messages)) {
      existing.messages = []
    }
    if (!Array.isArray(existing.active_doc_ids)) {
      existing.active_doc_ids = []
    }
    if (typeof existing.cross_doc !== 'boolean') {
      existing.cross_doc = !!existing.cross_doc
    }
    if (typeof existing.selectionTouched !== 'boolean') {
      existing.selectionTouched = false
    }
    if (typeof existing.persisted !== 'boolean') {
      existing.persisted = !!existing.persisted
    }
    return existing
  }
  
  // Computed
  const currentThread = computed(() => {
    if (!currentThreadId.value) return null
    return threads.value[currentThreadId.value] || null
  })
  
  const activeDocIdsForThread = computed(() => {
    if (!currentThreadId.value) return []
    const thread = threads.value[currentThreadId.value]
    return thread?.active_doc_ids || []
  })
  
  // Actions
  async function setUserId(id) {
    userId.value = id
    userIdEntered.value = true
    await loadThreads()
  }
  
  async function createNewThread() {
    const newThreadId = crypto.randomUUID()
    currentThreadId.value = newThreadId
    messages.value = []
    threads.value[newThreadId] = {
      messages: [],
      created_at: new Date().toISOString(),
      active_doc_ids: [],
      persisted: false,
      cross_doc: crossDocSearch.value,
      selectionTouched: false
    }
    selectedDocIds.value = []
    pendingAttachment.value = null
    if (userId.value) {
      try {
        const result = await apiService.seedThread(newThreadId, userId.value)
        if (result?.status === 'seeded' || result?.status === 'exists') {
          threads.value[newThreadId].persisted = true
        }
      } catch (error) {
        console.error('createNewThread: Failed to seed thread:', error)
      }
    }
    return newThreadId
  }
  
  function switchThread(threadId) {
    currentThreadId.value = threadId
    if (!threadId) {
      messages.value = []
      crossDocSearch.value = false
      selectedDocIds.value = []
      pendingAttachment.value = null
      return
    }
    const threadEntry = ensureThreadEntry(threadId)
    messages.value = [...(threadEntry?.messages || [])]
    crossDocSearch.value = !!threadEntry?.cross_doc
    selectedDocIds.value = [...(threadEntry?.active_doc_ids || [])]
    pendingAttachment.value = null
  }
  
  function addMessage(message, targetThreadId = currentThreadId.value) {
    if (!targetThreadId) {
      return
    }
    const threadEntry = ensureThreadEntry(targetThreadId)
    if (!threadEntry) {
      return
    }
    threadEntry.cross_doc = crossDocSearch.value
    const existing = threadEntry.messages || []
    const updated = [...existing, message]
    threadEntry.messages = updated
    if (currentThreadId.value === targetThreadId) {
      messages.value = [...updated]
    }
  }
  
  async function loadThreads() {
    if (!userId.value) return
    try {
      const response = await axios.get(`${API_BASE_URL}/threads`, {
        params: { user_id: userId.value, limit: 100 }
      })
      console.log('Store loadThreads: API response:', response.data)
      if (response.data && Array.isArray(response.data.threads)) {
        // Convert API threads to local format
        response.data.threads.forEach(thread => {
          const threadId = thread.thread_id
          if (!threadId) {
            return
          }
          const threadEntry = ensureThreadEntry(threadId)
          threadEntry.persisted = true
          threadEntry.created_at = threadEntry.created_at || thread.created_at
          if (
            !threadEntry.selectionTouched &&
            Array.isArray(threadEntry.active_doc_ids) &&
            threadEntry.active_doc_ids.length === 0
          ) {
            const docIdsFromThread = Array.isArray(thread.doc_ids) ? thread.doc_ids : []
            if (docIdsFromThread.length > 0) {
              threadEntry.active_doc_ids = [...docIdsFromThread]
              if (
                currentThreadId.value === threadId &&
                selectedDocIds.value.length === 0
              ) {
                selectedDocIds.value = [...docIdsFromThread]
              }
            }
          }
        })
        console.log('Store loadThreads: Updated', response.data.threads.length, 'threads in store')
      } else {
        console.log('Store loadThreads: No threads in response or invalid format')
      }
    } catch (error) {
      if (error.response?.status !== 404) {
        console.error('Store loadThreads: Error loading threads:', error)
        console.error('Store loadThreads: Error details:', error.response?.data || error.message)
      }
    }
  }
  
  async function loadDocuments() {
    try {
      const response = await axios.get(`${API_BASE_URL}/documents`, {
        params: { limit: 100 }
      })
      if (response.data?.documents) {
        documents.value = response.data.documents
      }
    } catch (error) {
      console.error('Error loading documents:', error)
    }
  }
  
  async function deleteDocument(docId) {
    try {
      await axios.delete(`${API_BASE_URL}/documents/${docId}`)
      documents.value = documents.value.filter(doc => doc.doc_id !== docId)
      return true
    } catch (error) {
      console.error('Error deleting document:', error)
      throw error
    }
  }
  
  function setActiveDocIds(docIds) {
    if (currentThreadId.value) {
      const threadEntry = ensureThreadEntry(currentThreadId.value)
      if (threadEntry) {
        threadEntry.active_doc_ids = [...docIds]
        threadEntry.selectionTouched = true
      }
    }
    selectedDocIds.value = [...docIds]
  }
  
  function setPendingAttachment(file) {
    pendingAttachment.value = file
  }
  
  function clearPendingAttachment() {
    pendingAttachment.value = null
  }
  
  function markDocProcessed(docId) {
    processedDocIds.value.add(docId)
  }
  
  function isDocProcessed(docId) {
    return processedDocIds.value.has(docId)
  }

  watch(crossDocSearch, (value) => {
    const threadId = currentThreadId.value
    if (!threadId) return
    const threadEntry = ensureThreadEntry(threadId)
    if (threadEntry) {
      threadEntry.cross_doc = value
    }
  })
  
  return {
    // State
    userId,
    userIdEntered,
    currentThreadId,
    threads,
    messages,
    documents,
    selectedDocIds,
    crossDocSearch,
    pendingAttachment,
    processedDocIds,
    activeQueries,
    ingesting,
    // Computed
    currentThread,
    activeDocIdsForThread,
    // Actions
    setUserId,
    createNewThread,
    switchThread,
    addMessage,
    loadThreads,
    loadDocuments,
    deleteDocument,
    setActiveDocIds,
    setPendingAttachment,
    clearPendingAttachment,
    markDocProcessed,
    isDocProcessed
  }
})

