import { defineStore } from 'pinia'
import { ref, computed, watch, nextTick } from 'vue'
import axios from 'axios'
import { apiService } from '../services/api'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const useAppStore = defineStore('app', () => {
  // State
  const userId = ref(null)
  const userIdEntered = ref(false)
  const currentThreadId = ref(null)
  const threads = ref({}) // { threadId: { messages, created_at, active_doc_ids, cross_doc, selectionTouched, is_processing } }
  const messages = ref([]) // Current thread messages
  const documents = ref([])
  const selectedDocIds = ref([]) // Multi-document selection
  const crossDocSearch = ref(false)
  const processedDocIds = ref(new Set())
  const activeQueries = ref({}) // { threadId: { status, query } }
  const ingesting = ref(false)
  const pendingAttachmentsMap = ref({}) // { threadId: File[] }
  const scrollSignal = ref({ threadId: null, token: 0, forceInstant: true })
  
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
        selectionTouched: false,
        is_processing: false
      }
      if (!pendingAttachmentsMap.value[threadId]) {
        pendingAttachmentsMap.value[threadId] = []
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
    if (!pendingAttachmentsMap.value[threadId]) {
      pendingAttachmentsMap.value[threadId] = []
    }
    if (!Object.prototype.hasOwnProperty.call(existing, 'is_processing')) {
      existing.is_processing = false
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

  const pendingAttachments = computed(() => {
    if (!currentThreadId.value) return []
    const attachments = pendingAttachmentsMap.value[currentThreadId.value]
    return Array.isArray(attachments) ? attachments : []
  })

  const isProcessingCurrentThread = computed(() => {
    if (!currentThreadId.value) return false
    const thread = threads.value[currentThreadId.value]
    return !!thread?.is_processing
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
      selectionTouched: false,
      is_processing: false
    }
    pendingAttachmentsMap.value = {
      ...pendingAttachmentsMap.value,
      [newThreadId]: []
    }
    selectedDocIds.value = []
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
    // With KeepAlive and per-thread components, we don't need to sync messages
    // Each thread component reads directly from threads[threadId].messages
    // Just update the current thread ID and global UI state
    currentThreadId.value = threadId
    if (!threadId) {
      messages.value = []
      crossDocSearch.value = false
      selectedDocIds.value = []
      return
    }
    
    // Load thread's UI state (cross-doc toggle, selected docs)
    const threadEntry = ensureThreadEntry(threadId)
    const threadMessages = threadEntry?.messages || []
    messages.value = [...threadMessages] // Keep for backward compatibility
    crossDocSearch.value = !!threadEntry?.cross_doc

    if (threadEntry?.selectionTouched) {
      selectedDocIds.value = [...(threadEntry.active_doc_ids || [])]
    } else {
      selectedDocIds.value = []
    }

    // Emit scroll signal to ensure UI scrolls to latest message when switching threads
    scrollSignal.value = {
      threadId,
      token: Date.now(),
      forceInstant: true
    }
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
    
    // Ensure new message has a timestamp that's later than the last message
    // This prevents new messages from appearing at the top
    let messageTimestamp = message.timestamp || message.created_at || new Date().toISOString()
    if (existing.length > 0) {
      // Get the last message's timestamp
      const lastMessage = existing[existing.length - 1]
      const lastTimestamp = new Date(lastMessage.timestamp || lastMessage.created_at || 0).getTime()
      const newTimestamp = new Date(messageTimestamp).getTime()
      
      // If new message timestamp is not later than last message, ensure it's at least 1ms later
      // This ensures stable sorting even when messages are created in quick succession
      if (newTimestamp <= lastTimestamp) {
        const now = Date.now()
        // Ensure the new timestamp is at least 1ms after the last message
        const minTimestamp = lastTimestamp + 1
        messageTimestamp = new Date(Math.max(now, minTimestamp)).toISOString()
        console.log(`addMessage: Adjusted timestamp for ${message.role} message to ensure it's after last message (${messageTimestamp})`)
      }
    }
    
    const messagePayload = {
      ...message,
      // Ensure timestamp is always set and is a valid ISO string, and is later than existing messages
      timestamp: messageTimestamp,
      attachments: Array.isArray(message.attachments)
        ? [...message.attachments]
        : message.attachments
    }
    
    // Since we ensure the new message timestamp is always later than the last message,
    // we can simply append it to the end without sorting
    // This maintains the correct order and prevents temporary reordering
    const updated = [...existing, messagePayload]
    
    // Only sort if there might be out-of-order messages (e.g., from backend merge)
    // For new messages, we know they're in order, so sorting is unnecessary
    // But we sort anyway to handle edge cases where messages might have been loaded out of order
    updated.sort((a, b) => {
      const timeA = new Date(a.timestamp || a.created_at || 0).getTime()
      const timeB = new Date(b.timestamp || b.created_at || 0).getTime()
      // If timestamps are equal, preserve insertion order (newer messages come after)
      if (timeA === timeB) {
        return 0
      }
      return timeA - timeB
    })
    
    console.log(`addMessage: Added ${messagePayload.role} message to thread ${targetThreadId.substring(0, 8)}... at ${messagePayload.timestamp}, total messages: ${updated.length}`)
    
    threadEntry.messages = updated
    if (currentThreadId.value === targetThreadId) {
      messages.value = [...updated]
    }

    // Emit scroll signal so UI can scroll to the latest message immediately
    scrollSignal.value = {
      threadId: targetThreadId,
      token: Date.now(),
      forceInstant: message.role === 'user'
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
          // Don't auto-select documents from thread history
          // Users must explicitly select documents they want to query
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
  
  function resetAttachmentInput() {
    if (typeof document === 'undefined') return
    const input = document.querySelector('input[type="file"][data-attachment-input]')
    if (input) {
      input.value = ''
    }
  }

  function addPendingAttachments(files, threadId = currentThreadId.value) {
    console.log('addPendingAttachments called:', { files, threadId, filesCount: files?.length })
    if (!threadId) {
      console.warn('addPendingAttachments: No threadId provided')
      return
    }
    ensureThreadEntry(threadId)
    const incoming = Array.isArray(files) ? files : [files]
    const normalized = incoming.filter(Boolean)
    console.log('addPendingAttachments: normalized files:', normalized.length, normalized.map(f => f.name))
    
    const current = Array.isArray(pendingAttachmentsMap.value[threadId])
      ? pendingAttachmentsMap.value[threadId]
      : []
    
    // Create a file signature for deduplication (name + size + lastModified)
    const getFileSignature = (file) => `${file.name}|${file.size}|${file.lastModified}`
    const currentSignatures = new Set(current.map(getFileSignature))
    
    // Filter out duplicates
    const uniqueNewFiles = normalized.filter(file => {
      const signature = getFileSignature(file)
      if (currentSignatures.has(signature)) {
        console.warn('addPendingAttachments: Duplicate file detected, skipping:', file.name)
        return false
      }
      return true
    })
    
    if (uniqueNewFiles.length < normalized.length) {
      const duplicateCount = normalized.length - uniqueNewFiles.length
      console.log(`addPendingAttachments: Filtered out ${duplicateCount} duplicate file(s)`)
    }
    
    pendingAttachmentsMap.value[threadId] = [...current, ...uniqueNewFiles]
    console.log('addPendingAttachments: Updated map for thread', threadId, ':', pendingAttachmentsMap.value[threadId].length, 'files')
    nextTick(resetAttachmentInput)
    
    return uniqueNewFiles.length
  }

  function removePendingAttachment(index, threadId = currentThreadId.value) {
    if (!threadId) return
    ensureThreadEntry(threadId)
    const current = Array.isArray(pendingAttachmentsMap.value[threadId])
      ? pendingAttachmentsMap.value[threadId]
      : []
    if (index < 0 || index >= current.length) return
    pendingAttachmentsMap.value[threadId] = [
      ...current.slice(0, index),
      ...current.slice(index + 1)
    ]
    nextTick(resetAttachmentInput)
  }

  function clearPendingAttachments(threadId = currentThreadId.value) {
    if (!threadId) return
    ensureThreadEntry(threadId)
    pendingAttachmentsMap.value[threadId] = []
    nextTick(resetAttachmentInput)
  }

  function setThreadProcessing(threadId, value) {
    if (!threadId) return
    const threadEntry = ensureThreadEntry(threadId)
    if (!threadEntry) return
    threadEntry.is_processing = value
  }

  function isThreadProcessing(threadId = currentThreadId.value) {
    if (!threadId) return false
    const threadEntry = threads.value[threadId]
    return !!threadEntry?.is_processing
  }
  
  async function refreshThreadHistory(threadId = currentThreadId.value) {
    if (!threadId || !userId.value) return
    const threadEntry = threads.value[threadId]
    if (!threadEntry || threadEntry.persisted === false) {
      // Not a persisted thread, no need to refresh
      return
    }
    try {
      const result = await apiService.getThread(threadId, userId.value)
      if (result?.messages !== undefined) {
        const localMessages = threadEntry.messages || []
        const backendMessages = result.messages || []
        
        // Find the latest backend message timestamp
        const latestBackendTime = backendMessages.length > 0
          ? Math.max(...backendMessages.map(msg => new Date(msg.timestamp || msg.created_at || 0).getTime()))
          : 0
        
        // Keep local messages that are newer than the latest backend message
        // These are messages that were just added but not yet persisted
        const newLocalMessages = localMessages.filter(msg => {
          const msgTime = new Date(msg.timestamp || msg.created_at || 0).getTime()
          return msgTime > latestBackendTime
        })
        
        // Merge: backend messages (source of truth) + new local messages (not yet persisted)
        // Use a stable sort that preserves order for messages with identical timestamps
        const mergedMessages = [...backendMessages, ...newLocalMessages].sort((a, b) => {
          const timeA = new Date(a.timestamp || a.created_at || 0).getTime()
          const timeB = new Date(b.timestamp || b.created_at || 0).getTime()
          // If timestamps are equal, preserve insertion order (backend messages first, then local)
          if (timeA === timeB) {
            // If both are in backend or both are local, preserve their relative order
            const aInBackend = backendMessages.some(bm => 
              (bm.timestamp || bm.created_at) === (a.timestamp || a.created_at) &&
              bm.content === a.content
            )
            const bInBackend = backendMessages.some(bm => 
              (bm.timestamp || bm.created_at) === (b.timestamp || b.created_at) &&
              bm.content === b.content
            )
            // Backend messages come before local messages with same timestamp
            if (aInBackend && !bInBackend) return -1
            if (!aInBackend && bInBackend) return 1
            return 0
          }
          return timeA - timeB
        })
        
        console.log(`refreshThreadHistory: Refreshed thread ${threadId.substring(0, 8)}... - ${backendMessages.length} backend + ${newLocalMessages.length} new local = ${mergedMessages.length} total`)
        
        // Update thread with merged messages
        threadEntry.messages = mergedMessages
        if (currentThreadId.value === threadId) {
          messages.value = [...mergedMessages]
          // After loading history, ensure UI scrolls to the latest message
          scrollSignal.value = {
            threadId,
            token: Date.now(),
            forceInstant: true
          }
        }
      }
    } catch (error) {
      console.error('Error refreshing thread history:', error)
    }
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
    processedDocIds,
    activeQueries,
    ingesting,
    pendingAttachmentsMap, // Export raw map for per-thread access
    scrollSignal,
    // Computed
    currentThread,
    activeDocIdsForThread,
    pendingAttachments,
    isProcessingCurrentThread,
    // Actions
    setUserId,
    createNewThread,
    switchThread,
    addMessage,
    loadThreads,
    loadDocuments,
    deleteDocument,
    setActiveDocIds,
    addPendingAttachments,
    removePendingAttachment,
    clearPendingAttachments,
    setThreadProcessing,
    isThreadProcessing,
    refreshThreadHistory,
    markDocProcessed,
    isDocProcessed
  }
})

