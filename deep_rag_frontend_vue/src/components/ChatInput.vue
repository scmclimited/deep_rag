<template>
  <div class="p-4 border-t border-dark-border bg-dark-bg">
    <!-- Single Ingestion Button (No Query) -->
    <div v-if="showIngestion" class="mb-4 p-4 bg-dark-surface rounded-xl border border-dark-border">
      <div class="flex items-center justify-between mb-3">
        <span class="text-sm font-medium">📤 Ingest Document (No Query)</span>
        <button
          @click="toggleIngestion"
          class="text-xs text-gray-400 hover:text-white transition-colors"
        >
          ✕ Close
        </button>
      </div>
      <div class="border-2 border-dashed border-dark-border rounded-lg p-6 text-center">
        <input
          type="file"
          ref="ingestFileInput"
          @change="handleIngestFileSelect"
          accept=".pdf,.txt,.png,.jpg,.jpeg"
          class="hidden"
        />
        <button
          @click="() => ingestFileInput?.click()"
          class="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg text-white font-medium transition-colors"
        >
          📄 Select File to Ingest
        </button>
        <p class="text-xs text-gray-400 mt-3">Limit 200MB per file • PDF, TXT, PNG, JPG, JPEG</p>
        <div v-if="ingestingFile" class="mt-3 text-sm text-blue-400">
          ⏳ Ingesting...
        </div>
        <div
          v-if="ingestionQueue.length > 0"
          class="mt-3 text-xs text-gray-400"
        >
          {{ store.ingesting ? 'Processing ingestion queue' : 'Queued ingestions' }} • {{ ingestionQueue.length }} file{{ ingestionQueue.length === 1 ? '' : 's' }} remaining
        </div>
      </div>
    </div>
    
    <!-- File Attachment Section (collapsible) -->
    <div v-if="showAttachment" class="mb-4 p-4 bg-dark-surface rounded-xl border border-dark-border">
      <div class="flex items-center justify-between mb-3">
        <span class="text-sm font-medium">📎 Attach file to next message</span>
        <button
          @click="toggleAttachment"
          class="text-xs text-gray-400 hover:text-white transition-colors"
        >
          ✕ Close
        </button>
      </div>
      <div class="border-2 border-dashed border-dark-border rounded-lg p-4 text-center">
        <input
          type="file"
          ref="fileInput"
          @change="handleFileSelect"
          accept=".pdf,.txt,.png,.jpg,.jpeg"
          class="hidden"
        />
        <button
          @click="() => fileInput?.click()"
          class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white transition-colors"
        >
          Browse files
        </button>
        <p class="text-xs text-gray-400 mt-2">Limit 200MB per file • PDF, TXT, PNG, JPG, JPEG</p>
      </div>
      <div v-if="store.pendingAttachment" class="mt-3 p-3 bg-dark-bg rounded-lg flex items-center justify-between">
        <span class="text-sm truncate flex-1">{{ store.pendingAttachment.name }}</span>
        <button
          @click="store.clearPendingAttachment()"
          class="text-xs text-red-400 hover:text-red-300 ml-2 transition-colors"
        >
          ✕ Clear
        </button>
      </div>
    </div>
    
    <!-- Input Field -->
    <div class="flex items-center gap-3">
      <button
        @click="toggleIngestion"
        class="p-2.5 hover:bg-dark-surface rounded-lg transition-colors"
        title="Ingest document (no query)"
      >
        📤
      </button>
      <button
        @click="toggleAttachment"
        class="p-2.5 hover:bg-dark-surface rounded-lg transition-colors"
        title="Attach file"
      >
        📎
      </button>
      <input
        v-model="inputText"
        type="text"
        :placeholder="isProcessing ? 'Thinking...' : 'Ask a question...'"
        class="flex-1 px-5 py-3.5 bg-dark-surface border border-dark-border rounded-xl text-white focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
        @keyup.enter="sendMessage"
      />
      <div v-if="isProcessing" class="flex items-center gap-2 text-sm text-blue-400 px-3">
        <span class="animate-pulse">⏳</span>
        <span>Thinking...</span>
      </div>
      <button
        @click="sendMessage"
        :disabled="isProcessing || store.ingesting || (!inputText.trim() && !store.pendingAttachment)"
        class="p-3.5 bg-blue-600 hover:bg-blue-700 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        title="Send message"
      >
        ➤
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useAppStore } from '../stores/app'
import { apiService } from '../services/api'

const store = useAppStore()
const inputText = ref('')
const showAttachment = ref(false)
const showIngestion = ref(false)
const isProcessing = ref(false)
const fileInput = ref(null)
const ingestFileInput = ref(null)
const ingestingFile = ref(false)
const ingestionQueue = ref([])

function toggleAttachment() {
  showAttachment.value = !showAttachment.value
  if (showAttachment.value) {
    showIngestion.value = false
  }
}

function toggleIngestion() {
  showIngestion.value = !showIngestion.value
  if (showIngestion.value) {
    showAttachment.value = false
  }
}

function handleFileSelect(event) {
  const file = event.target.files[0]
  if (file) {
    store.setPendingAttachment(file)
  }
}

async function handleIngestFileSelect(event) {
  const file = event.target.files[0]
  if (!file) return
  // Queue the ingestion request
  ingestionQueue.value.push({ file })
    if (ingestFileInput.value) {
      ingestFileInput.value.value = ''
    }
  processIngestionQueue()
}

async function processIngestionQueue() {
  if (store.ingesting) return
  const nextItem = ingestionQueue.value.shift()
  if (!nextItem) {
    ingestingFile.value = false
    return
  }

  ingestingFile.value = true
  store.ingesting = true
  try {
    await apiService.ingestFile(nextItem.file)
    store.loadDocuments()
  } catch (error) {
    const errorMessage = error.response?.data?.detail || error.message || 'Network Error'
    console.error('Error ingesting document:', errorMessage)
  } finally {
    store.ingesting = false
    if (ingestionQueue.value.length > 0) {
      await processIngestionQueue()
    } else {
    ingestingFile.value = false
      showIngestion.value = false
    }
  }
}

async function sendMessage() {
  if (store.ingesting) {
    alert('Please wait for the current ingestion to finish before sending a query.')
    return
  }

  if (!inputText.value.trim() && !store.pendingAttachment) return
  
  const question = inputText.value.trim()
  if (!question && !store.pendingAttachment) return
  
  const wasCrossDocQuery = !!store.crossDocSearch
  const previousSelectedDocIds = [...store.selectedDocIds]
  
  isProcessing.value = true
  
  // Store the question before clearing input
  const questionToSend = question || `[File: ${store.pendingAttachment?.name}]`
  let responseDocIds = []
  
  // Get or create thread ID FIRST (before adding message, as createNewThread clears messages)
  let threadId = store.currentThreadId
  if (!threadId) {
    threadId = await store.createNewThread()
  }
  
  // Add user message to chat IMMEDIATELY (after thread is created)
  // This ensures the message appears in the chat window right away
  const targetThreadId = threadId

  store.addMessage({
    role: 'user',
    content: questionToSend,
    timestamp: new Date().toISOString(),
    attachment: store.pendingAttachment?.name
  }, targetThreadId)
  
  // Clear input immediately after adding message to chat
  inputText.value = ''
  
  try {
    
    // Determine doc_ids to use for multi-document selection
    let selectedDocIds = []
    if (store.pendingAttachment) {
      // Will use doc_id from response
      selectedDocIds = []
    } else if (!store.crossDocSearch) {
      // When cross_doc=False, ONLY use selected_doc_ids (explicit user selection)
      // Don't fall back to activeDocIdsForThread - if user deselected, respect that
      selectedDocIds = store.selectedDocIds || []
    }
    // When cross_doc=True, selectedDocIds remains empty (search all)
    
    // Send request
    let response
    if (store.pendingAttachment) {
      // Determine doc_ids to use for multi-document selection (same logic as askGraph)
      let selectedDocIds = []
      if (!store.crossDocSearch) {
        // When cross_doc=False, ONLY use selected_doc_ids (explicit user selection)
        selectedDocIds = store.selectedDocIds || []
      }
      // When cross_doc=True, selectedDocIds remains empty (search all)
      
      response = await apiService.inferGraph(
        questionToSend,
        threadId,
        store.pendingAttachment,
        null,
        store.crossDocSearch,
        store.userId,  // Pass user_id for thread tracking
        selectedDocIds  // Pass selected doc_ids for multi-document selection
      )
      store.clearPendingAttachment()
      
      // DON'T auto-select documents after ingestion - let user explicitly select them
      // Only update activeDocIds if user has explicitly selected documents
      // This prevents auto-selection of documents that user hasn't selected
      
      // Auto-reload documents after submission
      store.loadDocuments()
    } else {
      // Use selectedDocIds for multi-document selection (not cross-doc)
      response = await apiService.askGraph(
        questionToSend,
        threadId,
        null,  // Don't use single docId, use selectedDocIds instead
        store.crossDocSearch,
        selectedDocIds,  // Pass selected doc_ids for multi-document selection
        store.userId  // Pass user_id for thread tracking
      )
    }
    
    responseDocIds = response.doc_ids || (response.doc_id ? [response.doc_id] : [])
    
    // Add assistant message with doc_ids and pages
    store.addMessage({
      role: 'assistant',
      content: response.answer || 'No answer received',
      timestamp: new Date().toISOString(),
      doc_id: response.doc_id,
      doc_ids: responseDocIds,
      doc_title: response.doc_title,
      pages: response.pages || [],
      confidence: response.confidence,
      action: response.action
    }, targetThreadId)
    
    if (
      !wasCrossDocQuery &&
      previousSelectedDocIds.length === 0 &&
      store.selectedDocIds.length === 0 &&
      responseDocIds.length > 0
    ) {
      store.setActiveDocIds(responseDocIds)
    }
    
    // Auto-reload documents after submission
    store.loadDocuments()
    
    // Refresh threads list after query (thread should now be in database)
    // The backend logs the thread synchronously before returning, so it should be available immediately
    if (store.userId) {
      // Refresh threads immediately - DB write completes before response is sent
      await store.loadThreads()
      // Also trigger Sidebar refresh if it's watching
      console.log('ChatInput: Threads refreshed after query')
    }
  } catch (error) {
    console.error('Error sending message:', error)
    const errorMessage = error.response?.data?.detail || error.message || 'Network Error'
    store.addMessage({
      role: 'assistant',
      content: `Error: ${errorMessage}`,
      timestamp: new Date().toISOString(),
      error: true
    }, targetThreadId)
  } finally {
    isProcessing.value = false
  }
}
</script>
