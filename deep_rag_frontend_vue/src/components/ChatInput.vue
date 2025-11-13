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
          multiple
          class="hidden"
        />
        <button
          @click="() => ingestFileInput?.click()"
          class="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg text-white font-medium transition-colors"
        >
          📄 Select File(s) to Ingest
        </button>
        <p class="text-xs text-gray-400 mt-3">Limit 200MB per file • PDF, TXT, PNG, JPG, JPEG</p>
        <div v-if="ingestingFile" class="mt-3 text-sm text-blue-400">
          ⏳ Ingesting...
        </div>
        <div
          v-if="store.ingesting && currentIngestionItem"
          class="mt-2 px-3 py-2 bg-blue-900/20 border border-blue-500 rounded-lg text-xs text-blue-200 flex items-center gap-2"
        >
          <span class="animate-pulse">⏳</span>
          <span class="truncate flex-1">Ingesting: {{ currentIngestionItem.displayName }}</span>
        </div>
        <div
          v-if="ingestionQueue.length > 0"
          class="mt-2 space-y-2 text-left"
        >
          <div
            v-for="(item, index) in ingestionQueue"
            :key="`${item.file.name}-${item.file.size}-${item.file.lastModified}-${index}`"
            class="flex items-center justify-between px-3 py-2 bg-dark-bg rounded-lg border border-dark-border"
          >
            <span class="text-xs truncate flex-1">
              {{ item.displayName }}
            </span>
            <button
              @click="removeQueuedIngestion(index)"
              class="text-xs text-red-400 hover:text-red-300 ml-2 transition-colors"
            >
              ✕
            </button>
          </div>
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
          multiple
          data-attachment-input
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
      <div
        v-if="pendingAttachments.length > 0"
        class="mt-3 space-y-2"
      >
        <div
          v-for="(attachment, index) in pendingAttachments"
          :key="`${attachment.name}-${index}`"
          class="p-3 bg-dark-bg rounded-lg flex items-center justify-between border border-dark-border"
        >
          <span class="text-sm truncate flex-1">{{ attachment.name }}</span>
          <button
            @click="store.removePendingAttachment(index, props.threadId)"
            class="text-xs text-red-400 hover:text-red-300 ml-2 transition-colors"
          >
            ✕ Remove
          </button>
        </div>
        <button
          @click="store.clearPendingAttachments(props.threadId)"
          class="text-xs text-gray-400 hover:text-white transition-colors"
        >
          Clear all
        </button>
      </div>
    </div>
    
    <!-- Input Field -->
    <div class="flex items-end gap-3">
      <button
        @click="toggleIngestion"
        class="p-2.5 hover:bg-dark-surface rounded-lg transition-colors mb-1"
        title="Ingest document (no query)"
      >
        📤
      </button>
      <button
        @click="toggleAttachment"
        class="p-2.5 hover:bg-dark-surface rounded-lg transition-colors mb-1"
        title="Attach file"
      >
        📎
      </button>
      <textarea
        ref="textareaRef"
        v-model="inputText"
        :placeholder="isProcessing ? 'Thinking...' : 'Ask a question...'"
        class="flex-1 px-5 py-3.5 bg-dark-surface border border-dark-border rounded-xl text-white focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all resize-none overflow-hidden"
        @keydown.enter.exact.prevent="handleEnterKey"
        @input="autoResize"
        :disabled="isProcessing"
        rows="1"
        style="min-height: 52px; max-height: 200px;"
      />
      <div v-if="isProcessing" class="flex items-center gap-2 text-sm text-blue-400 px-3 mb-2">
        <span class="animate-pulse">⏳</span>
        <span>Thinking...</span>
      </div>
      <button
        @click="sendMessage"
        :disabled="!canSend"
        class="p-3.5 bg-blue-600 hover:bg-blue-700 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-colors mb-1"
        title="Send message"
      >
        ➤
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, defineProps, nextTick, onMounted } from 'vue'
import { useAppStore } from '../stores/app'
import { apiService } from '../services/api'

const props = defineProps({
  threadId: {
    type: String,
    required: true
  }
})

const store = useAppStore()
const inputText = ref('')
const showAttachment = ref(false)
const showIngestion = ref(false)
const fileInput = ref(null)
const ingestFileInput = ref(null)
const ingestingFile = ref(false)
const ingestionQueue = ref([])
const currentIngestionSignature = ref(null)
const currentIngestionItem = ref(null)
const textareaRef = ref(null)

// Use computed to maintain reactivity with store for THIS thread
const pendingAttachments = computed(() => {
  const attachments = store.pendingAttachmentsMap?.[props.threadId]
  return Array.isArray(attachments) ? attachments : []
})
const isProcessing = computed(() => {
  const thread = store.threads[props.threadId]
  return !!thread?.is_processing
})
const canSend = computed(() => {
  const hasQuestion = inputText.value.trim().length > 0
  if (!hasQuestion) return false
  if (isProcessing.value) return false
  if (store.ingesting) return false
  return true
})

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

function getFileSignature(file) {
  return `${file.name}|${file.size}|${file.lastModified}`
}

function formatDisplayName(file) {
  return file?.name || 'Untitled Document'
}

function formatDocumentTitle(file) {
  if (!file?.name) return 'Untitled Document'
  const withoutExtension = file.name.replace(/\.[^/.]+$/, '')
  const spaced = withoutExtension.replace(/[_-]+/g, ' ').trim()
  return spaced || withoutExtension || file.name
}

// Auto-resize textarea based on content
function autoResize() {
  if (!textareaRef.value) return
  
  // Reset height to auto to get the correct scrollHeight
  textareaRef.value.style.height = 'auto'
  
  // Set height to scrollHeight (content height) but respect max-height
  const newHeight = Math.min(textareaRef.value.scrollHeight, 200)
  textareaRef.value.style.height = `${newHeight}px`
}

// Initialize textarea height on mount
onMounted(() => {
  if (textareaRef.value) {
    autoResize()
  }
})

function handleEnterKey() {
  // Only send message if canSend is true (respects all validation logic)
  if (canSend.value) {
    sendMessage()
  } else {
    console.log('handleEnterKey: Cannot send message, canSend is false')
  }
}

async function handleFileSelect(event) {
  console.log('handleFileSelect called, event:', event)
  const files = Array.from(event.target.files || [])
  console.log('handleFileSelect: files selected:', files.length, files.map(f => f.name))
  if (!files.length) {
    console.warn('handleFileSelect: No files selected')
    return
  }
  // Use the thread ID from props (this component is bound to a specific thread)
  const threadId = props.threadId
  console.log('handleFileSelect: Using threadId from props:', threadId)
  console.log('handleFileSelect: Calling addPendingAttachments with', files.length, 'files for thread', threadId)
  const addedCount = store.addPendingAttachments(files, threadId)
  console.log('handleFileSelect: After addPendingAttachments, pendingAttachments:', pendingAttachments.value)
  
  // Notify user if duplicates were filtered out
  if (addedCount < files.length) {
    const duplicateCount = files.length - addedCount
    alert(`${duplicateCount} duplicate file(s) were not added. Each file can only be attached once.`)
  }
  
  if (event.target) {
    event.target.value = ''
  }
}

async function handleIngestFileSelect(event) {
  const rawFiles = Array.from(event.target?.files || []).filter(Boolean)
  if (rawFiles.length === 0) return

  const existingSignatures = new Set(
    ingestionQueue.value.map(item => getFileSignature(item.file))
  )
  if (currentIngestionSignature.value) {
    existingSignatures.add(currentIngestionSignature.value)
  }

  const uniqueNewFiles = rawFiles.filter(file => {
    const signature = getFileSignature(file)
    if (existingSignatures.has(signature)) {
      return false
    }
    existingSignatures.add(signature)
    return true
  })

  if (uniqueNewFiles.length < rawFiles.length) {
    const duplicateCount = rawFiles.length - uniqueNewFiles.length
    alert(`${duplicateCount} duplicate file(s) were not added. Each file can only be ingested once.`)
  }

  uniqueNewFiles.forEach(file => {
    ingestionQueue.value.push({
      file,
      displayName: formatDisplayName(file),
      docTitle: formatDocumentTitle(file)
    })
  })

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
    currentIngestionSignature.value = null
    currentIngestionItem.value = null
    return
  }

  ingestingFile.value = true
  store.ingesting = true
  currentIngestionSignature.value = getFileSignature(nextItem.file)
  currentIngestionItem.value = nextItem
  try {
    await apiService.ingestFile(nextItem.file, nextItem.docTitle)
    store.loadDocuments()
  } catch (error) {
    const errorMessage = error.response?.data?.detail || error.message || 'Network Error'
    console.error('Error ingesting document:', errorMessage)
  } finally {
    store.ingesting = false
    currentIngestionSignature.value = null
    currentIngestionItem.value = null
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

  const question = inputText.value.trim()
  const attachments = Array.isArray(pendingAttachments.value)
    ? [...pendingAttachments.value]
    : pendingAttachments.value
      ? [pendingAttachments.value]
      : []
  const hasAttachments = attachments.length > 0

  if (!question) {
    if (hasAttachments) {
      alert('Please add a question before sending an attachment.')
    }
    return
  }
  
  // Validate that user has documents available for querying
  // Skip validation if user has attachments (they're providing documents now)
  if (!hasAttachments) {
    const activeDocIdsForThread = Array.isArray(store.activeDocIdsForThread?.value)
      ? store.activeDocIdsForThread.value
      : []
    const hasSelectedDocs =
      (store.selectedDocIds && store.selectedDocIds.length > 0) ||
      activeDocIdsForThread.length > 0
    const crossDocEnabled = store.crossDocSearch
    const hasAnyDocs = store.documents && store.documents.length > 0
    
    // If cross-doc is disabled, no docs selected, and no attachments, reject immediately
    if (!crossDocEnabled && !hasSelectedDocs) {
      // Add the "no documents" message immediately without API call
      const noDocsMessage = "No documents selected. Choose a document from the sidebar, attach a document to your next message, or enable Cross-Document Search."
      
      store.addMessage({
        role: 'user',
        content: question,
        timestamp: new Date().toISOString()
      }, props.threadId)
      
      store.addMessage({
        role: 'assistant',
        content: noDocsMessage,
        timestamp: new Date().toISOString(),
        doc_ids: [],
        pages: [],
        confidence: 0,
        action: 'guidance'
      }, props.threadId)
      
      inputText.value = ''
      return
    }
  }
  
  const wasCrossDocQuery = !!store.crossDocSearch
  const previousSelectedDocIds = [...store.selectedDocIds]
  
  // Store the question before clearing input
  const questionToSend = question
  let responseDocIds = []
  
  // Use the thread ID from props (this component is bound to a specific thread)
  const threadId = props.threadId
  console.log('sendMessage: Using threadId from props:', threadId)
  store.setThreadProcessing(threadId, true)
  
  // Add user message to chat IMMEDIATELY (after thread is created)
  // This ensures the message appears in the chat window right away
  const targetThreadId = threadId

  store.addMessage({
    role: 'user',
    content: questionToSend,
    timestamp: new Date().toISOString(),
    attachments: hasAttachments ? attachments.map(file => file.name) : []
  }, targetThreadId)
  
  // Clear input immediately after adding message to chat
  inputText.value = ''
  
  // Reset textarea height after clearing input
  await nextTick()
  if (textareaRef.value) {
    textareaRef.value.style.height = 'auto'
    textareaRef.value.style.height = '52px' // Reset to min-height
  }
  
  try {
    // Determine doc_ids to use for multi-document selection
    let selectedDocIds = []
    if (!hasAttachments && !store.crossDocSearch) {
      // When cross_doc=False, ONLY use selected_doc_ids (explicit user selection)
      // Don't fall back to activeDocIdsForThread - if user deselected, respect that
      selectedDocIds = store.selectedDocIds || []
    }
    // When cross_doc=True, selectedDocIds remains empty (search all)

    // Send request
    let response
    if (hasAttachments) {
      let attachmentSelectedDocIds = []
      if (!store.crossDocSearch) {
        attachmentSelectedDocIds = store.selectedDocIds || []
      }
      response = await apiService.inferGraph(
        questionToSend,
        threadId,
        attachments,
        null,
        store.crossDocSearch,
        store.userId,  // Pass user_id for thread tracking
        attachmentSelectedDocIds  // Pass selected doc_ids for multi-document selection
      )
      store.clearPendingAttachments(targetThreadId)
      
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
    if ((!responseDocIds || responseDocIds.length === 0) && Array.isArray(response.uploaded_doc_ids)) {
      responseDocIds = [...response.uploaded_doc_ids]
    }
    
    // Add assistant message with doc_ids and pages
    store.addMessage({
      role: 'assistant',
      content: response.answer || 'No answer received',
      timestamp: new Date().toISOString(),
      doc_id: response.doc_id,
      doc_ids: responseDocIds,
      doc_title: response.doc_title,
      doc_titles: Array.isArray(response.doc_titles) ? [...response.doc_titles] : null,
      pages: response.pages || [],
      confidence: response.confidence,
      action: response.action
    }, targetThreadId)
    
    // Auto-reload documents after submission
    store.loadDocuments()
    
    // Refresh threads list after query (thread should now be in database)
    // The backend logs the thread synchronously before returning, so it should be available immediately
    if (store.userId) {
      // Refresh threads immediately - DB write completes before response is sent
      await store.loadThreads()
      // DON'T refresh thread history immediately after adding messages
      // The local messages are already in correct order with proper timestamps
      // Only refresh thread history when switching threads or loading old threads
      // This prevents messages from being reordered when backend timestamps differ slightly
      console.log('ChatInput: Threads refreshed after query (skipping thread history refresh to preserve order)')
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
    store.setThreadProcessing(targetThreadId, false)
  }
}

function removeQueuedIngestion(index) {
  if (store.ingesting && index === 0) return
  ingestionQueue.value.splice(index, 1)
}
</script>
