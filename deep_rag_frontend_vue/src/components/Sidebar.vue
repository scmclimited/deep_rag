<template>
  <div 
    class="sidebar bg-dark-surface border-r border-dark-border flex flex-col h-screen fixed left-0 top-0 overflow-y-auto z-50 transition-all duration-300"
    :class="expanded ? 'sidebar-expanded' : 'sidebar-collapsed'"
  >
    <!-- Close button for mobile -->
    <button
      v-if="showMobileClose"
      @click="$emit('close')"
      class="absolute top-2 right-2 p-2 text-gray-400 hover:text-white md:hidden"
      title="Close sidebar"
    >
      ✕
    </button>
    <!-- User ID Input (if not entered) -->
    <div v-if="!store.userIdEntered" class="p-4">
      <h2 class="text-xl font-bold mb-4">🧠 Deep RAG</h2>
      <p class="text-sm text-gray-400 mb-4">Enter your User ID to continue</p>
      <input
        v-model="userIdInput"
        type="text"
        placeholder="Enter your user ID"
        class="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-white mb-2"
        @keyup.enter="handleUserIdSubmit"
      />
      <button
        @click="handleUserIdSubmit"
        class="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white"
      >
        Continue
      </button>
    </div>
    
    <!-- Main Sidebar Content -->
    <div v-else class="flex flex-col h-full">
      <!-- Threads Section -->
      <div class="p-4 border-b border-dark-border">
        <h3 class="font-semibold mb-2">💬 Threads</h3>
        <button
          @click="handleCreateNewThread"
          class="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm mb-2"
        >
          + New Thread
        </button>
        <div class="text-xs text-gray-400 mb-2">
          Current: {{ store.currentThreadId?.substring(0, 8) }}...
        </div>
        <div class="text-xs text-gray-400 mb-2">
          {{ threadsList.length }} threads
        </div>
        <div v-if="threadsList.length === 0" class="text-xs text-gray-500 p-2 italic">
          No threads found. Create a new thread or make a query to start.
        </div>
        <div v-else class="space-y-1 max-h-48 overflow-y-auto">
          <div
            v-for="thread in threadsList"
            :key="thread.thread_id"
            @click="switchToThread(thread.thread_id)"
            class="p-2 bg-dark-bg rounded border border-dark-border hover:border-blue-500 cursor-pointer text-xs transition-colors relative"
            :class="{ 'border-blue-500 bg-blue-900/20': thread.thread_id === store.currentThreadId }"
          >
            <div class="flex items-center justify-between">
              <div class="font-medium truncate flex-1">{{ thread.thread_id.substring(0, 8) }}...</div>
              <!-- Status indicator -->
              <div v-if="isThreadProcessing(thread.thread_id)" class="flex items-center gap-1 ml-2">
                <span class="animate-pulse text-blue-400">⏳</span>
                <span class="text-blue-400 text-xs">Thinking...</span>
              </div>
              <div v-else class="flex items-center gap-1 ml-2">
                <span class="text-green-400">✓</span>
                <span class="text-green-400 text-xs">Clear</span>
              </div>
            </div>
            <div class="text-gray-400 text-xs mt-1 truncate">{{ thread.latest_query || 'No messages' }}</div>
          </div>
        </div>
      </div>
      
      <!-- Documents Section -->
      <div class="p-4 border-b border-dark-border flex-1 overflow-y-auto">
        <div class="flex items-center justify-between mb-2">
          <h3 class="font-semibold">📚 Documents</h3>
          <button
            @click="refreshDocuments"
            class="text-xs text-blue-400 hover:text-blue-300"
            title="Refresh documents list"
          >
            🔄
          </button>
        </div>
        <div v-if="store.documents.length === 0" class="text-xs text-gray-400">
          No documents found
        </div>
        <div v-else class="space-y-2">
          <div
            v-for="doc in store.documents"
            :key="doc.doc_id"
            class="p-2 bg-dark-bg rounded border border-dark-border hover:border-blue-500 transition-colors"
            :class="{ 'border-blue-500 bg-blue-900/20': isDocActive(doc.doc_id) }"
          >
            <div class="flex items-center justify-between mb-1">
              <span class="text-sm font-medium truncate flex-1" :title="doc.title">{{ doc.title }}</span>
              <input
                type="checkbox"
                :checked="isDocActive(doc.doc_id)"
                @change="toggleDocSelection(doc.doc_id)"
                class="ml-2 cursor-pointer"
                :class="{ 'ring-2 ring-blue-500': isDocActive(doc.doc_id) }"
              />
            </div>
            <div class="text-xs text-gray-400 font-mono truncate" :title="doc.doc_id">{{ doc.doc_id }}</div>
            <div class="flex flex-wrap gap-2 mt-2">
              <button
                @click.stop="toggleDiagnostics(doc.doc_id)"
                class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors flex-1 min-w-[110px]"
                title="View diagnostics"
              >
                🔍 Diagnostics
              </button>
              <button
                @click.stop="handleDeleteDocument(doc.doc_id)"
                class="text-xs px-2 py-1 bg-red-700 hover:bg-red-600 rounded transition-colors flex-1 min-w-[110px]"
                title="Delete document"
              >
                🗑️ Delete
              </button>
            </div>
            <!-- Diagnostics Display (like Streamlit expander) -->
            <div v-if="expandedDiagnostics[doc.doc_id]" class="mt-2 pt-2 border-t border-dark-border">
              <div v-if="!diagnosticsData[doc.doc_id]" class="text-xs text-gray-400">
                Click "🔍 Diagnostics" to load...
              </div>
              <div v-else-if="diagnosticsData[doc.doc_id].loading" class="text-xs text-blue-400">
                ⏳ Loading diagnostics...
              </div>
              <div v-else-if="diagnosticsData[doc.doc_id].error" class="text-xs text-red-400">
                ❌ Error: {{ diagnosticsData[doc.doc_id].error }}
              </div>
              <div v-else-if="diagnosticsData[doc.doc_id].data" class="text-xs">
                <div class="bg-dark-bg rounded p-2 max-h-64 overflow-y-auto">
                  <pre class="text-xs text-gray-300 whitespace-pre-wrap break-words">{{ JSON.stringify(diagnosticsData[doc.doc_id].data, null, 2) }}</pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Settings Section -->
      <div class="p-4 border-t border-dark-border">
        <h3 class="font-semibold mb-2">⚙️ Settings</h3>
        <label class="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            v-model="store.crossDocSearch"
            class="rounded cursor-pointer"
          />
          Cross-Document Search
        </label>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { useAppStore } from '../stores/app'
import { apiService } from '../services/api'

const props = defineProps({
  expanded: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['close'])

const store = useAppStore()
const userIdInput = ref('')
const threadList = ref([])
const diagnosticsData = ref({}) // { docId: { data, loading, error } }
const expandedDiagnostics = ref({}) // { docId: true/false }

const showMobileClose = computed(() => {
  return window.innerWidth < 768 && props.expanded
})

// Computed property for thread list
function getLatestUserMessage(messages = []) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') {
      return messages[i]
    }
  }
  return null
}

function getLastMessage(messages = []) {
  return messages.length > 0 ? messages[messages.length - 1] : null
}

function buildThreadSummary(threadId, threadData, isPersisted = false) {
  const messages = Array.isArray(threadData?.messages) ? threadData.messages : []
  const latestUserMessage = getLatestUserMessage(messages)
  const lastMessage = getLastMessage(messages)
  const createdAt = threadData?.created_at || lastMessage?.timestamp || new Date().toISOString()

  return {
    thread_id: threadId,
    user_id: store.userId || null,
    created_at: createdAt,
    last_activity: lastMessage?.timestamp || createdAt,
    latest_query: latestUserMessage?.content || (isPersisted ? 'No messages yet' : 'New thread'),
    query_count: messages.filter(msg => msg.role === 'user').length,
    doc_ids: Array.isArray(threadData?.active_doc_ids) ? threadData.active_doc_ids : [],
    isPersisted
  }
}

const threadsList = computed(() => {
  const merged = new Map()

  // Seed with threads returned from the API (persisted threads)
  threadList.value.forEach(thread => {
    if (thread?.thread_id) {
      const summary = buildThreadSummary(thread.thread_id, thread, true)
      merged.set(thread.thread_id, summary)
      if (!store.threads[thread.thread_id]) {
        store.threads[thread.thread_id] = {
          messages: [],
          created_at: thread.created_at,
          active_doc_ids: []
        }
      }
    }
  })

  // Merge in locally created threads and latest metadata from the store
  Object.entries(store.threads || {}).forEach(([threadId, threadData]) => {
    const summary = buildThreadSummary(threadId, threadData)

    if (merged.has(threadId)) {
      const existing = merged.get(threadId)
      existing.last_activity = summary.last_activity
      existing.latest_query = summary.latest_query
      existing.query_count = Math.max(existing.query_count || 0, summary.query_count)
      existing.doc_ids =
        summary.doc_ids.length > 0 ? summary.doc_ids : existing.doc_ids || []
      existing.created_at = existing.created_at || summary.created_at
    } else {
      merged.set(threadId, summary)
    }
  })

  return Array.from(merged.values()).sort((a, b) => {
    const dateA = new Date(a.last_activity || a.created_at || 0)
    const dateB = new Date(b.last_activity || b.created_at || 0)
    return dateB.getTime() - dateA.getTime()
  })
})

async function handleUserIdSubmit() {
  if (userIdInput.value.trim()) {
    await store.setUserId(userIdInput.value.trim())
  }
}

async function loadThreads() {
  if (!store.userId) {
    console.log('loadThreads: No userId, skipping')
    return
  }
  try {
    console.log('loadThreads: Loading threads for userId:', store.userId)
    const result = await apiService.getThreads(store.userId)
    console.log('loadThreads: API result:', result)
    console.log('loadThreads: Threads count:', result?.count, 'Threads array length:', result?.threads?.length)
    
    // Always update threadList, even if empty
    if (result && Array.isArray(result.threads)) {
      threadList.value = result.threads
      console.log('loadThreads: Loaded', result.threads.length, 'threads')
      // Also update store threads
      result.threads.forEach(thread => {
        if (!store.threads[thread.thread_id]) {
          store.threads[thread.thread_id] = {
            messages: [],
            created_at: thread.created_at,
            active_doc_ids: [],
            persisted: true,
            cross_doc: false,
            selectionTouched: false
          }
        } else {
          store.threads[thread.thread_id].persisted = true
          store.threads[thread.thread_id].created_at =
            store.threads[thread.thread_id].created_at || thread.created_at
          if (typeof store.threads[thread.thread_id].cross_doc !== 'boolean') {
            store.threads[thread.thread_id].cross_doc = false
          }
          if (typeof store.threads[thread.thread_id].selectionTouched !== 'boolean') {
            store.threads[thread.thread_id].selectionTouched = false
          }
          if (!Array.isArray(store.threads[thread.thread_id].active_doc_ids)) {
            store.threads[thread.thread_id].active_doc_ids = []
          }
        }
      })
    } else {
      console.log('loadThreads: No threads in result or invalid format')
      threadList.value = []
    }
  } catch (error) {
    console.error('Error loading threads:', error)
    console.error('Error details:', error.response?.data || error.message)
    threadList.value = []
  }
}

async function handleCreateNewThread() {
  await store.createNewThread()
  // Refresh threads list after creating new one
  loadThreads()
}

async function switchToThread(threadId) {
  console.log('switchToThread: Switching to thread:', threadId)
  try {
    // Switch thread first (this preserves current thread state)
    store.switchThread(threadId)
    console.log('switchToThread: Store switched to thread')
    
    // Only load from API if thread is persisted
    const threadEntry = store.threads?.[threadId]
    if (threadEntry && threadEntry.persisted !== false) {
      console.log('switchToThread: Loading thread history from API')
    await loadThreadHistory(threadId)
    console.log('switchToThread: Thread history loaded successfully')
    } else {
      console.log('switchToThread: Thread not persisted yet, using local cache')
    }
  } catch (error) {
    console.error('switchToThread: Error switching thread:', error)
  }
}

async function loadThreadHistory(threadId) {
  if (!store.userId) return
  const threadEntry = store.threads?.[threadId]
  if (threadEntry && threadEntry.persisted === false) {
    console.log('loadThreadHistory: Thread not persisted yet, relying on local cache')
    if (store.currentThreadId === threadId) {
      store.messages = [...(threadEntry.messages || [])]
    }
    return
  }
  try {
    const result = await apiService.getThread(threadId, store.userId)
    if (result?.messages !== undefined) {
      // Update thread in store (always, regardless of active state)
      if (!store.threads[threadId]) {
        store.threads[threadId] = {
          messages: [],
          created_at: result.created_at || new Date().toISOString(),
          active_doc_ids: [],
          persisted: true
        }
      } else {
        store.threads[threadId].persisted = true
      }
      
      // Only overwrite local messages if backend has MORE messages than local
      // This prevents losing local messages when backend hasn't synced yet
      const localMessages = store.threads[threadId].messages || []
      const backendMessages = result.messages || []
      
      if (backendMessages.length > localMessages.length) {
        // Backend has more messages - use backend (it's more up-to-date)
        console.log(`loadThreadHistory: Backend has ${backendMessages.length} messages, local has ${localMessages.length}, using backend`)
        store.threads[threadId].messages = backendMessages
        if (store.currentThreadId === threadId) {
          store.messages = [...backendMessages]
        }
      } else {
        // Keep local messages if they're equal or more than backend
        console.log(`loadThreadHistory: Keeping ${localMessages.length} local messages, backend has ${backendMessages.length}`)
      if (store.currentThreadId === threadId) {
          store.messages = [...localMessages]
        }
      }
    }
  } catch (error) {
    console.error('Error loading thread history:', error)
    if (error.response?.status === 404) {
      console.warn('loadThreadHistory: Thread not found in backend (likely not persisted yet)')
    }
  }
}

function refreshDocuments() {
  store.loadDocuments()
}

function isDocActive(docId) {
  // Check if doc is in selected docs or active docs for current thread
  if (store.selectedDocIds.includes(docId)) return true
  if (store.activeDocIdsForThread.includes(docId)) return true
  return false
}

function toggleDocSelection(docId) {
  const current = [...store.selectedDocIds]
  const index = current.indexOf(docId)
  if (index > -1) {
    current.splice(index, 1)
  } else {
    current.push(docId)
  }
  store.setActiveDocIds(current)
}

function toggleDiagnostics(docId) {
  // Toggle expander
  expandedDiagnostics.value[docId] = !expandedDiagnostics.value[docId]
  
  // If expanding and data not loaded, fetch diagnostics
  if (expandedDiagnostics.value[docId] && !diagnosticsData.value[docId]) {
    loadDiagnostics(docId)
  }
}

async function loadDiagnostics(docId) {
  // Initialize diagnostics data
  if (!diagnosticsData.value[docId]) {
    diagnosticsData.value[docId] = { loading: false, data: null, error: null }
  }
  
  diagnosticsData.value[docId].loading = true
  diagnosticsData.value[docId].error = null
  
  try {
    const result = await apiService.getDiagnostics(docId)
    diagnosticsData.value[docId].data = result
    diagnosticsData.value[docId].loading = false
    console.log('Diagnostics loaded for', docId, ':', result)
  } catch (error) {
    const errorMessage = error.response?.data?.detail || error.message || 'Unknown error'
    diagnosticsData.value[docId].error = errorMessage
    diagnosticsData.value[docId].loading = false
    console.error('Error loading diagnostics:', errorMessage)
  }
}

async function handleDeleteDocument(docId) {
  if (!confirm(`Delete document "${docId.substring(0, 8)}..."?`)) {
    return
  }
  try {
    await store.deleteDocument(docId)
    alert('Document deleted successfully!')
    // Refresh documents after deletion
    store.loadDocuments()
  } catch (error) {
    alert(`Error deleting document: ${error.message}`)
  }
}

function isThreadProcessing(threadId) {
  // Check if the thread is currently processing
  return store.isThreadProcessing(threadId)
}

// Watch for thread changes to refresh thread list
watch(() => store.currentThreadId, () => {
  loadThreads()
})

// Watch for new threads being created to refresh list
watch(() => Object.keys(store.threads).length, () => {
  loadThreads()
})

// Watch for messages being added (indicates a query was made, thread should be in DB)
watch(() => store.messages.length, async (newLength, oldLength) => {
  // Only refresh if messages actually increased (new message added)
  if (newLength > oldLength && store.userId) {
    // Refresh threads immediately - DB write completes before response is sent
    console.log('Sidebar: Message added, refreshing threads')
    await loadThreads()
  }
})

onMounted(() => {
  if (store.userIdEntered) {
    store.loadDocuments()
    loadThreads()
  }
})
</script>
