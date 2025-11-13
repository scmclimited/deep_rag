<template>
  <div 
    class="sidebar bg-dark-surface border-r border-dark-border flex flex-col h-screen fixed left-0 top-0 overflow-y-auto overflow-x-hidden z-50 transition-all duration-300"
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
        <!-- Tabs for Active/Archived -->
        <div class="flex gap-1 mb-2 border-b border-dark-border">
          <button
            @click="activeTab = 'active'"
            class="flex-1 px-2 py-1 text-xs font-medium transition-colors"
            :class="activeTab === 'active' 
              ? 'text-blue-400 border-b-2 border-blue-400' 
              : 'text-gray-400 hover:text-gray-300'"
          >
            Active
          </button>
          <button
            @click="activeTab = 'archived'"
            class="flex-1 px-2 py-1 text-xs font-medium transition-colors"
            :class="activeTab === 'archived' 
              ? 'text-blue-400 border-b-2 border-blue-400' 
              : 'text-gray-400 hover:text-gray-300'"
          >
            Archived
          </button>
        </div>
        <div class="text-xs text-gray-400 mb-2">
          Current: {{ store.currentThreadId?.substring(0, 8) }}...
        </div>
        <div class="text-xs text-gray-400 mb-2">
          {{ currentThreadsList.length }} {{ activeTab === 'active' ? 'active' : 'archived' }} threads
        </div>
        <div v-if="currentThreadsList.length === 0" class="text-xs text-gray-500 p-2 italic">
          <span v-if="activeTab === 'active'">No active threads found. Create a new thread or make a query to start.</span>
          <span v-else>No archived threads found.</span>
        </div>
        <div v-else class="space-y-1 max-h-48 overflow-y-auto">
          <div
            v-for="thread in currentThreadsList"
            :key="thread.thread_id"
            class="p-2 bg-dark-bg rounded border border-dark-border hover:border-blue-500 transition-colors relative group"
            :class="{ 'border-blue-500 bg-blue-900/20': thread.thread_id === store.currentThreadId }"
          >
            <div 
              @click="switchToThread(thread.thread_id)"
              class="cursor-pointer"
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
            <!-- Archive/Unarchive button (visible on hover) -->
            <button
              @click.stop="activeTab === 'active' ? handleArchiveThread(thread.thread_id) : handleUnarchiveThread(thread.thread_id)"
              class="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-gray-300 hover:text-white"
              :title="activeTab === 'active' ? 'Archive thread' : 'Unarchive thread'"
            >
              {{ activeTab === 'active' ? '📦' : '📤' }}
            </button>
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
                <div class="bg-dark-bg rounded p-3 max-h-96 overflow-y-auto overflow-x-auto">
                  <pre class="text-xs text-gray-300 whitespace-pre-wrap break-words font-mono">{{ JSON.stringify(diagnosticsData[doc.doc_id].data, null, 2) }}</pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Settings Section -->
      <div class="px-4 py-4 border-t border-dark-border overflow-x-hidden max-w-full box-border">
        <h3 class="font-semibold mb-2 truncate">⚙️ Settings</h3>
        <label class="flex items-center gap-2 text-sm cursor-pointer break-words max-w-full">
          <input
            type="checkbox"
            v-model="store.crossDocSearch"
            class="rounded cursor-pointer flex-shrink-0"
          />
          <span class="break-words overflow-wrap-anywhere min-w-0 flex-1 pr-1">Cross-Document Search</span>
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
const archivedThreadList = ref([])
const activeTab = ref('active') // 'active' or 'archived'
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

  // Use API-provided fields if available (from /threads endpoint), otherwise extract from messages
  const latestQuery = threadData?.latest_query || latestUserMessage?.content || (isPersisted ? 'No messages yet' : 'New thread')
  const lastActivity = threadData?.last_activity || lastMessage?.timestamp || createdAt
  const queryCount = threadData?.query_count !== undefined ? threadData.query_count : messages.filter(msg => msg.role === 'user').length
  const docIds = Array.isArray(threadData?.doc_ids) ? threadData.doc_ids : (Array.isArray(threadData?.active_doc_ids) ? threadData.active_doc_ids : [])

  return {
    thread_id: threadId,
    user_id: store.userId || threadData?.user_id || null,
    created_at: createdAt,
    last_activity: lastActivity,
    latest_query: latestQuery,
    query_count: queryCount,
    doc_ids: docIds,
    isPersisted
  }
}

// Helper function to build threads list from API data
function buildThreadsListFromAPI(apiThreads) {
  const merged = new Map()

  // Seed with threads returned from the API (persisted threads)
  // These have latest_query, last_activity, query_count from the backend
  apiThreads.forEach(thread => {
    if (thread?.thread_id) {
      const summary = buildThreadSummary(thread.thread_id, thread, true)
      merged.set(thread.thread_id, summary)
      // Ensure thread exists in store for message loading
      if (!store.threads[thread.thread_id]) {
        store.threads[thread.thread_id] = {
          messages: [],
          created_at: thread.created_at,
          active_doc_ids: [],
          persisted: true,
          cross_doc: false,
          selectionTouched: false,
          is_processing: false
        }
      } else {
        // Update store thread metadata from API if available
        const storeThread = store.threads[thread.thread_id]
        if (thread.created_at && !storeThread.created_at) {
          storeThread.created_at = thread.created_at
        }
        if (storeThread.persisted === undefined) {
          storeThread.persisted = true
        }
        // If the user never touched selection for this thread, ensure no active docs persist
        if (!storeThread.selectionTouched) {
          storeThread.active_doc_ids = []
        }
      }
    }
  })

  // Merge in locally created threads and latest metadata from the store
  // These might have messages loaded but not the API metadata
  // IMPORTANT: Only merge threads that are NOT in the API response (for active threads only)
  // For archived threads, we only show what the API returns (no local merging)
  if (apiThreads === threadList.value) {
    // This is the active threads list - merge with local store
    // BUT: Only include threads that are NOT archived
    // We check the API response to see if a thread is archived (if it's not in the API response for active threads, it might be archived)
    const apiThreadIds = new Set(apiThreads.map(t => t.thread_id))
    const archivedThreadIds = new Set(archivedThreadList.value.map(t => t.thread_id))
    
    Object.entries(store.threads || {}).forEach(([threadId, threadData]) => {
      // CRITICAL: Skip threads that are in the archived list - they should NOT appear in active list
      if (archivedThreadIds.has(threadId)) {
        return // Skip archived threads
      }
      
      // CRITICAL: Only merge threads that are in the API response OR are new local threads
      // If a thread is archived, it won't be in the API response for active threads
      // So we should NOT add it back from the store
      if (merged.has(threadId)) {
        // Thread is in API response - update metadata
        const existing = merged.get(threadId)
        const summary = buildThreadSummary(threadId, threadData)
        // Update with message-based data if it's more recent or if API data is missing
        if (summary.latest_query && summary.latest_query !== 'No messages yet' && 
            (!existing.latest_query || existing.latest_query === 'No messages yet')) {
          existing.latest_query = summary.latest_query
        }
        if (summary.last_activity && (!existing.last_activity || 
            new Date(summary.last_activity) > new Date(existing.last_activity))) {
          existing.last_activity = summary.last_activity
        }
        existing.query_count = Math.max(existing.query_count || 0, summary.query_count)
        if (summary.doc_ids.length > 0) {
          existing.doc_ids = summary.doc_ids
        }
        if (!existing.created_at && summary.created_at) {
          existing.created_at = summary.created_at
        }
      } else if (!apiThreadIds.has(threadId)) {
        // New local thread not in API response (only for active threads)
        // This is a new thread that hasn't been persisted yet
        // Only add if it's not archived (we can't check archive status for new threads, so assume they're active)
        const summary = buildThreadSummary(threadId, threadData)
        merged.set(threadId, summary)
      }
      // If threadId is NOT in merged AND NOT in apiThreadIds, it means it's archived
      // So we DON'T add it back - this prevents archived threads from reappearing
    })
  }
  // For archived threads (apiThreads === archivedThreadList.value), don't merge with local store
  // Only show what the API returns

  let finalList = Array.from(merged.values())
  
  // CRITICAL: If building active threads list, filter out any threads that are in the archived list
  if (apiThreads === threadList.value) {
    const archivedThreadIds = new Set(archivedThreadList.value.map(t => t.thread_id))
    finalList = finalList.filter(thread => !archivedThreadIds.has(thread.thread_id))
  }
  
  return finalList.sort((a, b) => {
    const dateA = new Date(a.last_activity || a.created_at || 0)
    const dateB = new Date(b.last_activity || b.created_at || 0)
    return dateB.getTime() - dateA.getTime()
  })
}

const threadsList = computed(() => {
  // Build active threads list and ensure archived threads are excluded
  const activeList = buildThreadsListFromAPI(threadList.value)
  // Double-check: filter out any threads that are in archived list
  const archivedThreadIds = new Set(archivedThreadList.value.map(t => t.thread_id))
  return activeList.filter(thread => !archivedThreadIds.has(thread.thread_id))
})

const archivedThreadsList = computed(() => {
  return buildThreadsListFromAPI(archivedThreadList.value)
})

// Current threads list based on active tab
const currentThreadsList = computed(() => {
  if (activeTab.value === 'active') {
    // For active tab, ensure no archived threads are included
    const activeList = threadsList.value
    const archivedThreadIds = new Set(archivedThreadList.value.map(t => t.thread_id))
    return activeList.filter(thread => !archivedThreadIds.has(thread.thread_id))
  } else {
    // For archived tab, return archived threads
    return archivedThreadsList.value
  }
})

async function handleUserIdSubmit() {
  if (userIdInput.value.trim()) {
    await store.setUserId(userIdInput.value.trim())
    // CRITICAL: Also load threads in Sidebar to update threadList.value for returning users
    // This ensures the thread list UI updates immediately when a returning user logs in
    await loadThreads()
  }
}

async function loadThreads() {
  if (!store.userId) {
    console.log('loadThreads: No userId, skipping')
    return
  }
  try {
    console.log('loadThreads: Loading active threads for userId:', store.userId)
    const result = await apiService.getThreads(store.userId, 100, false)
    console.log('loadThreads: API result:', result)
    console.log('loadThreads: Threads count:', result?.count, 'Threads array length:', result?.threads?.length)
    
    // Always update threadList, even if empty
    if (result && Array.isArray(result.threads)) {
      // CRITICAL: Filter out any threads that are in the archived list
      // This ensures archived threads don't appear in active list even if API returns them
      const archivedThreadIds = new Set(archivedThreadList.value.map(t => t.thread_id))
      const filteredThreads = result.threads.filter(thread => !archivedThreadIds.has(thread.thread_id))
      threadList.value = filteredThreads
      console.log('loadThreads: Loaded', result.threads.length, 'threads from API, filtered to', filteredThreads.length, 'active threads (excluded', result.threads.length - filteredThreads.length, 'archived)')
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
          if (!store.threads[thread.thread_id].selectionTouched) {
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

async function loadArchivedThreads() {
  if (!store.userId) {
    console.log('loadArchivedThreads: No userId, skipping')
    return
  }
  try {
    console.log('loadArchivedThreads: Loading archived threads for userId:', store.userId)
    const result = await apiService.getThreads(store.userId, 100, true)
    console.log('loadArchivedThreads: API result:', result)
    
    if (result && Array.isArray(result.threads)) {
      archivedThreadList.value = result.threads
      console.log('loadArchivedThreads: Loaded', result.threads.length, 'archived threads')
    } else {
      console.log('loadArchivedThreads: No threads in result or invalid format')
      archivedThreadList.value = []
    }
  } catch (error) {
    console.error('Error loading archived threads:', error)
    console.error('Error details:', error.response?.data || error.message)
    archivedThreadList.value = []
  }
}

async function handleCreateNewThread() {
  await store.createNewThread()
  // Refresh threads list after creating new one
  loadThreads()
  // Auto-refresh documents when creating a new thread
  await refreshDocuments()
}

async function switchToThread(threadId) {
  console.log('switchToThread: Switching to thread:', threadId)
  try {
    // Switch thread first (this preserves current thread state)
    store.switchThread(threadId)
    console.log('switchToThread: Store switched to thread')
    
    // Auto-refresh documents when switching threads
    await refreshDocuments()
    
    // Always load from API if thread is persisted (even if already loaded, refresh to ensure sync)
    const threadEntry = store.threads?.[threadId]
    if (threadEntry && threadEntry.persisted !== false) {
      console.log('switchToThread: Loading thread history from API (refreshing for returning user)')
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
  // Use store's refreshThreadHistory method for consistency
  await store.refreshThreadHistory(threadId)
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

async function handleArchiveThread(threadId) {
  if (!store.userId) {
    console.error('Cannot archive thread: no userId')
    return
  }
  if (!confirm(`Archive thread "${threadId.substring(0, 8)}..."?`)) {
    return
  }
  try {
    const response = await apiService.archiveThread(threadId, store.userId, true)
    console.log('Thread archived successfully:', threadId, response)
    
    // Immediately remove from active thread list (optimistic update)
    const beforeFilter = threadList.value.length
    threadList.value = threadList.value.filter(t => t.thread_id !== threadId)
    console.log(`Optimistic update: Removed thread from active list (${beforeFilter} -> ${threadList.value.length})`)
    
    // Refresh both active and archived threads lists from API
    // Load archived first to ensure it's available for the computed property check
    await loadArchivedThreads()
    await loadThreads()
    
    // Verify the thread is not in active list
    const stillInActive = threadList.value.some(t => t.thread_id === threadId)
    const inArchived = archivedThreadList.value.some(t => t.thread_id === threadId)
    console.log(`After refresh: thread in active=${stillInActive}, in archived=${inArchived}`)
    
    if (stillInActive) {
      console.warn('WARNING: Thread still appears in active list after archiving!')
      // Force remove again
      threadList.value = threadList.value.filter(t => t.thread_id !== threadId)
    }
    
    console.log('Thread lists refreshed after archiving')
  } catch (error) {
    console.error('Error archiving thread:', error)
    alert(`Failed to archive thread: ${error.response?.data?.detail || error.message}`)
    // Revert optimistic update on error
    await loadThreads()
  }
}

async function handleUnarchiveThread(threadId) {
  if (!store.userId) {
    console.error('Cannot unarchive thread: no userId')
    return
  }
  if (!confirm(`Unarchive thread "${threadId.substring(0, 8)}..."?`)) {
    return
  }
  try {
    const response = await apiService.archiveThread(threadId, store.userId, false)
    console.log('Thread unarchived successfully:', threadId, response)
    
    // Immediately remove from archived thread list (optimistic update)
    archivedThreadList.value = archivedThreadList.value.filter(t => t.thread_id !== threadId)
    
    // Refresh both active and archived threads lists from API
    await Promise.all([
      loadThreads(),
      loadArchivedThreads()
    ])
    
    // Switch to active tab to show the unarchived thread
    activeTab.value = 'active'
    
    console.log('Thread lists refreshed after unarchiving')
  } catch (error) {
    console.error('Error unarchiving thread:', error)
    alert(`Failed to unarchive thread: ${error.response?.data?.detail || error.message}`)
    // Revert optimistic update on error
    await loadArchivedThreads()
  }
}

function isThreadProcessing(threadId) {
  // Check if the thread is currently processing
  return store.isThreadProcessing(threadId)
}

// Watch for active tab changes to load archived threads when needed
watch(() => activeTab.value, async (newTab) => {
  if (newTab === 'archived' && archivedThreadList.value.length === 0) {
    console.log('Sidebar: Switched to archived tab, loading archived threads')
    await loadArchivedThreads()
  }
})

// Watch for changes in archived thread list and ensure they're removed from active list
watch(() => archivedThreadList.value, () => {
  // When archived list changes, filter active list to remove archived threads
  // This ensures archived threads are immediately removed from active tab
  const archivedThreadIds = new Set(archivedThreadList.value.map(t => t.thread_id))
  const beforeFilter = threadList.value.length
  threadList.value = threadList.value.filter(thread => !archivedThreadIds.has(thread.thread_id))
  const afterFilter = threadList.value.length
  if (beforeFilter !== afterFilter) {
    console.log(`Sidebar: Filtered active list - removed ${beforeFilter - afterFilter} archived thread(s)`)
  }
}, { deep: true, immediate: false })

// Watch for changes in threadList and ensure archived threads are filtered
watch(() => threadList.value, () => {
  // When active thread list changes, ensure archived threads are filtered out
  const archivedThreadIds = new Set(archivedThreadList.value.map(t => t.thread_id))
  const hasArchived = threadList.value.some(thread => archivedThreadIds.has(thread.thread_id))
  if (hasArchived) {
    console.log('Sidebar: Active thread list contains archived threads, filtering them out')
    threadList.value = threadList.value.filter(thread => !archivedThreadIds.has(thread.thread_id))
  }
}, { deep: true, immediate: false })

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

// Watch for userId changes to auto-load threads for returning users
watch(() => store.userId, async (newUserId, oldUserId) => {
  if (newUserId && newUserId !== oldUserId) {
    console.log('Sidebar: userId changed, loading threads for:', newUserId)
    await loadThreads()
    await refreshDocuments()
  }
}, { immediate: false })

onMounted(() => {
  if (store.userIdEntered && store.userId) {
    console.log('Sidebar: onMounted with existing userId, loading threads and documents')
    store.loadDocuments()
    loadThreads()
  }
})
</script>
