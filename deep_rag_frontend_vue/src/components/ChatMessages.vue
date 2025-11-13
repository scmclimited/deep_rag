<template>
  <div ref="messagesContainer" class="flex-1 overflow-y-auto p-4">
    <div class="w-full max-w-4xl mx-auto space-y-6">
      <div v-if="threadMessages.length === 0" class="text-center text-gray-400 mt-16">
        <p class="text-lg mb-2">👋 Welcome to Deep RAG Chat</p>
        <p class="text-sm">Start a conversation by uploading or selecting a document...</p>
      </div>
      <div
        v-for="(message, index) in threadMessages"
        :key="index"
        class="flex gap-4"
        :class="message.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <div
          class="max-w-2xl rounded-2xl px-5 py-4 shadow-lg"
        :class="message.role === 'user' 
          ? 'bg-blue-600 text-white' 
          : 'bg-dark-surface text-white border border-dark-border'"
      >
        <div v-if="message.attachments && message.attachments.length > 0" class="text-xs text-gray-300 mb-2 space-y-1">
          <div
            v-for="(attachment, idx) in message.attachments"
            :key="`${attachment}-${idx}`"
            class="flex items-center gap-1"
          >
            <span>📎</span>
            <span>{{ attachment }}</span>
          </div>
        </div>
        <div class="whitespace-pre-wrap leading-relaxed">{{ cleanAnswerText(message.content) }}</div>
        
        <!-- Document IDs and Pages -->
        <div v-if="message.doc_ids && message.doc_ids.length > 0" class="mt-3 pt-3 border-t border-gray-600 space-y-2">
          <div class="text-xs font-semibold text-gray-300 mb-1">📚 Sources:</div>
          <div v-for="(docId, idx) in message.doc_ids" :key="docId" class="text-xs space-y-1">
            <div class="flex items-center gap-2 flex-wrap">
              <span class="font-mono text-blue-300">{{ docId }}</span>
              <span v-if="getDocTitle(message, idx)" class="text-gray-400">({{ getDocTitle(message, idx) }})</span>
            </div>
            <div v-if="message.pages && message.pages.length > 0" class="text-gray-400 ml-2">
              📑 Pages: {{ message.pages.join(', ') }}
            </div>
          </div>
        </div>
        <div v-else-if="message.doc_id" class="mt-3 pt-3 border-t border-gray-600">
          <div class="text-xs font-semibold text-gray-300 mb-1">📚 Source:</div>
          <div class="flex items-center gap-2 flex-wrap">
            <span class="font-mono text-blue-300 text-xs">{{ message.doc_id }}</span>
            <span v-if="message.doc_title" class="text-gray-400 text-xs">({{ message.doc_title }})</span>
          </div>
          <div v-if="message.pages && message.pages.length > 0" class="text-gray-400 text-xs mt-1">
            📑 Pages: {{ message.pages.join(', ') }}
          </div>
        </div>
        
        <!-- Confidence Score - Hide when 0% or answer is "I don't know" -->
        <div v-if="shouldShowConfidence(message)" class="mt-3 pt-3 border-t border-gray-600">
          <div class="flex items-center gap-2 text-xs">
            <span>{{ getConfidenceIcon(message.confidence) }}</span>
            <span class="text-gray-300">Confidence: {{ message.confidence.toFixed(1) }}%</span>
          </div>
        </div>
      </div>
    </div>
    </div>
  </div>
</template>

<script setup>
import { computed, defineProps, ref, watch, onMounted, nextTick } from 'vue'
import { useAppStore } from '../stores/app'

const props = defineProps({
  threadId: {
    type: String,
    required: true
  }
})

const store = useAppStore()
const messagesContainer = ref(null)

// Get messages for this specific thread, sorted by timestamp (earliest to latest)
const threadMessages = computed(() => {
  const thread = store.threads[props.threadId]
  const messages = thread?.messages || []
  // Sort by timestamp to ensure earliest to latest order
  return [...messages].sort((a, b) => {
    const timeA = new Date(a.timestamp || a.created_at || 0).getTime()
    const timeB = new Date(b.timestamp || b.created_at || 0).getTime()
    return timeA - timeB
  })
})

// Auto-scroll to bottom function
function scrollToBottom(forceInstant = false) {
  // Ensure DOM updates are applied before scrolling
  nextTick(() => {
    requestAnimationFrame(() => {
      if (!messagesContainer.value) {
        return
      }

      const { scrollHeight, clientHeight } = messagesContainer.value

      // Always scroll when new messages arrive; even if scrollHeight equals clientHeight,
      // we still want to ensure we're positioned at the bottom.
      messagesContainer.value.scrollTo({
        top: scrollHeight,
        behavior: forceInstant ? 'auto' : 'smooth'
      })
    })
  })
}

// Track previous thread ID to detect thread switches
const previousThreadId = ref(null)

// Watch for thread changes and scroll to bottom
watch(() => props.threadId, async (newThreadId, oldThreadId) => {
  previousThreadId.value = oldThreadId
  // When switching threads, wait for messages to load then scroll
  // Use multiple attempts to catch async loading
  const attemptScroll = (attempt = 0) => {
    if (attempt < 10) {
      setTimeout(() => {
        const thread = store.threads[newThreadId]
        const hasMessages = thread?.messages?.length > 0
        const messagesLength = threadMessages.value.length
        
        if (hasMessages && messagesLength > 0) {
          // Messages loaded - wait for DOM to render, then scroll instantly
          nextTick(() => {
            setTimeout(() => {
              scrollToBottom(true)
            }, 50)
          })
        } else if (attempt < 9) {
          // Messages not loaded yet - try again with longer delay
          attemptScroll(attempt + 1)
        }
      }, attempt === 0 ? 150 : attempt < 5 ? 200 : 300)
    }
  }
  attemptScroll()
}, { immediate: true })

// Watch for messages being populated (especially when loading old threads)
watch(() => {
  const thread = store.threads[props.threadId]
  return thread?.messages?.length || 0
}, (newLength, oldLength) => {
  // If messages were just loaded (length changed from 0 or increased significantly)
  // This handles the case when switching to an old thread and messages load asynchronously
  if (newLength > 0 && (oldLength === 0 || newLength > oldLength)) {
    // Wait for DOM to render, then scroll instantly when messages are first loaded
    nextTick(() => {
      scrollToBottom(true)
    })
  }
}, { immediate: true })

// Watch for new messages and scroll to bottom
// Use a more robust watcher that tracks length changes
watch(() => threadMessages.value.length, (newLength, oldLength) => {
  if (newLength > oldLength) {
    // New message added - wait for DOM to render, then scroll to bottom
    nextTick(() => {
      scrollToBottom(true)
    })
  } else if (newLength > 0 && oldLength === 0) {
    // Messages just loaded (e.g., when switching to an old thread)
    // Wait longer for DOM to fully render
    nextTick(() => {
      scrollToBottom(true) // Use instant scroll for initial load
    })
  }
}, { immediate: true })

// Also watch the messages array itself for deep changes (content updates)
watch(threadMessages, (newMessages, oldMessages) => {
  // Only scroll if messages actually changed (not just reference)
  if (newMessages.length !== (oldMessages?.length || 0)) {
    scrollToBottom(true)
  }
}, { deep: true })

// Listen for global scroll signals (emitted when new messages are added)
watch(() => store.scrollSignal.value.token, (token) => {
  if (!token) return
  const signal = store.scrollSignal.value
  if (signal.threadId === props.threadId) {
    scrollToBottom(signal.forceInstant)
  }
})

// Scroll to bottom on mount
onMounted(() => {
  // Wait for messages to load, then scroll instantly
  const attemptScroll = (attempt = 0) => {
    if (attempt < 5) {
      setTimeout(() => {
        const thread = store.threads[props.threadId]
        const hasMessages = thread?.messages?.length > 0
        if (hasMessages) {
          // Messages loaded - scroll instantly
          scrollToBottom(true)
        } else if (attempt < 4) {
          // Messages not loaded yet - try again
          attemptScroll(attempt + 1)
        }
      }, attempt === 0 ? 200 : 300)
    }
  }
  attemptScroll()
})

function getConfidenceIcon(confidence) {
  if (confidence >= 80) return '🟢'
  if (confidence >= 60) return '🟡'
  return '🔴'
}

// Check if confidence should be displayed
// Hide when confidence is 0% or answer is "I don't know"
function shouldShowConfidence(message) {
  if (message.confidence === undefined || message.confidence === null) return false
  // Hide if confidence is 0% (or very close to 0)
  if (message.confidence <= 0.01) return false
  // Hide if answer is "I don't know" (case-insensitive)
  const answerText = message.content?.toLowerCase().trim() || ''
  if (answerText === "i don't know." || answerText === "i don't know") return false
  return true
}

// Remove per-chunk confidence from citations in answer text
// Citations like "[1] doc:... p2-2 (confidence: 20.4%)" should be "[1] doc:... p2-2"
// BUT preserve confidence scores in "Documents used for analysis" section
function cleanAnswerText(text) {
  if (!text) return text
  
  // Split text into sections
  const docsAnalysisIndex = text.indexOf('Documents used for analysis')
  
  if (docsAnalysisIndex === -1) {
    // No "Documents used for analysis" section - remove all confidence scores
    return text.replace(/\(confidence:\s*\d+\.?\d*%\)/gi, '')
  }
  
  // Split into parts: before "Documents used for analysis" and after
  const beforeSection = text.substring(0, docsAnalysisIndex)
  const sectionAndAfter = text.substring(docsAnalysisIndex)
  
  // Remove confidence scores only from the part before "Documents used for analysis"
  const cleanedBefore = beforeSection.replace(/\(confidence:\s*\d+\.?\d*%\)/gi, '')
  
  // Keep the "Documents used for analysis" section intact (with confidence scores)
  return cleanedBefore + sectionAndAfter
}

function getDocTitle(message, index) {
  if (!message) return null
  if (Array.isArray(message.doc_titles) && message.doc_titles[index]) {
    return message.doc_titles[index]
  }
  if (message.doc_title && index === 0) {
    return message.doc_title
  }
  const docId = Array.isArray(message.doc_ids) ? message.doc_ids[index] : null
  if (!docId) return null
  const docMeta = store.documents?.find?.(doc => doc.doc_id === docId)
  if (docMeta?.title) {
    return docMeta.title
  }
  return null
}
</script>
