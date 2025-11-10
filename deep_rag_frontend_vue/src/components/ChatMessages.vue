<template>
  <div class="flex-1 overflow-y-auto p-4 space-y-6">
    <div v-if="threadMessages.length === 0" class="text-center text-gray-400 mt-16">
      <p class="text-lg mb-2">👋 Welcome to Deep RAG Chat</p>
      <p class="text-sm">Start a conversation by asking a question or uploading a document</p>
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
</template>

<script setup>
import { computed, defineProps } from 'vue'
import { useAppStore } from '../stores/app'

const props = defineProps({
  threadId: {
    type: String,
    required: true
  }
})

const store = useAppStore()

// Get messages for this specific thread
const threadMessages = computed(() => {
  const thread = store.threads[props.threadId]
  return thread?.messages || []
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
function cleanAnswerText(text) {
  if (!text) return text
  // Remove (confidence: X.X%) from citations
  return text.replace(/\(confidence:\s*\d+\.?\d*%\)/gi, '')
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
