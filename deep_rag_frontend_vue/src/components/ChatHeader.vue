<template>
  <div class="p-4 border-b border-dark-border bg-dark-bg/50 backdrop-blur-sm sticky top-0 z-10">
    <h1 class="text-2xl font-bold mb-1">💬 Deep RAG Chat</h1>
    <div class="text-xs text-gray-400">
      Thread: {{ store.currentThreadId?.substring(0, 8) }}... | 
      API: {{ apiBaseUrl }}
    </div>
  </div>
</template>

<script setup>
import { useAppStore } from '../stores/app'

const store = useAppStore()
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
</script>
