<template>
  <div class="min-h-screen bg-dark-bg text-white flex flex-col">
    <!-- Collapsed Sidebar Indicator Button (when sidebar is collapsed) -->
    <button
      v-if="showMobileToggle && !sidebarExpanded"
      @click="toggleSidebar"
      class="fixed top-4 left-4 z-[60] p-3 bg-dark-surface border border-dark-border rounded-lg hover:bg-dark-bg transition-colors shadow-lg"
      title="Expand sidebar"
    >
      <span class="text-xl">☰</span>
    </button>
    
    <!-- Sidebar -->
    <Sidebar :expanded="sidebarExpanded" @close="closeSidebar" />
    
    <!-- Overlay for mobile sidebar (behind sidebar, above main content) -->
    <div
      v-if="showMobileToggle && sidebarExpanded"
      @click="closeSidebar"
      class="fixed inset-0 bg-black/50 z-30 md:hidden"
    ></div>
    
    <!-- Main Chat Area - Centered with dynamic width based on window size -->
    <div 
      class="flex-1 flex flex-col w-full px-4 md:px-6 lg:px-8 main-content"
      :class="{ 'pointer-events-none': showMobileToggle && sidebarExpanded }"
    >
      <ChatHeader />
      <!-- Per-thread view with KeepAlive to cache component instances -->
      <KeepAlive :max="10">
        <ThreadView 
          v-if="store.currentThreadId" 
          :key="store.currentThreadId"
          :threadId="store.currentThreadId"
        />
      </KeepAlive>
      <!-- Welcome screen when no thread is active -->
      <div v-if="!store.currentThreadId" class="flex-1 flex items-center justify-center">
        <div class="text-center">
          <h2 class="text-2xl font-bold mb-4">👋 Welcome to Deep RAG Chat</h2>
          <p class="text-gray-400">Start a conversation by asking a question or uploading a document</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, KeepAlive } from 'vue'
import { useAppStore } from './stores/app'
import Sidebar from './components/Sidebar.vue'
import ChatHeader from './components/ChatHeader.vue'
import ThreadView from './components/ThreadView.vue'

const store = useAppStore()

const sidebarExpanded = ref(false)
const showMobileToggle = ref(false)

function checkScreenSize() {
  showMobileToggle.value = window.innerWidth < 768
  if (!showMobileToggle.value) {
    sidebarExpanded.value = true // Always show on larger screens
  }
}

function toggleSidebar() {
  sidebarExpanded.value = !sidebarExpanded.value
}

function closeSidebar() {
  if (showMobileToggle.value) {
    sidebarExpanded.value = false
  }
}

onMounted(() => {
  checkScreenSize()
  window.addEventListener('resize', checkScreenSize)
})

onUnmounted(() => {
  window.removeEventListener('resize', checkScreenSize)
})
</script>
