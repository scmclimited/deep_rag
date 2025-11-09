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
    
    <!-- Main Chat Area - Dynamic width based on available space (80% of remaining width) -->
    <div 
      class="flex-1 flex flex-col max-w-4xl mx-auto w-full px-4 md:px-6 lg:px-8 main-content"
      :class="{ 'pointer-events-none': showMobileToggle && sidebarExpanded }"
    >
      <ChatHeader />
      <ChatMessages />
      <ChatInput />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatHeader from './components/ChatHeader.vue'
import ChatMessages from './components/ChatMessages.vue'
import ChatInput from './components/ChatInput.vue'

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
