<template>
  <div class="flex-1 flex flex-col h-full">
    <ChatMessages :threadId="threadId" />
    <ChatInput :threadId="threadId" />
  </div>
</template>

<script setup>
import { defineProps } from 'vue'
import ChatMessages from './ChatMessages.vue'
import ChatInput from './ChatInput.vue'

const props = defineProps({
  threadId: {
    type: String,
    required: true
  }
})
</script>

