/**
 * Example: Using AI Elements with a custom Python FastAPI backend
 * 
 * This shows how to modify the useChat hook to point to your own backend
 * instead of the default /api/chat route.
 */

'use client';

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import {
  Message,
  MessageContent,
  MessageResponse,
  MessageActions,
  MessageAction,
} from '@/components/ai-elements/message';
import {
  PromptInput,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  type PromptInputMessage,
} from '@/components/ai-elements/prompt-input';
import { Fragment, useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { CopyIcon, RefreshCcwIcon, WrenchIcon } from 'lucide-react';
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from '@/components/ai-elements/reasoning';
import { Loader } from '@/components/ai-elements/loader';
import {
  Tool,
  ToolContent,
  ToolResult,
  ToolTrigger,
} from '@/components/ai-elements/tool';

// Point to your FastAPI backend
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

const ChatBotWithPythonBackend = () => {
  const [input, setInput] = useState('');
  
  // Configure useChat to use your Python backend
  const { messages, sendMessage, status, regenerate } = useChat({
    // Point to your FastAPI endpoint
    api: `${BACKEND_URL}/api/chat`,
    
    // Optional: Custom headers if needed
    headers: {
      'Content-Type': 'application/json',
    },
    
    // Optional: Handle errors
    onError: (error) => {
      console.error('Chat error:', error);
    },
  });

  const handleSubmit = (message: PromptInputMessage) => {
    const hasText = Boolean(message.text);
    if (!hasText) return;

    sendMessage({ text: message.text || '' });
    setInput('');
  };

  return (
    <div className="max-w-4xl mx-auto p-6 relative size-full h-screen">
      <div className="flex flex-col h-full">
        <Conversation className="h-full">
          <ConversationContent>
            {messages.map((message) => (
              <div key={message.id}>
                {message.parts.map((part, i) => {
                  switch (part.type) {
                    case 'text':
                      return (
                        <Message key={`${message.id}-${i}`} from={message.role}>
                          <MessageContent>
                            <MessageResponse>
                              {part.text}
                            </MessageResponse>
                          </MessageContent>
                          {message.role === 'assistant' && (
                            <MessageActions>
                              <MessageAction
                                onClick={() => regenerate()}
                                label="Retry"
                              >
                                <RefreshCcwIcon className="size-3" />
                              </MessageAction>
                              <MessageAction
                                onClick={() =>
                                  navigator.clipboard.writeText(part.text)
                                }
                                label="Copy"
                              >
                                <CopyIcon className="size-3" />
                              </MessageAction>
                            </MessageActions>
                          )}
                        </Message>
                      );
                    
                    case 'reasoning':
                      // Reasoning/thinking from models like Gemini with thinking enabled
                      return (
                        <Reasoning
                          key={`${message.id}-${i}`}
                          className="w-full"
                          isStreaming={
                            status === 'streaming' && 
                            i === message.parts.length - 1 && 
                            message.id === messages.at(-1)?.id
                          }
                        >
                          <ReasoningTrigger />
                          <ReasoningContent>{part.text}</ReasoningContent>
                        </Reasoning>
                      );
                    
                    case 'tool-invocation':
                      // Tool calls from your Python backend
                      return (
                        <Tool key={`${message.id}-${i}`}>
                          <ToolTrigger>
                            <WrenchIcon className="size-4" />
                            {part.toolInvocation.toolName}
                          </ToolTrigger>
                          <ToolContent>
                            <div className="text-xs text-muted-foreground">
                              Args: {JSON.stringify(part.toolInvocation.args)}
                            </div>
                            {part.toolInvocation.result && (
                              <ToolResult>
                                {JSON.stringify(part.toolInvocation.result)}
                              </ToolResult>
                            )}
                          </ToolContent>
                        </Tool>
                      );
                    
                    default:
                      return null;
                  }
                })}
              </div>
            ))}
            {status === 'submitted' && <Loader />}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <PromptInput onSubmit={handleSubmit} className="mt-4">
          <PromptInputBody>
            <PromptInputTextarea
              onChange={(e) => setInput(e.target.value)}
              value={input}
              placeholder="Send a message to your Python backend..."
            />
          </PromptInputBody>
          <PromptInputFooter>
            <PromptInputTools>
              {/* Add your custom buttons here */}
            </PromptInputTools>
            <PromptInputSubmit disabled={!input && !status} status={status} />
          </PromptInputFooter>
        </PromptInput>
      </div>
    </div>
  );
};

export default ChatBotWithPythonBackend;

