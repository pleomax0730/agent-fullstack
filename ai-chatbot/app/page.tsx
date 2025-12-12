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
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from '@/components/ai-elements/prompt-input';
import { useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { CopyIcon, RefreshCcwIcon } from 'lucide-react';
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from '@/components/ai-elements/reasoning';
import { Loader } from '@/components/ai-elements/loader';
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from '@/components/ai-elements/tool';
import { isToolUIPart, getToolName } from 'ai';
import type { ToolUIPart } from 'ai';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export default function Home() {
  const [input, setInput] = useState('');

  const { messages, sendMessage, status, regenerate } = useChat({
    transport: new DefaultChatTransport({
      api: `${BACKEND_URL}/api/chat`,
    }),
    onError: (error) => {
      console.error('Chat error:', error);
    },
  });

  const handleSubmit = () => {
    if (!input.trim()) return;
    sendMessage({ text: input });
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
                  // Handle text parts
                  if (part.type === 'text') {
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
                  }

                  // Handle reasoning parts
                  if (part.type === 'reasoning') {
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
                  }

                  // Handle tool parts (type is tool-{toolName} in v5/v6)
                  if (isToolUIPart(part)) {
                    const toolPart = part as ToolUIPart;
                    const toolName = getToolName(toolPart);
                    return (
                      <Tool key={`${message.id}-${i}`}>
                        <ToolHeader
                          title={toolName}
                          type={toolPart.type}
                          state={toolPart.state}
                        />
                        <ToolContent>
                          <ToolInput input={toolPart.input} />
                          <ToolOutput
                            output={toolPart.output}
                            errorText={toolPart.errorText}
                          />
                        </ToolContent>
                      </Tool>
                    );
                  }

                  return null;
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
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
          </PromptInputBody>
          <PromptInputFooter>
            <PromptInputTools />
            <PromptInputSubmit disabled={!input.trim()} status={status} />
          </PromptInputFooter>
        </PromptInput>
      </div>
    </div>
  );
}
