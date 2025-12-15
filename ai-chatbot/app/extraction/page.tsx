'use client';

import { useState, useEffect } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import {
    Conversation,
    ConversationContent,
    ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import {
    Message,
    MessageContent,
    MessageResponse,
    MessageAttachments,
    MessageAttachment,
} from '@/components/ai-elements/message';
import {
    PromptInput,
    PromptInputBody,
    PromptInputFooter,
    PromptInputSubmit,
    PromptInputTextarea,
    PromptInputTools,
    PromptInputAttachments,
    PromptInputAttachment,
    PromptInputActionMenu,
    PromptInputActionMenuTrigger,
    PromptInputActionMenuContent,
    PromptInputActionAddAttachments,
} from '@/components/ai-elements/prompt-input';
import {
    Reasoning,
    ReasoningContent,
    ReasoningTrigger,
} from '@/components/ai-elements/reasoning';
import { Loader } from '@/components/ai-elements/loader';
import { Shimmer } from '@/components/ai-elements/shimmer';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '@/components/ui/tooltip';
import type { FileUIPart } from 'ai';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

type FieldType = 'string' | 'int' | 'float';

interface ExtractionField {
    label: string;
    field_name: string;
    value: string | number | null;
    type: FieldType;
    description: string;
}

interface ExtractionSchema {
    fields: ExtractionField[];
    reasoning: string;
}

// -----------------------------------------------------------------------------
// Schema Display
// -----------------------------------------------------------------------------

function FieldTypeTag({ type }: { type: FieldType }) {
    const colors: Record<FieldType, string> = {
        string: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
        int: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
        float: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    };
    return (
        <span className={`px-2 py-0.5 rounded text-xs font-mono ${colors[type]}`}>
            {type}
        </span>
    );
}

function SchemaTable({ schema }: { schema: ExtractionSchema }) {
    return (
        <div className="space-y-4">
            <div className="rounded-lg border overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                        <tr>
                            <th className="px-4 py-3 text-left font-medium whitespace-nowrap">欄位名稱</th>
                            <th className="px-4 py-3 text-left font-medium">取值</th>
                            <th className="px-4 py-3 text-left font-medium">類型</th>
                            <th className="px-4 py-3 text-left font-medium">說明</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {schema.fields.map((field, i) => (
                            <tr key={i} className="hover:bg-muted/30">
                                <td className="px-4 py-3 font-medium whitespace-nowrap">
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger asChild>
                                                <span className="cursor-help border-b border-dotted border-muted-foreground">
                                                    {field.label}
                                                </span>
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p className="font-mono text-xs">{field.field_name}</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </td>
                                <td className="px-4 py-3 text-foreground/90">{field.value ?? '-'}</td>
                                <td className="px-4 py-3"><FieldTypeTag type={field.type} /></td>
                                <td className="px-4 py-3 text-muted-foreground text-xs">{field.description}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <p className="text-sm text-muted-foreground italic">{schema.reasoning}</p>
        </div>
    );
}

// Skeleton shimmer for loading state - using ai-elements Shimmer and Loader
function SchemaTableSkeleton() {
    return (
        <div className="space-y-4">
            <div className="rounded-lg border overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                        <tr>
                            <th className="px-4 py-3 text-left font-medium">欄位名稱</th>
                            <th className="px-4 py-3 text-left font-medium">提取的值</th>
                            <th className="px-4 py-3 text-left font-medium">類型</th>
                            <th className="px-4 py-3 text-left font-medium">說明</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {[1, 2, 3, 4].map((i) => (
                            <tr key={i} className="animate-pulse">
                                <td className="px-4 py-3">
                                    <div className="h-4 w-20 bg-muted rounded" />
                                </td>
                                <td className="px-4 py-3">
                                    <div className="h-4 w-32 bg-muted rounded" />
                                </td>
                                <td className="px-4 py-3">
                                    <div className="h-5 w-14 bg-muted rounded" />
                                </td>
                                <td className="px-4 py-3">
                                    <div className="h-3 w-40 bg-muted rounded" />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader size={14} />
                <Shimmer duration={1.5}>正在分析文件...</Shimmer>
            </div>
        </div>
    );
}

function tryParseSchema(text: string): ExtractionSchema | null {
    try {
        const parsed = JSON.parse(text);
        if (parsed.fields && Array.isArray(parsed.fields)) {
            return parsed as ExtractionSchema;
        }
    } catch {
        // Not valid JSON
    }
    return null;
}


// -----------------------------------------------------------------------------
// Main Page
// -----------------------------------------------------------------------------

export default function ExtractionPage() {
    const [input, setInput] = useState('');
    const [currentSchema, setCurrentSchema] = useState<ExtractionSchema | null>(null);

    const { messages, sendMessage, status } = useChat({
        transport: new DefaultChatTransport({
            api: `${BACKEND_URL}/api/chat`,
        }),
        onError: (error) => {
            console.error('Chat error:', error);
        },
    });

    // Extract schema from latest assistant message
    useEffect(() => {
        const lastAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant');
        if (lastAssistantMsg) {
            for (const part of lastAssistantMsg.parts) {
                if (part.type === 'text') {
                    const schema = tryParseSchema(part.text);
                    if (schema) {
                        setCurrentSchema(schema);
                        break;
                    }
                }
            }
        }
    }, [messages]);

    const handleSubmit = async ({ text, files }: { text: string; files: FileUIPart[] }) => {
        const parts: Array<{ type: string; text?: string; url?: string; mediaType?: string; filename?: string }> = [];

        for (const file of files) {
            parts.push({
                type: 'file',
                url: file.url,
                mediaType: file.mediaType,
                filename: file.filename,
            });
        }

        if (text.trim()) {
            parts.push({ type: 'text', text });
        } else if (files.length > 0) {
            parts.push({ type: 'text', text: 'Analyze this document and suggest extraction fields.' });
        }

        if (parts.length > 0) {
            sendMessage({ parts });
        }

        setInput('');
    };

    const isLoading = status === 'submitted' || status === 'streaming';

    return (
        <div className="max-w-4xl mx-auto p-6 h-screen flex flex-col">
            <header className="mb-6">
                <h1 className="text-2xl font-semibold">Extraction Field Generator</h1>
                <p className="text-muted-foreground">
                    Upload a PDF or image to generate extraction fields, then refine through conversation.
                </p>
            </header>

            <Conversation className="flex-1">
                <ConversationContent>
                    {messages.map((message) => (
                        <div key={message.id}>
                            {/* Render file attachments first */}
                            {message.parts.some(part => part.type === 'file') && (
                                <MessageAttachments>
                                    {message.parts
                                        .filter(part => part.type === 'file')
                                        .map((part, i) => (
                                            <MessageAttachment
                                                key={`${message.id}-file-${i}`}
                                                data={part as FileUIPart}
                                            />
                                        ))}
                                </MessageAttachments>
                            )}

                            {message.parts.map((part, i) => {
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

                                if (part.type === 'text') {
                                    const schema = tryParseSchema(part.text);

                                    if (schema && message.role === 'assistant') {
                                        return (
                                            <Message key={`${message.id}-${i}`} from={message.role}>
                                                <MessageContent>
                                                    <SchemaTable schema={schema} />
                                                </MessageContent>
                                            </Message>
                                        );
                                    }

                                    return (
                                        <Message key={`${message.id}-${i}`} from={message.role}>
                                            <MessageContent>
                                                <MessageResponse>{part.text}</MessageResponse>
                                            </MessageContent>
                                        </Message>
                                    );
                                }

                                // Skip file parts as they're rendered separately above
                                if (part.type === 'file') {
                                    return null;
                                }

                                return null;
                            })}

                            {/* Show skeleton if assistant is thinking/streaming but hasn't produced text yet */}
                            {message.role === 'assistant' &&
                                status === 'streaming' &&
                                !message.parts.some(part => part.type === 'text') && (
                                    <SchemaTableSkeleton />
                                )}
                        </div>
                    ))}
                    {status === 'submitted' && <SchemaTableSkeleton />}
                </ConversationContent>
                <ConversationScrollButton />
            </Conversation>

            <PromptInput onSubmit={handleSubmit} accept="application/pdf,image/*" className="mt-4">
                <PromptInputAttachments>
                    {(attachment) => <PromptInputAttachment data={attachment} />}
                </PromptInputAttachments>
                <PromptInputBody>
                    <PromptInputTextarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={
                            currentSchema
                                ? 'Describe changes (e.g., "add invoice_number field", "remove address")'
                                : 'Upload a document to analyze...'
                        }
                        disabled={isLoading}
                    />
                </PromptInputBody>
                <PromptInputFooter>
                    <PromptInputTools>
                        <PromptInputActionMenu>
                            <PromptInputActionMenuTrigger />
                            <PromptInputActionMenuContent>
                                <PromptInputActionAddAttachments label="Upload PDF or Image" />
                            </PromptInputActionMenuContent>
                        </PromptInputActionMenu>
                    </PromptInputTools>
                    <PromptInputSubmit disabled={isLoading} status={status} />
                </PromptInputFooter>
            </PromptInput>
        </div>
    );
}
