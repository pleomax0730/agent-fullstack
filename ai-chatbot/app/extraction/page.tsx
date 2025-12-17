'use client';

import { useState, useEffect, useMemo } from 'react';
import { MessageSquare, X, FileText, Image as ImageIcon } from 'lucide-react';
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
import {
    ResizablePanelGroup,
    ResizablePanel,
    ResizableHandle,
} from '@/components/ui/resizable';
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
    const [isChatOpen, setIsChatOpen] = useState(true);
    const [activeFileIndex, setActiveFileIndex] = useState(0);

    const { messages, sendMessage, status } = useChat({
        transport: new DefaultChatTransport({
            api: `${BACKEND_URL}/api/chat`,
        }),
        onError: (error) => {
            console.error('Chat error:', error);
        },
    });

    // Extract all uploaded files from messages
    const uploadedFiles = useMemo(() => {
        const files: FileUIPart[] = [];
        for (const msg of messages) {
            for (const part of msg.parts) {
                if (part.type === 'file') {
                    files.push(part as FileUIPart);
                }
            }
        }
        return files;
    }, [messages]);

    // Auto-select latest file when new file is uploaded
    useEffect(() => {
        if (uploadedFiles.length > 0) {
            setActiveFileIndex(uploadedFiles.length - 1);
        }
    }, [uploadedFiles.length]);

    // Auto-expand chat when streaming
    useEffect(() => {
        if (status === 'streaming' || status === 'submitted') {
            setIsChatOpen(true);
        }
    }, [status]);

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
        const parts: Array<{ type: 'file'; url: string; mediaType: string; filename?: string } | { type: 'text'; text: string }> = [];

        for (const file of files) {
            parts.push({
                type: 'file' as const,
                url: file.url,
                mediaType: file.mediaType,
                filename: file.filename,
            });
        }

        if (text.trim()) {
            parts.push({ type: 'text' as const, text });
        } else if (files.length > 0) {
            parts.push({ type: 'text' as const, text: 'Analyze this document and suggest extraction fields.' });
        }

        if (parts.length > 0) {
            sendMessage({ parts });
        }

        setInput('');
    };

    const isLoading = status === 'submitted' || status === 'streaming';
    const activeFile = uploadedFiles[activeFileIndex];

    return (
        <div className="h-screen flex flex-col">
            <header className="p-4 border-b shrink-0 flex items-center justify-between">
                <div>
                    <h1 className="text-xl font-semibold">欄位提取工具</h1>
                    <p className="text-sm text-muted-foreground">
                        上傳文件，自動生成提取欄位，並透過對話調整
                    </p>
                </div>
                {/* Chat toggle button in header when collapsed */}
                {!isChatOpen && (
                    <button
                        onClick={() => setIsChatOpen(true)}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                    >
                        <MessageSquare size={16} />
                        <span className="text-sm">開啟對話</span>
                    </button>
                )}
            </header>

            <div className="flex-1 flex overflow-hidden">
                {/* Main content: Document + Table */}
                <ResizablePanelGroup direction="horizontal" className="flex-1">
                    {/* Left Panel: Document Preview */}
                    <ResizablePanel defaultSize={45} minSize={25}>
                        <div className="h-full flex flex-col border-r">
                            {/* File selector header */}
                            {uploadedFiles.length > 0 && (
                                <div className="p-3 border-b bg-muted/30 flex items-center gap-2">
                                    <span className="text-sm font-medium">文件預覽</span>
                                    {uploadedFiles.length > 1 && (
                                        <select
                                            value={activeFileIndex}
                                            onChange={(e) => setActiveFileIndex(Number(e.target.value))}
                                            className="text-sm bg-background border rounded px-2 py-1"
                                        >
                                            {uploadedFiles.map((file, i) => (
                                                <option key={i} value={i}>
                                                    {file.filename || `文件 ${i + 1}`}
                                                </option>
                                            ))}
                                        </select>
                                    )}
                                </div>
                            )}
                            {/* File viewer */}
                            <div className="flex-1 overflow-auto p-4">
                                {activeFile ? (
                                    activeFile.mediaType?.startsWith('image/') ? (
                                        <img
                                            src={activeFile.url}
                                            alt={activeFile.filename || 'Preview'}
                                            className="max-w-full h-auto rounded-lg border"
                                        />
                                    ) : activeFile.mediaType === 'application/pdf' ? (
                                        <iframe
                                            src={activeFile.url}
                                            className="w-full h-full rounded-lg border"
                                            title={activeFile.filename || 'PDF Preview'}
                                        />
                                    ) : (
                                        <div className="text-muted-foreground text-center py-8">
                                            不支援預覽此檔案類型
                                        </div>
                                    )
                                ) : (
                                    <div className="h-full flex items-center justify-center text-muted-foreground">
                                        <div className="text-center space-y-3">
                                            <FileText size={48} className="mx-auto opacity-30" />
                                            <p>尚未上傳文件</p>
                                            <p className="text-sm">在右側對話區上傳文件</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </ResizablePanel>

                    <ResizableHandle withHandle />

                    {/* Right Panel: Extraction Results */}
                    <ResizablePanel defaultSize={55} minSize={30}>
                        <div className="h-full p-4 overflow-auto">
                            {currentSchema ? (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <h2 className="text-lg font-medium">提取結果</h2>
                                        <span className="text-xs text-muted-foreground">
                                            {currentSchema.fields.length} 個欄位
                                        </span>
                                    </div>
                                    <SchemaTable schema={currentSchema} />
                                </div>
                            ) : (
                                <div className="h-full flex items-center justify-center text-muted-foreground">
                                    <div className="text-center space-y-3">
                                        <ImageIcon size={48} className="mx-auto opacity-30" />
                                        <p className="text-lg">尚無提取結果</p>
                                        <p className="text-sm">上傳文件後將自動分析</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </ResizablePanel>
                </ResizablePanelGroup>

                {/* Collapsible Chat Panel */}
                <div
                    className={`border-l transition-all duration-300 flex flex-col ${isChatOpen ? 'w-[380px]' : 'w-0'
                        } overflow-hidden`}
                >
                    {isChatOpen && (
                        <>
                            {/* Chat header with close button */}
                            <div className="p-3 border-b bg-muted/30 flex items-center justify-between shrink-0">
                                <span className="text-sm font-medium">對話</span>
                                <button
                                    onClick={() => setIsChatOpen(false)}
                                    className="p-1 rounded hover:bg-muted transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>

                            {/* Chat content */}
                            <Conversation className="flex-1">
                                <ConversationContent>
                                    {messages.map((message) => (
                                        <div key={message.id}>
                                            {/* File attachments shown as small thumbnails in chat */}
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
                                                                    <p className="text-sm text-muted-foreground">
                                                                        ✅ 已更新提取結果
                                                                    </p>
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

                                                if (part.type === 'file') return null;
                                                return null;
                                            })}

                                            {message.role === 'assistant' &&
                                                status === 'streaming' &&
                                                !message.parts.some(part => part.type === 'text') && (
                                                    <div className="flex items-center gap-2 text-sm text-muted-foreground p-4">
                                                        <Loader size={14} />
                                                        <Shimmer duration={1.5}>正在分析...</Shimmer>
                                                    </div>
                                                )}
                                        </div>
                                    ))}
                                    {status === 'submitted' && (
                                        <div className="flex items-center gap-2 text-sm text-muted-foreground p-4">
                                            <Loader size={14} />
                                            <Shimmer duration={1.5}>正在分析...</Shimmer>
                                        </div>
                                    )}
                                </ConversationContent>
                                <ConversationScrollButton />
                            </Conversation>

                            {/* Chat input */}
                            <div className="p-3 border-t shrink-0">
                                <PromptInput onSubmit={handleSubmit} accept="application/pdf,image/*">
                                    <PromptInputAttachments>
                                        {(attachment) => <PromptInputAttachment data={attachment} />}
                                    </PromptInputAttachments>
                                    <PromptInputBody>
                                        <PromptInputTextarea
                                            value={input}
                                            onChange={(e) => setInput(e.target.value)}
                                            placeholder={
                                                currentSchema
                                                    ? '描述修改...'
                                                    : '上傳文件開始...'
                                            }
                                            disabled={isLoading}
                                        />
                                    </PromptInputBody>
                                    <PromptInputFooter>
                                        <PromptInputTools>
                                            <PromptInputActionMenu>
                                                <PromptInputActionMenuTrigger />
                                                <PromptInputActionMenuContent>
                                                    <PromptInputActionAddAttachments label="上傳檔案" />
                                                </PromptInputActionMenuContent>
                                            </PromptInputActionMenu>
                                        </PromptInputTools>
                                        <PromptInputSubmit disabled={isLoading} status={status} />
                                    </PromptInputFooter>
                                </PromptInput>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
