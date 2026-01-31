'use client';

import { useState, useEffect, useMemo } from 'react';
import { MessageSquare, X, FileText, Image as ImageIcon, Database, AlertCircle, Bot, Check, Search, PenTool, Table2 } from 'lucide-react';
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
import {
    Tool,
    ToolHeader,
    ToolContent,
    ToolInput,
    ToolOutput,
} from '@/components/ai-elements/tool';
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { FileUIPart, ToolUIPart } from 'ai';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

type FieldType = 'string' | 'int' | 'float' | 'boolean';

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

interface ExtractedData {
    [key: string]: string | number | null;
}

interface AgentInfo {
    agent_id: string;
    name: string;
    task: string;
    description?: string;
}

interface QueryResultItem {
    [key: string]: string | number | null;
}

interface QueryResults {
    items: QueryResultItem[];
    query: string;
    agentName?: string;
}

// -----------------------------------------------------------------------------
// Schema Display
// -----------------------------------------------------------------------------

function FieldTypeTag({ type }: { type: FieldType }) {
    const colors: Record<FieldType, string> = {
        string: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
        int: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
        float: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
        boolean: 'bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-100',
    };
    return (
        <span className={`px-2 py-0.5 rounded text-xs font-mono ${colors[type]}`}>
            {type}
        </span>
    );
}

function TruncatedDescription({ text, maxLength = 80 }: { text: string; maxLength?: number }) {
    const needsTruncation = text.length > maxLength;
    const displayText = needsTruncation ? text.slice(0, maxLength) + '...' : text;

    if (!needsTruncation) {
        return <span>{text}</span>;
    }

    return (
        <TooltipProvider>
            <Tooltip delayDuration={200}>
                <TooltipTrigger asChild>
                    <span className="cursor-help border-b border-dotted border-muted-foreground/50">
                        {displayText}
                    </span>
                </TooltipTrigger>
                <TooltipContent side="bottom" align="start" className="max-w-md">
                    <p className="text-xs whitespace-pre-wrap break-words">{text}</p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
}

function SchemaTable({ schema }: { schema: ExtractionSchema }) {
    return (
        <div className="rounded-lg border overflow-hidden max-h-[400px] overflow-y-auto">
            <table className="w-full text-sm">
                <thead className="bg-background sticky top-0 shadow-sm">
                    <tr>
                        <th className="px-4 py-3 text-left font-medium whitespace-nowrap w-[140px]">欄位名稱</th>
                        <th className="px-4 py-3 text-left font-medium w-[70px]">類型</th>
                        <th className="px-4 py-3 text-left font-medium">說明</th>
                    </tr>
                </thead>
                <tbody className="divide-y">
                    {schema.fields.map((field, i) => (
                        <tr key={i} className="hover:bg-muted/30">
                            <td className="px-4 py-3 font-medium">
                                <TooltipProvider>
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <span className="cursor-help border-b border-dotted border-muted-foreground truncate block">
                                                {field.label}
                                            </span>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                            <p className="font-mono text-xs">{field.field_name}</p>
                                        </TooltipContent>
                                    </Tooltip>
                                </TooltipProvider>
                            </td>
                            <td className="px-4 py-3"><FieldTypeTag type={field.type} /></td>
                            <td className="px-4 py-3 text-muted-foreground text-xs">
                                <TruncatedDescription text={field.description} maxLength={80} />
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function ExtractedDataTable({ data }: { data: ExtractedData }) {
    // Get all keys from the data object
    const entries = Object.entries(data).filter(([_, v]) => v !== null);
    
    return (
        <div className="space-y-4">
            <div className="rounded-lg border overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                        <tr>
                            <th className="px-4 py-3 text-left font-medium whitespace-nowrap">欄位名稱</th>
                            <th className="px-4 py-3 text-left font-medium">提取值</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {entries.map(([key, value], i) => (
                            <tr key={i} className="hover:bg-muted/30">
                                <td className="px-4 py-3 font-medium whitespace-nowrap">{key}</td>
                                <td className="px-4 py-3 text-foreground/90">
                                    {String(value)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

// Skeleton shimmer for loading state
function SchemaTableSkeleton() {
    return (
        <div className="space-y-4">
            <div className="rounded-lg border overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                        <tr>
                            <th className="px-4 py-3 text-left font-medium">欄位名稱</th>
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

function DataExtractionSkeleton() {
    return (
        <div className="space-y-4">
            <div className="rounded-lg border overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                        <tr>
                            <th className="px-4 py-3 text-left font-medium whitespace-nowrap">欄位名稱</th>
                            <th className="px-4 py-3 text-left font-medium">提取值</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {[1, 2, 3, 4, 5].map((i) => (
                            <tr key={i} className="animate-pulse">
                                <td className="px-4 py-3">
                                    <div className="h-4 w-24 bg-muted rounded" />
                                </td>
                                <td className="px-4 py-3">
                                    <div className="h-4 w-full max-w-[200px] bg-muted rounded" />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader size={14} />
                <Shimmer duration={1.5}>正在擷取資料中...</Shimmer>
            </div>
        </div>
    );
}

function AgentCard({ 
    agent, 
    isSelected, 
    onClick 
}: { 
    agent: AgentInfo; 
    isSelected?: boolean;
    onClick?: () => void;
}) {
    return (
        <button
            onClick={onClick}
            className={`w-full text-left rounded-lg border p-3 transition-all ${
                isSelected 
                    ? 'border-primary bg-primary/5 ring-1 ring-primary' 
                    : 'hover:border-primary/50 hover:bg-muted/30'
            }`}
        >
            <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                    <div className={`p-1.5 rounded-lg shrink-0 ${isSelected ? 'bg-primary/20' : 'bg-primary/10'}`}>
                        <Bot size={16} className="text-primary" />
                    </div>
                    <span className="font-medium truncate">{agent.name}</span>
                </div>
                {isSelected && (
                    <div className="p-0.5 rounded-full bg-primary text-primary-foreground shrink-0">
                        <Check size={12} />
                    </div>
                )}
            </div>
            <p className="text-xs text-muted-foreground line-clamp-2 mt-1.5 ml-8">{agent.task}</p>
        </button>
    );
}

function AgentList({ 
    agents, 
    selectedAgentId, 
    onSelect 
}: { 
    agents: AgentInfo[]; 
    selectedAgentId?: string;
    onSelect: (agent: AgentInfo) => void;
}) {
    if (agents.length === 0) return null;
    
    return (
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {agents.map((agent) => (
                <AgentCard
                    key={agent.agent_id}
                    agent={agent}
                    isSelected={agent.agent_id === selectedAgentId}
                    onClick={() => onSelect(agent)}
                />
            ))}
        </div>
    );
}

function AgentSearchSkeleton() {
    return (
        <div className="space-y-4">
            <div className="rounded-lg border p-4 space-y-3 animate-pulse">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-muted w-10 h-10" />
                    <div className="space-y-2">
                        <div className="h-5 w-32 bg-muted rounded" />
                        <div className="h-3 w-48 bg-muted rounded" />
                    </div>
                </div>
                <div className="space-y-2">
                    <div className="h-3 w-16 bg-muted rounded" />
                    <div className="h-4 w-full bg-muted rounded" />
                    <div className="h-4 w-3/4 bg-muted rounded" />
                </div>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader size={14} />
                <Shimmer duration={1.5}>正在搜尋小幫手...</Shimmer>
            </div>
        </div>
    );
}

function QueryResultsTable({ results }: { results: QueryResults }) {
    if (results.items.length === 0) {
        return (
            <div className="text-center py-8 text-muted-foreground">
                <p>找不到符合條件的資料</p>
            </div>
        );
    }

    // Get all unique keys from all items
    const allKeys = Array.from(
        new Set(results.items.flatMap(item => Object.keys(item)))
    );

    return (
        <div className="rounded-lg border overflow-hidden">
            <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                <table className="w-full text-sm">
                    <thead className="bg-background sticky top-0 shadow-sm">
                        <tr>
                            <th className="px-3 py-2 text-left font-medium text-xs text-muted-foreground w-10">#</th>
                            {allKeys.map((key) => (
                                <th key={key} className="px-3 py-2 text-left font-medium text-xs whitespace-nowrap">
                                    {key}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {results.items.map((item, i) => (
                            <tr key={i} className="hover:bg-muted/30">
                                <td className="px-3 py-2 text-muted-foreground text-xs">{i + 1}</td>
                                {allKeys.map((key) => (
                                    <td key={key} className="px-3 py-2 max-w-[200px]">
                                        <TruncatedDescription 
                                            text={item[key] !== null ? String(item[key]) : '-'} 
                                            maxLength={50} 
                                        />
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function QueryResultsSkeleton() {
    return (
        <div className="space-y-4">
            <div className="rounded-lg border overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                        <tr>
                            <th className="px-3 py-2 text-left font-medium w-10">#</th>
                            <th className="px-3 py-2 text-left font-medium">欄位 1</th>
                            <th className="px-3 py-2 text-left font-medium">欄位 2</th>
                            <th className="px-3 py-2 text-left font-medium">欄位 3</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {[1, 2, 3, 4, 5].map((i) => (
                            <tr key={i} className="animate-pulse">
                                <td className="px-3 py-2">
                                    <div className="h-4 w-6 bg-muted rounded" />
                                </td>
                                <td className="px-3 py-2">
                                    <div className="h-4 w-24 bg-muted rounded" />
                                </td>
                                <td className="px-3 py-2">
                                    <div className="h-4 w-32 bg-muted rounded" />
                                </td>
                                <td className="px-3 py-2">
                                    <div className="h-4 w-20 bg-muted rounded" />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader size={14} />
                <Shimmer duration={1.5}>正在查詢資料...</Shimmer>
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

function tryParseExtractedData(text: string): ExtractedData | null {
    try {
        const parsed = JSON.parse(text);
        if (typeof parsed === 'object' && !Array.isArray(parsed) && !parsed.fields) {
            return parsed as ExtractedData;
        }
    } catch {
        // Not valid JSON
    }
    return null;
}

/**
 * Convert hub-agent schema format to frontend ExtractionSchema format.
 * Hub format: {name, task, fields: [{name, display_name, field_type, description}]}
 * Frontend format: {fields: [{label, field_name, value, type, description}], reasoning}
 */
function normalizeHubSchema(hubSchema: any): ExtractionSchema {
    const fields: ExtractionField[] = (hubSchema.fields || []).map((f: any) => ({
        label: f.display_name || f.name || '',
        field_name: f.name || '',
        value: null,
        type: (f.field_type || 'STRING').toLowerCase() as FieldType,
        description: f.description || '',
    }));
    
    return {
        fields,
        reasoning: `小幫手名稱: ${hubSchema.name || ''}\n任務: ${hubSchema.task || ''}`,
    };
}

function normalizeAgentFieldsToSchema(fields: any[], agent: AgentInfo | null): ExtractionSchema {
    const mapped: ExtractionField[] = (fields || []).map((f: any) => {
        const rawType = (f?.field_type || f?.type || 'STRING').toString().toLowerCase();
        const type: FieldType = (rawType === 'int' || rawType === 'float' || rawType === 'boolean') ? (rawType as FieldType) : 'string';
        return {
            label: f?.display_name || f?.label || f?.name || '',
            field_name: f?.name || f?.field_name || '',
            value: null,
            type,
            description: f?.description || '',
        };
    });

    return {
        fields: mapped,
        reasoning: agent ? `小幫手名稱: ${agent.name}\n任務: ${agent.task}` : '',
    };
}

// Helper to map tool state from SSE events
function getToolState(toolPart: any): ToolUIPart['state'] {
    if (toolPart.output !== undefined) {
        if (toolPart.output?.error) return 'output-error';
        return 'output-available';
    }
    if (toolPart.input !== undefined) {
        return 'input-available';
    }
    return 'input-streaming';
}


// -----------------------------------------------------------------------------
// Main Page
// -----------------------------------------------------------------------------

export default function HubAgentPage() {
    const [input, setInput] = useState('');
    const [currentSchema, setCurrentSchema] = useState<ExtractionSchema | null>(null);
    const [extractedData, setExtractedData] = useState<ExtractedData | null>(null);
    const [extractionError, setExtractionError] = useState<string | null>(null);
    const [availableAgents, setAvailableAgents] = useState<AgentInfo[]>([]);
    const [cachedAgentFields, setCachedAgentFields] = useState<any[] | null>(null);
    const [selectedAgent, setSelectedAgent] = useState<AgentInfo | null>(null);
    const [queryResults, setQueryResults] = useState<QueryResults | null>(null);
    const [isChatOpen, setIsChatOpen] = useState(true);
    const [activeFileIndex, setActiveFileIndex] = useState(0);
    const [activeTab, setActiveTab] = useState<string>('agent');
    const [agentSubTab, setAgentSubTab] = useState<string>('search');

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

    // Check current activity state for shimmer messages
    const activityState = useMemo(() => {
        const lastAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant');
        if (!lastAssistantMsg) return { isDesigningSchema: false, activeTool: null };
        
        let isDesigningSchema = false;
        let activeTool: string | null = null;
        
        for (const part of lastAssistantMsg.parts) {
            // Tool parts have type like 'tool-search-agents', 'tool-design-agent-schema', etc.
            if (part.type.startsWith('tool-')) {
                const toolName = part.type.replace('tool-', '');
                const toolPart = part as any;
                const state = toolPart.state;
                
                // Tool is active if state is 'input-streaming' or 'input-available'
                // 'output-available' means the tool has finished
                if (state === 'input-streaming' || state === 'input-available') {
                    activeTool = toolName;
                    if (toolName === 'design-agent-schema') {
                        isDesigningSchema = true;
                    }
                }
            }
        }
        
        return { isDesigningSchema, activeTool };
    }, [messages]);

    const { isDesigningSchema, activeTool } = activityState;

    // Auto-switch tabs based on active tool
    useEffect(() => {
        if (activeTool === 'extract-data') {
            setActiveTab('data');
        } else if (activeTool === 'retrieve-data') {
            setActiveTab('query');
        } else if (activeTool === 'search-agents') {
            setActiveTab('agent');
            setAgentSubTab('search');
        } else if (activeTool === 'design-agent-schema') {
            setActiveTab('agent');
            setAgentSubTab('design');
        }
    }, [activeTool]);

    // Get shimmer message based on current state
    const getShimmerMessage = () => {
        if (activeTool) {
            const toolMessages: Record<string, string> = {
                'design-agent-schema': '正在設計欄位...',
                'extract-data': '正在擷取資料...',
                'search-agents': '正在搜尋小幫手...',
                'retrieve-data': '正在查詢資料...',
                'query-knowledge-base': '正在查詢資料庫...',
            };
            return toolMessages[activeTool] || `正在執行 ${activeTool}...`;
        }
        if (status === 'streaming') return '正在思考...';
        return '正在處理...';
    };

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

    // Extract schema and data from tool results and update tab states
    // Use refs to track processed message IDs to avoid re-processing
    const processedSchemaRefs = useMemo(() => new Set<string>(), []);
    const processedDataRefs = useMemo(() => new Set<string>(), []);
    const processedAgentRefs = useMemo(() => new Set<string>(), []);
    const processedQueryRefs = useMemo(() => new Set<string>(), []);
    
    useEffect(() => {
        let latestSchema: ExtractionSchema | null = null;
        let latestData: ExtractedData | null = null;
        let latestAgents: AgentInfo[] = [];
        let latestSelectedAgent: AgentInfo | null = null;
        let latestQueryResults: QueryResults | null = null;
        
        for (const message of messages) {
            if (message.role !== 'assistant') continue;
            
            for (const part of message.parts) {
                // Check for tool invocation with output
                if (part.type === 'tool-invocation' || part.type.startsWith('tool-')) {
                    const toolPart = part as any;
                    const toolName = toolPart.toolName || part.type.replace('tool-', '');
                    const output = toolPart.output || toolPart.result;
                    
                    if (!output) continue;
                    
                    // Handle search-agents results
                    if (toolName === 'search-agents') {
                        const key = `${message.id}-agent`;
                        if (!processedAgentRefs.has(key)) {
                            try {
                                const parsed = typeof output === 'string' ? JSON.parse(output) : output;
                                if (parsed.agents && Array.isArray(parsed.agents) && parsed.agents.length > 0) {
                                    latestAgents = parsed.agents as AgentInfo[];
                                    // Don't auto-select - user needs to click to select
                                }
                                // Cache fields for later use when user selects an agent
                                if (parsed.fields && Array.isArray(parsed.fields) && parsed.fields.length > 0) {
                                    setCachedAgentFields(parsed.fields);
                                }

                                processedAgentRefs.add(key);
                            } catch (e) {
                                console.error('Failed to parse agent:', e);
                            }
                        }
                    }
                    
                    // Handle design-agent-schema results
                    if (toolName === 'design-agent-schema') {
                        const key = `${message.id}-schema`;
                        if (!processedSchemaRefs.has(key)) {
                            try {
                                const parsed = typeof output === 'string' ? JSON.parse(output) : output;
                                // The result might be wrapped in extraction_schema or schema
                                const schemaData = parsed.extraction_schema || parsed.schema || parsed;
                                
                                // Check if it has fields array
                                if (schemaData.fields && Array.isArray(schemaData.fields)) {
                                    latestSchema = schemaData.fields[0]?.label 
                                        ? schemaData  // Already frontend format
                                        : normalizeHubSchema(schemaData);  // Hub format, needs conversion
                                    processedSchemaRefs.add(key);
                                }
                            } catch (e) {
                                console.error('Failed to parse schema:', e);
                            }
                        }
                    }
                    
                    // Handle extract-data results
                    if (toolName === 'extract-data') {
                        const key = `${message.id}-data`;
                        if (!processedDataRefs.has(key)) {
                            try {
                                const parsed = typeof output === 'string' ? JSON.parse(output) : output;
                                
                                // Check for error response
                                if (parsed.error) {
                                    setExtractionError(parsed.error);
                                    setExtractedData(null);
                                    processedDataRefs.add(key);
                                } else {
                                    // The result might be the extracted entity directly or wrapped
                                    const entity = parsed.entity || parsed.extracted || parsed;
                                    if (typeof entity === 'object' && !Array.isArray(entity)) {
                                        latestData = entity;
                                        setExtractionError(null);  // Clear any previous error
                                        processedDataRefs.add(key);
                                    }
                                }
                            } catch (e) {
                                console.error('Failed to parse extracted data:', e);
                                setExtractionError('解析擷取結果失敗');
                            }
                        }
                    }
                    
                    // Handle retrieve-data results
                    if (toolName === 'retrieve-data') {
                        const key = `${message.id}-query`;
                        if (!processedQueryRefs.has(key)) {
                            try {
                                const parsed = typeof output === 'string' ? JSON.parse(output) : output;
                                
                                // API returns: { mode, field_definitions, entities, next_cursor }
                                // entities is an array of { extracted_fields: {...}, ... }
                                // We only care about extracted_fields from each entity
                                let items: QueryResultItem[] = [];
                                if (parsed.entities && Array.isArray(parsed.entities)) {
                                    // Extract the extracted_fields from each entity
                                    items = parsed.entities.map((entity: any) => entity.extracted_fields || entity);
                                } else if (Array.isArray(parsed)) {
                                    items = parsed.map((entity: any) => entity.extracted_fields || entity);
                                } else if (parsed.results && Array.isArray(parsed.results)) {
                                    items = parsed.results.map((entity: any) => entity.extracted_fields || entity);
                                } else if (parsed.data && Array.isArray(parsed.data)) {
                                    items = parsed.data.map((entity: any) => entity.extracted_fields || entity);
                                }
                                
                                // Get query from tool input
                                const toolInput = toolPart.input || toolPart.args || {};
                                latestQueryResults = {
                                    items,
                                    query: toolInput.query || '',
                                    agentName: selectedAgent?.name,
                                };
                                processedQueryRefs.add(key);
                            } catch (e) {
                                console.error('Failed to parse query results:', e);
                            }
                        }
                    }
                }
                
                // Also check for schema/data in text parts (bridge emits schema as text)
                if (part.type === 'text') {
                    const textKey = `${message.id}-${part.text.slice(0, 50)}`;
                    
                    const schema = tryParseSchema(part.text);
                    if (schema && !processedSchemaRefs.has(textKey + '-schema')) {
                        latestSchema = schema;
                        processedSchemaRefs.add(textKey + '-schema');
                    }
                    
                    const data = tryParseExtractedData(part.text);
                    if (data && !processedDataRefs.has(textKey + '-data')) {
                        latestData = data;
                        processedDataRefs.add(textKey + '-data');
                    }
                }
            }
        }
        
        // Update states with latest found values
        if (latestAgents.length > 0) {
            setAvailableAgents(latestAgents);
            // Clear selection when new agents are found - user needs to select
            setSelectedAgent(null);
            setActiveTab('agent');
            setAgentSubTab('search');
        }
        // Only set schema from design-agent-schema tool (not from search-agents)
        if (latestSchema) {
            setCurrentSchema(latestSchema);
            setActiveTab('agent');
            setAgentSubTab('design');
        }
        if (latestData) {
            setExtractedData(latestData);
            setActiveTab('data');
        }
        if (latestQueryResults) {
            setQueryResults(latestQueryResults);
            setActiveTab('query');
        }
    }, [messages, processedSchemaRefs, processedDataRefs, processedAgentRefs, processedQueryRefs, selectedAgent]);

    // Check for search-agents results to show suggestion buttons
    const suggestionButtons = useMemo(() => {
        const buttons: Array<{ label: string; action: string }> = [];
        
        // Find the last assistant message
        const lastAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant');
        
        // Only show buttons if the LAST message is from assistant and contains search-agents tool
        // This ensures buttons disappear after user replies
        if (lastAssistantMsg && messages[messages.length - 1].id === lastAssistantMsg.id) {
            let hasSearchAgents = false;
            
            for (const part of lastAssistantMsg.parts) {
                // Check for tool-search-agents type (Vercel AI SDK format with tool- prefix)
                if (part.type === 'tool-search-agents' || (part.type === 'tool-invocation' && (part as any).toolName === 'search-agents')) {
                    hasSearchAgents = true;
                    const toolPart = part as any;
                    
                    // Only show suggestions if tool is finished and has output
                    const output = toolPart.output || toolPart.result;
                    if (output) {
                        try {
                            const parsedOutput = typeof output === 'string' 
                                ? JSON.parse(output) 
                                : output;
                            
                            if (parsedOutput.agents && Array.isArray(parsedOutput.agents)) {
                                // Add button for each found agent
                                parsedOutput.agents.forEach((agent: any) => {
                                    buttons.push({
                                        label: `切換至「${agent.name}」`,
                                        action: `我想使用「${agent.name}」`
                                    });
                                });
                            }
                        } catch (e) {
                            console.error('Failed to parse search-agents output:', e);
                        }
                    }
                }
            }
            
            // If we found agents, or if we ran search-agents but found nothing
            if (hasSearchAgents) {
                buttons.push({
                    label: '建立新小幫手',
                    action: '我想建立一個新的小幫手'
                });
            }
        }
        
        return buttons;
    }, [messages]);

    const handleSuggestionClick = (text: string) => {
        handleSubmit({ text, files: [] });
    };

    // Toggle agent selection to view its fields - don't trigger chat
    const handleAgentSelect = (agent: AgentInfo) => {
        // Toggle: if already selected, deselect
        if (selectedAgent?.agent_id === agent.agent_id) {
            setSelectedAgent(null);
            // Only clear schema if it came from cached fields (search result)
            // Don't clear if it came from design-agent-schema
            if (cachedAgentFields && cachedAgentFields.length > 0) {
                setCurrentSchema(null);
            }
            return;
        }
        
        setSelectedAgent(agent);
        // Only load schema from cached fields if no schema exists yet
        // This preserves schema from design-agent-schema
        if (!currentSchema && cachedAgentFields && cachedAgentFields.length > 0) {
            const schema = normalizeAgentFieldsToSchema(cachedAgentFields, agent);
            setCurrentSchema(schema);
        }
    };

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
        }
        // Don't auto-add text when only file is uploaded - let agent ask

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
                    <h1 className="text-xl font-semibold">KMind Hub</h1>
                    <p className="text-sm text-muted-foreground">
                        上傳文件，設計提取欄位，並執行資料擷取
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
                {/* Main content: Document + Tabs */}
                <ResizablePanelGroup direction="horizontal" className="flex-1">
                    {/* Left Panel: Document Preview */}
                    <ResizablePanel defaultSize={40} minSize={25}>
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

                    {/* Right Panel: Tabbed Results */}
                    <ResizablePanel defaultSize={60} minSize={35}>
                        <div className="h-full flex flex-col">
                            <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
                                <div className="border-b px-4 pt-2">
                                    <TabsList className="grid w-full max-w-lg grid-cols-3">
                                        <TabsTrigger value="agent" className="flex items-center gap-2">
                                            <Bot size={14} />
                                            小幫手設定
                                        </TabsTrigger>
                                        <TabsTrigger value="data" className="flex items-center gap-2">
                                            <Database size={14} />
                                            擷取結果
                                        </TabsTrigger>
                                        <TabsTrigger value="query" className="flex items-center gap-2">
                                            <Table2 size={14} />
                                            查詢結果
                                            {queryResults && queryResults.items.length > 0 && (
                                                <span className="ml-1 px-1.5 py-0.5 rounded-full bg-primary/10 text-primary text-[10px]">
                                                    {queryResults.items.length}
                                                </span>
                                            )}
                                        </TabsTrigger>
                                    </TabsList>
                                </div>

                                <TabsContent value="agent" className="flex-1 flex flex-col mt-0 overflow-hidden">
                                    {/* Sub-tabs for agent section */}
                                    <Tabs value={agentSubTab} onValueChange={setAgentSubTab} className="flex-1 flex flex-col">
                                        <div className="px-4 pt-3 pb-2 border-b bg-muted/20">
                                            <TabsList className="h-8">
                                                <TabsTrigger value="search" className="text-xs px-3 h-7 gap-1.5">
                                                    <Search size={12} />
                                                    可用小幫手
                                                    {availableAgents.length > 0 && (
                                                        <span className="ml-1 px-1.5 py-0.5 rounded-full bg-primary/10 text-primary text-[10px]">
                                                            {availableAgents.length}
                                                        </span>
                                                    )}
                                                </TabsTrigger>
                                                <TabsTrigger value="design" className="text-xs px-3 h-7 gap-1.5">
                                                    <PenTool size={12} />
                                                    欄位設計
                                                    {currentSchema && (
                                                        <span className="ml-1 px-1.5 py-0.5 rounded-full bg-primary/10 text-primary text-[10px]">
                                                            {currentSchema.fields.length}
                                                        </span>
                                                    )}
                                                </TabsTrigger>
                                            </TabsList>
                                        </div>

                                        {/* Sub-tab: Search Results */}
                                        <TabsContent value="search" className="flex-1 p-4 overflow-auto mt-0">
                                            {activeTool === 'search-agents' ? (
                                                <AgentSearchSkeleton />
                                            ) : availableAgents.length > 0 ? (
                                                <div className="space-y-3">
                                                    <p className="text-xs text-muted-foreground">點擊選擇小幫手來查看其欄位定義</p>
                                                    <AgentList 
                                                        agents={availableAgents} 
                                                        selectedAgentId={selectedAgent?.agent_id}
                                                        onSelect={handleAgentSelect}
                                                    />
                                                    {/* Show selected agent's cached fields */}
                                                    {selectedAgent && cachedAgentFields && cachedAgentFields.length > 0 && (
                                                        <div className="mt-4 pt-4 border-t">
                                                            <h3 className="text-sm font-medium mb-2">「{selectedAgent.name}」的欄位 ({cachedAgentFields.length} 個)</h3>
                                                            <SchemaTable schema={normalizeAgentFieldsToSchema(cachedAgentFields, selectedAgent)} />
                                                        </div>
                                                    )}
                                                </div>
                                            ) : (
                                                <div className="h-48 flex items-center justify-center text-muted-foreground border rounded-lg bg-muted/10 border-dashed">
                                                    <div className="text-center space-y-3">
                                                        <Search size={48} className="mx-auto opacity-30" />
                                                        <p className="text-lg">尚未搜尋小幫手</p>
                                                        <p className="text-sm">請在對話中搜尋現有的小幫手</p>
                                                    </div>
                                                </div>
                                            )}
                                        </TabsContent>

                                        {/* Sub-tab: Schema Design */}
                                        <TabsContent value="design" className="flex-1 p-4 overflow-auto mt-0">
                                            {isDesigningSchema ? (
                                                <SchemaTableSkeleton />
                                            ) : currentSchema ? (
                                                <div className="space-y-3">
                                                    <p className="text-xs text-muted-foreground">透過上傳文件自動產生的欄位定義</p>
                                                    <SchemaTable schema={currentSchema} />
                                                </div>
                                            ) : (
                                                <div className="h-48 flex items-center justify-center text-muted-foreground border rounded-lg bg-muted/10 border-dashed">
                                                    <div className="text-center space-y-3">
                                                        <PenTool size={48} className="mx-auto opacity-30" />
                                                        <p className="text-lg">尚未設計欄位</p>
                                                        <p className="text-sm">上傳文件後系統會自動分析並設計欄位</p>
                                                    </div>
                                                </div>
                                            )}
                                        </TabsContent>
                                    </Tabs>
                                </TabsContent>

                                <TabsContent value="data" className="flex-1 p-4 overflow-auto mt-0">
                                    {activeTool === 'extract-data' ? (
                                        <DataExtractionSkeleton />
                                    ) : extractionError ? (
                                        <div className="h-full flex items-center justify-center">
                                            <div className="text-center space-y-3 max-w-md">
                                                <AlertCircle size={48} className="mx-auto text-destructive opacity-70" />
                                                <p className="text-lg font-medium text-destructive">擷取失敗</p>
                                                <p className="text-sm text-muted-foreground break-all">{extractionError}</p>
                                                <p className="text-xs text-muted-foreground">請檢查文件格式或重試</p>
                                            </div>
                                        </div>
                                    ) : extractedData ? (
                                        <div className="space-y-4">
                                            <div className="flex items-center justify-between">
                                                <h2 className="text-lg font-medium">擷取結果</h2>
                                                <span className="text-xs text-muted-foreground">
                                                    {Object.keys(extractedData).filter(k => extractedData[k] !== null).length} 個欄位
                                                </span>
                                            </div>
                                            <ExtractedDataTable data={extractedData} />
                                        </div>
                                    ) : (
                                        <div className="h-full flex items-center justify-center text-muted-foreground">
                                            <div className="text-center space-y-3">
                                                <Database size={48} className="mx-auto opacity-30" />
                                                <p className="text-lg">尚無擷取資料</p>
                                                <p className="text-sm">請先設計欄位後再擷取資料</p>
                                            </div>
                                        </div>
                                    )}
                                </TabsContent>

                                <TabsContent value="query" className="flex-1 p-4 overflow-auto mt-0">
                                    {activeTool === 'retrieve-data' ? (
                                        <QueryResultsSkeleton />
                                    ) : queryResults ? (
                                        <div className="space-y-4">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <h2 className="text-lg font-medium">查詢結果</h2>
                                                    {queryResults.query && (
                                                        <p className="text-xs text-muted-foreground mt-0.5">
                                                            查詢條件：{queryResults.query}
                                                        </p>
                                                    )}
                                                </div>
                                                <span className="text-xs text-muted-foreground">
                                                    {queryResults.items.length} 筆資料
                                                </span>
                                            </div>
                                            <QueryResultsTable results={queryResults} />
                                        </div>
                                    ) : (
                                        <div className="h-full flex items-center justify-center text-muted-foreground">
                                            <div className="text-center space-y-3">
                                                <Table2 size={48} className="mx-auto opacity-30" />
                                                <p className="text-lg">尚無查詢結果</p>
                                                <p className="text-sm">請先選擇小幫手後再查詢資料</p>
                                            </div>
                                        </div>
                                    )}
                                </TabsContent>
                            </Tabs>
                        </div>
                    </ResizablePanel>
                </ResizablePanelGroup>

                {/* Collapsible Chat Panel */}
                <div
                    className={`border-l transition-all duration-300 flex flex-col ${isChatOpen ? 'w-[400px]' : 'w-0'
                        } overflow-hidden`}
                >
                    {isChatOpen && (
                        <>
                            {/* Chat header with close button */}
                            <div className="px-4 py-2.5 border-b bg-muted/30 flex items-center justify-between shrink-0">
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

                                                // Display tool invocations
                                                if (part.type === 'tool-invocation') {
                                                    const toolPart = part as any;
                                                    return (
                                                        <Tool key={`${message.id}-${i}`} defaultOpen={false}>
                                                            <ToolHeader
                                                                title={toolPart.toolName}
                                                                type={`tool-${toolPart.toolName}`}
                                                                state={getToolState(toolPart)}
                                                            />
                                                            <ToolContent>
                                                                <ToolInput input={toolPart.input || toolPart.args || {}} />
                                                                {toolPart.output !== undefined && (
                                                                    <ToolOutput
                                                                        output={toolPart.output}
                                                                        errorText={toolPart.output?.error}
                                                                    />
                                                                )}
                                                            </ToolContent>
                                                        </Tool>
                                                    );
                                                }

                                                if (part.type === 'text') {
                                                    const schema = tryParseSchema(part.text);
                                                    const data = tryParseExtractedData(part.text);
                                                    
                                                    if (schema && message.role === 'assistant') {
                                                        return (
                                                            <Message key={`${message.id}-${i}`} from={message.role}>
                                                                <MessageContent>
                                                                    <p className="text-sm text-muted-foreground">
                                                                        ✅ 已更新欄位設計 ({schema.fields.length} 個欄位)
                                                                    </p>
                                                                </MessageContent>
                                                            </Message>
                                                        );
                                                    }
                                                    
                                                    if (data && message.role === 'assistant') {
                                                        return (
                                                            <Message key={`${message.id}-${i}`} from={message.role}>
                                                                <MessageContent>
                                                                    <p className="text-sm text-muted-foreground">
                                                                        ✅ 已完成資料擷取
                                                                    </p>
                                                                </MessageContent>
                                                            </Message>
                                                        );
                                                    }
                                                    
                                                    const isStreaming = status === 'streaming' && 
                                                        message.id === messages.at(-1)?.id &&
                                                        i === message.parts.length - 1;
                                                    
                                                    return (
                                                        <Message key={`${message.id}-${i}`} from={message.role}>
                                                            <MessageContent>
                                                                <MessageResponse className={isStreaming ? 'streaming-text smooth-text' : 'smooth-text'}>
                                                                    {part.text}
                                                                </MessageResponse>
                                                            </MessageContent>
                                                        </Message>
                                                    );
                                                }

                                                if (part.type === 'file') return null;
                                                return null;
                                            })}

                                        </div>
                                    ))}
                                    {/* Show shimmer with context-aware message */}
                                    {(() => {
                                        // Show shimmer when:
                                        // 1. Just submitted (waiting for response)
                                        // 2. Streaming with active tool
                                        // 3. Streaming but latest assistant message has no text yet
                                        const lastAssistant = messages.filter(m => m.role === 'assistant').at(-1);
                                        const hasTextContent = lastAssistant?.parts.some(p => p.type === 'text');
                                        const shouldShowShimmer = status === 'submitted' || 
                                            (status === 'streaming' && (activeTool || !hasTextContent));
                                        
                                        return shouldShowShimmer && (
                                            <div className="flex items-center gap-2 text-sm text-muted-foreground p-4">
                                                <Loader size={14} />
                                                <Shimmer duration={1.5}>{getShimmerMessage()}</Shimmer>
                                            </div>
                                        );
                                    })()}
                                </ConversationContent>
                                <ConversationScrollButton />
                            </Conversation>

                            {/* Suggestion Buttons */}
                            {suggestionButtons.length > 0 && !isLoading && (
                                <div className="px-4 pb-2 flex gap-2 overflow-x-auto">
                                    {suggestionButtons.map((btn, i) => (
                                        <button
                                            key={i}
                                            onClick={() => handleSuggestionClick(btn.action)}
                                            className="whitespace-nowrap px-3 py-1.5 text-xs rounded-full bg-primary/10 text-primary hover:bg-primary/20 transition-colors border border-primary/20"
                                        >
                                            {btn.label}
                                        </button>
                                    ))}
                                </div>
                            )}

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
                                                    ? '可以說「擷取資料」或調整欄位...'
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
