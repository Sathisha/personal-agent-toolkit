import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import {
  MessageSquare, 
  Settings, 
  FileText, 
  Cpu, 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Image as ImageIcon, 
  Send, 
  Plus, 
  Trash2, 
  RefreshCw, 
  Check, 
  X, 
  ChevronDown, 
  ChevronUp, 
  Terminal, 
  Play, 
  Database,
  ExternalLink,
  Loader,
  Search,
  WifiOff,
  Square
} from 'lucide-react';
import { AppConfig, Message, McpServer, ImageAttachment, ToolCallState } from './types';

// Speech Recognition configuration
const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;
if (recognition) {
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';
}

const DEFAULT_CONFIG: AppConfig = {
  api_type: 'lmstudio',
  api_url: 'http://host.docker.internal:51234',
  model_name: '',
  api_key: '',
  system_prompt: 'You are my personal agent, an advanced interactive AI coding and productivity assistant. You possess capabilities for planning, step-by-step thinking, document search, and terminal tool usage. Respond clearly and concisely.\n\nBefore answering, decide whether the question depends on real-world information that could have changed or that you cannot know for certain from training alone — this includes weather and forecasts, hotel/restaurant/business listings, prices, availability, travel schedules, current events, sports results, exchange rates, anything tied to a specific date (including future ones), or questions about a specific software/product\'s current capabilities or latest version (e.g. "does X support Y", "what\'s new in the latest version of Z") since those change after your training cutoff. For these, always call search_web first rather than answering from memory or saying the information isn\'t available. If an exact match doesn\'t exist (e.g. a precise forecast too far in the future), search for the closest useful substitute — such as seasonal/climate norms or recent trends — instead of giving up. Only skip searching for stable, general knowledge such as math, definitions, code, or well-established historical facts.\n\nWhen researching something with the web search tool: treat it as a loop, not a single lookup. Search, read what came back, and if it is incomplete or ambiguous, briefly say what is still missing and search again with a refined query — repeat this a few times until you have enough good information, then answer. If a search result begins with \'NO_INTERNET:\', there is no connectivity right now: stop searching immediately, say so plainly, and answer from what you already know instead of retrying.',
  use_tools: true,
  embedding_type: 'lmstudio',
  embedding_url: 'http://host.docker.internal:51234',
  embedding_model: ''
};

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState<'chat' | 'settings' | 'rag' | 'mcp'>('chat');
  
  // App Config
  const [config, setConfig] = useState<AppConfig>(() => {
    const saved = localStorage.getItem('agent_config');
    return saved ? JSON.parse(saved) : DEFAULT_CONFIG;
  });

  // Chat State
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [attachments, setAttachments] = useState<ImageAttachment[]>([]);
  // Use a ref so the WebSocket onmessage closure always reads the latest value (avoids stale closure bug)
  const streamingMessageIdRef = useRef<string | null>(null);
  const [streamingMessageId, setStreamingMessageIdState] = useState<string | null>(null);
  const setStreamingMessageId = (id: string | null) => {
    streamingMessageIdRef.current = id;
    setStreamingMessageIdState(id);
  };
  
  // Voice State
  const [isRecording, setIsRecording] = useState(false);
  const [isVoiceOutputEnabled, setIsVoiceOutputEnabled] = useState(false);
  
  // RAG State
  const [ragStatus, setRagStatus] = useState<{
    documents: string[];
    total_chunks: number;
    indexed_chunks: number;
    unembedded_chunks: number;
  }>({ documents: [], total_chunks: 0, indexed_chunks: 0, unembedded_chunks: 0 });
  const [isUploadingRag, setIsUploadingRag] = useState(false);

  // MCP State
  const [mcpServers, setMcpServers] = useState<McpServer[]>([]);
  const [newMcpServer, setNewMcpServer] = useState({
    name: '',
    command: '',
    args: '',
    env: ''
  });
  const [mcpError, setMcpError] = useState('');
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [modelFetchError, setModelFetchError] = useState('');
  const [isFetchingModels, setIsFetchingModels] = useState(false);

  // WebSockets and References
  const wsRef = useRef<WebSocket | null>(null);
  const mountedRef = useRef(true);       // tracks if component is still mounted
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [wsStatus, setWsStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const ragInputRef = useRef<HTMLInputElement | null>(null);

  // Save config to localstorage
  useEffect(() => {
    localStorage.setItem('agent_config', JSON.stringify(config));
  }, [config]);

  // Connect to WebSockets — guarded against double-mount and race conditions
  const connectWebSocket = () => {
    if (!mountedRef.current) return;  // Don't reconnect after unmount

    // Already open or connecting — do nothing
    if (wsRef.current &&
        (wsRef.current.readyState === WebSocket.OPEN ||
         wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setWsStatus('connecting');
    const backendHost = window.location.hostname;
    const ws = new WebSocket(`ws://${backendHost}:8005/api/chat`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return; }
      setWsStatus('connected');
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;  // Don't reconnect after unmount
      setWsStatus('disconnected');
      // Schedule reconnect, but cancel any previous pending timer first
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(() => {
        if (mountedRef.current) connectWebSocket();
      }, 3000);
    };

    ws.onerror = () => {
      // Errors are followed by onclose — no extra handling needed
      setWsStatus('disconnected');
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleWebSocketEventRef.current(msg);
    };
  };

  useEffect(() => {
    mountedRef.current = true;
    connectWebSocket();
    fetchRagStatus();
    fetchMcpServers();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent onclose from triggering reconnect
        wsRef.current.onerror = null;
        // Only close if the socket is actually open — avoids "closed before established" warning
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.close();
        }
      }
    };
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessageId]);

  const selectedModelOption = modelOptions.includes(config.model_name) ? config.model_name : '__custom__';

  // Fetch model options when settings panel becomes active or endpoint changes
  useEffect(() => {
    if (activeTab !== 'settings') return;
    if (!config.api_url) return;
    fetchModelOptions();
  }, [activeTab, config.api_url, config.api_type, config.api_key]);

  // Voice Speech Synthesis (Text to Speech)
  const speakText = (text: string) => {
    if (!isVoiceOutputEnabled) return;
    // Clean markdown/think tags out of speaking text
    const cleanText = text.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/[\*#_`]/g, '').trim();
    if (!cleanText) return;

    window.speechSynthesis.cancel(); // Stop current speech
    const utterance = new SpeechSynthesisUtterance(cleanText);
    window.speechSynthesis.speak(utterance);
  };

  // Voice Speech Recognition (Speech to Text)
  useEffect(() => {
    if (!recognition) return;

    recognition.onstart = () => {
      setIsRecording(true);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInputText(prev => prev ? prev + ' ' + transcript : transcript);
    };

    recognition.onerror = (e: any) => {
      console.error('Speech recognition error:', e);
      setIsRecording(false);
    };
  }, []);

  const toggleRecording = () => {
    if (!recognition) {
      alert("Speech Recognition not supported in this browser. Please use Chrome/Edge.");
      return;
    }
    if (isRecording) {
      recognition.stop();
    } else {
      recognition.start();
    }
  };

  // REST API: Fetch RAG Status
  const fetchRagStatus = async () => {
    try {
      const backendHost = window.location.hostname;
      const res = await fetch(`http://${backendHost}:8005/api/rag/status`);
      const data = await res.json();
      setRagStatus(data);
    } catch (e) {
      console.error('Error fetching RAG status:', e);
    }
  };

  // REST API: Fetch MCP Servers
  const fetchMcpServers = async () => {
    try {
      const backendHost = window.location.hostname;
      const res = await fetch(`http://${backendHost}:8005/api/mcp/servers`);
      const data = await res.json();
      setMcpServers(data);
    } catch (e) {
      console.error('Error fetching MCP servers:', e);
    }
  };

  const fetchModelOptions = async () => {
    setIsFetchingModels(true);
    setModelFetchError('');
    try {
      const backendHost = window.location.hostname;
      const query = new URLSearchParams({
        api_type: config.api_type,
        api_url: config.api_url,
        api_key: config.api_key || ''
      });
      const res = await fetch(`http://${backendHost}:8005/api/models?${query.toString()}`);
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.detail || data.error || 'Failed to fetch models');
      }
      setModelOptions(Array.isArray(data.models) ? data.models : []);
    } catch (e: any) {
      console.error('Error fetching model list:', e);
      setModelOptions([]);
      setModelFetchError(e?.message || 'Unable to fetch model list.');
    } finally {
      setIsFetchingModels(false);
    }
  };

  // REST API: Add MCP Server
  const handleAddMcpServer = async (e: React.FormEvent) => {
    e.preventDefault();
    setMcpError('');
    if (!newMcpServer.name || !newMcpServer.command) {
      setMcpError('Name and Command are required.');
      return;
    }

    try {
      let envObj = {};
      if (newMcpServer.env) {
        try {
          envObj = JSON.parse(newMcpServer.env);
        } catch {
          setMcpError('Environment variables must be valid JSON object.');
          return;
        }
      }

      const argsArr = newMcpServer.args 
        ? newMcpServer.args.split(',').map(a => a.trim()).filter(Boolean)
        : [];

      const backendHost = window.location.hostname;
      const res = await fetch(`http://${backendHost}:8005/api/mcp/servers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newMcpServer.name,
          command: newMcpServer.command,
          args: argsArr,
          env: envObj
        })
      });
      const data = await res.json();
      if (data.status === 'success' || data.status === 'configured_but_failed_connection') {
        setNewMcpServer({ name: '', command: '', args: '', env: '' });
        fetchMcpServers();
        if (data.status === 'configured_but_failed_connection') {
          setMcpError(data.message);
        }
      } else {
        setMcpError(data.message || 'Failed to add MCP server.');
      }
    } catch (err: any) {
      setMcpError(err.message || 'Connection error.');
    }
  };

  // REST API: Remove MCP Server
  const handleRemoveMcpServer = async (name: string) => {
    try {
      const backendHost = window.location.hostname;
      await fetch(`http://${backendHost}:8005/api/mcp/servers/${name}`, { method: 'DELETE' });
      fetchMcpServers();
    } catch (e) {
      console.error('Error removing MCP server:', e);
    }
  };

  // REST API: RAG File Upload
  const handleRagFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploadingRag(true);
    const backendHost = window.location.hostname;

    for (let i = 0; i < files.length; i++) {
      const formData = new FormData();
      formData.append('file', files[i]);
      formData.append('embedding_type', config.embedding_type);
      formData.append('embedding_url', config.embedding_url);
      formData.append('embedding_model', config.embedding_model);
      formData.append('api_key', config.api_key);

      try {
        const res = await fetch(`http://${backendHost}:8005/api/rag/upload`, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        console.log('Upload success:', data);
      } catch (err) {
        console.error('Upload error:', err);
      }
    }
    
    setIsUploadingRag(false);
    fetchRagStatus();
    if (ragInputRef.current) ragInputRef.current.value = '';
  };

  // REST API: Clear RAG Library
  const handleClearRag = async () => {
    if (!confirm('Are you sure you want to clear all indexed documents?')) return;
    try {
      const backendHost = window.location.hostname;
      await fetch(`http://${backendHost}:8005/api/rag/clear`, { method: 'POST' });
      fetchRagStatus();
    } catch (e) {
      console.error('Error clearing RAG:', e);
    }
  };

  // UI Attachment Handler (Images)
  const addImageFile = (file: File) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = (reader.result as string).split(',')[1];
      setAttachments(prev => [...prev, {
        mime_type: file.type,
        base64: base64String
      }]);
    };
    reader.readAsDataURL(file);
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    for (let i = 0; i < files.length; i++) {
      addImageFile(files[i]);
    }

    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Screenshot / clipboard image paste handler
  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items;
    if (!items || items.length === 0) return;

    const imageItems = Array.from(items).filter(item => item.kind === 'file' && item.type.startsWith('image/'));
    if (imageItems.length === 0) return;

    // Prevent the raw image data (or its filename) from being pasted as text
    e.preventDefault();
    for (const item of imageItems) {
      const file = item.getAsFile();
      if (file) addImageFile(file);
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  // Submit User Message
  const handleSubmitMessage = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (streamingMessageIdRef.current) return; // a response is already in flight — use Stop instead
    if (!inputText.trim() && attachments.length === 0) return;

    // Build the query package
    const userQuery = inputText;
    const queryImages = [...attachments];

    // Clear inputs
    setInputText('');
    setAttachments([]);

    // 1. Add User Message to screen
    const userMsgId = 'user_' + Date.now();
    const newUserMsg: Message = {
      id: userMsgId,
      role: 'user',
      content: userQuery,
      images: queryImages
    };

    // 2. Add empty Assistant message for streaming
    const assistantMsgId = 'assistant_' + Date.now();
    const newAssistantMsg: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      thinking: undefined,   // Only set when model actually emits <think> tokens
      thinkingCollapsed: false,
      toolCalls: []
    };

    // Calculate conversation history in backend expected format.
    // Past assistant tool calls must be followed by their matching 'tool' role
    // messages (with the tool's output) — omitting them, or leaving
    // `arguments` as an object instead of a JSON string, makes LM Studio
    // reject the whole 'messages' array on the next turn.
    const historyPayload: Array<Record<string, any>> = [];
    for (const m of messages) {
      if (m.role === 'assistant' && m.toolCalls && m.toolCalls.length > 0) {
        historyPayload.push({
          role: 'assistant',
          content: m.content || '',
          tool_calls: m.toolCalls.map(tc => ({
            id: tc.id,
            type: 'function',
            function: { name: tc.name, arguments: JSON.stringify(tc.arguments) }
          }))
        });
        for (const tc of m.toolCalls) {
          historyPayload.push({
            role: 'tool',
            tool_call_id: tc.id,
            name: tc.name,
            content: tc.output ?? ''
          });
        }
      } else {
        historyPayload.push({
          role: m.role,
          content: m.content
        });
      }
    }

    setMessages(prev => [...prev, newUserMsg, newAssistantMsg]);
    setStreamingMessageId(assistantMsgId);

    // 3. Send over WebSocket
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      // Reconnect and print error
      connectWebSocket();
      setMessages(prev => prev.slice(0, -2)); // Remove message placeholder
      alert('WebSocket is currently disconnected. Reconnecting, please try again.');
      return;
    }

    const payload = {
      query: userQuery,
      images: queryImages,
      history: historyPayload,
      config: {
        api_type: config.api_type,
        api_url: config.api_url,
        model_name: config.model_name,
        api_key: config.api_key,
        system_prompt: config.system_prompt,
        use_tools: config.use_tools,
        embedding_type: config.embedding_type,
        embedding_url: config.embedding_url,
        embedding_model: config.embedding_model
      }
    };

    wsRef.current.send(JSON.stringify(payload));
  };

  // Stop an in-flight response. The backend has no way to be told "stop" mid-stream
  // (it's blocked awaiting the next chunk from the LLM), so the only real way to abort
  // is to drop the connection — the existing auto-reconnect logic brings it back.
  const handleStopGeneration = () => {
    const currentId = streamingMessageIdRef.current;
    if (!currentId) return;

    setMessages(prev => prev.map(m => {
      if (m.id !== currentId) return m;
      return { ...m, content: (m.content || '') + '\n\n_(Stopped by user.)_' };
    }));
    setStreamingMessageId(null);

    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  const handleGenerateImage = async () => {
    if (!inputText.trim()) return;
    const prompt = inputText.trim();
    setInputText('');

    const userMsgId = 'user_' + Date.now();
    const userMessage: Message = {
      id: userMsgId,
      role: 'user',
      content: prompt
    };

    const assistantMsgId = 'assistant_' + Date.now();
    const assistantPlaceholder: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: 'Generating image...',
      images: [],
      toolCalls: []
    };

    setMessages(prev => [...prev, userMessage, assistantPlaceholder]);

    try {
      const backendHost = window.location.hostname;
      const res = await fetch(`http://${backendHost}:8005/api/images/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          model: config.model_name,
          api_type: config.api_type,
          api_url: config.api_url,
          api_key: config.api_key,
          n: 1,
          size: '1024x1024'
        })
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || data.error || 'Image generation failed');
      }

      const images = (data.images || []).map((img: any) => ({
        mime_type: img.mime_type || 'image/png',
        base64: img.base64,
        url: img.url
      }));

      setMessages(prev => prev.map(m => {
        if (m.id !== assistantMsgId) return m;
        return {
          ...m,
          content: images.length ? 'Here is your generated image.' : 'No image returned.',
          images,
          toolCalls: []
        };
      }));
    } catch (err: any) {
      setMessages(prev => prev.map(m => {
        if (m.id !== assistantMsgId) return m;
        return {
          ...m,
          content: `⚠️ Image generation failed: ${err?.message || err}`,
          toolCalls: []
        };
      }));
    }
  };

  // WebSocket Event Processing — stored in a ref so the ws.onmessage closure is never stale
  const handleWebSocketEventRef = useRef<(event: any) => void>(() => {});
  const handleWebSocketEvent = (event: any) => {
    const { event: eventName, content } = event;
    const currentId = streamingMessageIdRef.current;
    if (!currentId && eventName !== 'error') return; // Guard: no active stream

    setMessages(prev => {
      return prev.map(m => {
        if (m.id !== currentId) return m;

        const msg = { ...m };

        if (eventName === 'thinking_start') {
          msg.thinking = '';
          msg.thinkingCollapsed = false;
        } else if (eventName === 'thinking') {
          msg.thinking = (msg.thinking || '') + content;
        } else if (eventName === 'thinking_end') {
          // Keep it as-is
        } else if (eventName === 'text') {
          msg.content = (msg.content || '') + content;
        } else if (eventName === 'image') {
          msg.images = [...(msg.images || []), content];
        } else if (eventName === 'tool_start') {
          const tc = msg.toolCalls || [];
          msg.toolCalls = [...tc, {
            id: content.id,
            name: content.name,
            arguments: content.arguments,
            status: 'running' as const,
            isCollapsed: false,
            searchRound: content.search_round
          }];
        } else if (eventName === 'tool_end') {
          msg.toolCalls = (msg.toolCalls || []).map(tc => {
            if (tc.id !== content.id) return tc;
            const failed = content.output.startsWith('Error') || content.output.startsWith('NO_INTERNET:');
            return {
              ...tc,
              output: content.output,
              status: (failed ? 'error' : 'done') as 'done' | 'error',
              isCollapsed: true
            };
          });
        } else if (eventName === 'done') {
          setTimeout(() => { speakText(msg.content); }, 100);
          setStreamingMessageId(null);
        } else if (eventName === 'error') {
          msg.content = (msg.content || '') + `\n\n⚠️ Error: ${content}`;
          setStreamingMessageId(null);
        }

        return msg;
      });
    });
  };
  // Keep the ref always pointing to the latest version of the handler
  handleWebSocketEventRef.current = handleWebSocketEvent;

  // Collapsible Toggles
  const toggleThinkingCollapse = (msgId: string) => {
    setMessages(prev => prev.map(m => {
      if (m.id !== msgId) return m;
      return { ...m, thinkingCollapsed: !m.thinkingCollapsed };
    }));
  };

  const toggleToolCallCollapse = (msgId: string, toolId: string) => {
    setMessages(prev => prev.map(m => {
      if (m.id !== msgId) return m;
      return {
        ...m,
        toolCalls: (m.toolCalls || []).map(tc => {
          if (tc.id !== toolId) return tc;
          return { ...tc, isCollapsed: !tc.isCollapsed };
        })
      };
    }));
  };

  // Quick Action triggers
  const triggerModelChoice = (model: string) => {
    setConfig(prev => ({ ...prev, model_name: model }));
  };

  return (
    <div className="app-container">
      {/* 1. Left Nav Bar */}
      <nav className="side-nav">
        <div className="logo-container">
          <MessageSquare size={28} className="logo-icon" />
        </div>
        
        <button 
          onClick={() => setActiveTab('chat')} 
          className={`nav-btn ${activeTab === 'chat' ? 'active' : ''}`}
        >
          <MessageSquare size={22} />
          <span className="tooltip">Chat Space</span>
        </button>

        <button 
          onClick={() => setActiveTab('rag')} 
          className={`nav-btn ${activeTab === 'rag' ? 'active' : ''}`}
        >
          <Database size={22} />
          <span className="tooltip">RAG Library ({ragStatus.documents.length})</span>
        </button>

        <button 
          onClick={() => setActiveTab('mcp')} 
          className={`nav-btn ${activeTab === 'mcp' ? 'active' : ''}`}
        >
          <Cpu size={22} />
          <span className="tooltip">MCP Servers ({mcpServers.length})</span>
        </button>

        <div style={{ flexGrow: 1 }} />

        <button 
          onClick={() => setActiveTab('settings')} 
          className={`nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
        >
          <Settings size={22} />
          <span className="tooltip">Agent Settings</span>
        </button>
      </nav>

      {/* 2. Collapsible Sidebars based on active Tab */}
      
      {/* Settings Panel */}
      <div className={`config-sidebar ${activeTab === 'settings' ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <h2>Agent Settings</h2>
          <button className="nav-btn" onClick={() => setActiveTab('chat')} style={{width: 32, height: 32}}><X size={18} /></button>
        </div>
        <div className="sidebar-content">
          <div className="form-group">
            <label>API Connection Type</label>
            <select 
              value={config.api_type} 
              onChange={e => setConfig(prev => ({ ...prev, api_type: e.target.value as any }))}
              className="form-select"
            >
              <option value="ollama">Ollama (Local)</option>
              <option value="lmstudio">LM Studio (Local)</option>
              <option value="openai">OpenAI Endpoint</option>
            </select>
          </div>

          <div className="form-group">
            <label>Endpoint URL</label>
            <input 
              type="text" 
              value={config.api_url} 
              onChange={e => setConfig(prev => ({ ...prev, api_url: e.target.value }))}
              className="form-input"
              placeholder="http://host.docker.internal:11434"
            />
          </div>

          <div className="form-group">
            <label>Model Name</label>
            {modelOptions.length > 0 ? (
              <select
                value={selectedModelOption}
                onChange={e => {
                  const value = e.target.value;
                  if (value === '__custom__') {
                    setConfig(prev => ({ ...prev, model_name: '' }));
                  } else {
                    setConfig(prev => ({ ...prev, model_name: value }));
                  }
                }}
                className="form-select"
              >
                <option value="" disabled>
                  Choose loaded model or custom
                </option>
                {modelOptions.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
                <option value="__custom__">Use custom model name</option>
              </select>
            ) : null}
            <input 
              type="text" 
              value={config.model_name} 
              onChange={e => setConfig(prev => ({ ...prev, model_name: e.target.value }))}
              className="form-input"
              placeholder="Type or paste a custom model name"
            />
            <div style={{marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center'}}>
              <button type="button" onClick={fetchModelOptions} className="btn-secondary" style={{padding: '2px 6px', fontSize: 11}}>
                {isFetchingModels ? 'Refreshing...' : 'Refresh'}
              </button>
              <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>
                Choose a loaded model, or type any custom model name.
              </span>
            </div>
            {modelFetchError ? (
              <div style={{ marginTop: 6, color: '#fca5a5', fontSize: 12 }}>{modelFetchError}</div>
            ) : null}
          </div>

          <div className="form-group">
            <label>API Key (Optional)</label>
            <input 
              type="password" 
              value={config.api_key} 
              onChange={e => setConfig(prev => ({ ...prev, api_key: e.target.value }))}
              className="form-input"
              placeholder="sk-..."
            />
          </div>

          <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <label htmlFor="use-tools-checkbox">Enable Tool Calling Loop</label>
            <input 
              id="use-tools-checkbox"
              type="checkbox" 
              checked={config.use_tools} 
              onChange={e => setConfig(prev => ({ ...prev, use_tools: e.target.checked }))}
              style={{ width: 18, height: 18, accentColor: 'var(--accent-cyan)' }}
            />
          </div>

          <div style={{ borderTop: '1px solid var(--border-dark)', margin: '20px 0' }} />
          <h3>Embeddings (RAG) Config</h3>
          <div style={{height: 10}} />

          <div className="form-group">
            <label>Embedding Model</label>
            <input 
              type="text" 
              value={config.embedding_model} 
              onChange={e => setConfig(prev => ({ ...prev, embedding_model: e.target.value }))}
              className="form-input"
              placeholder="nomic-embed-text"
            />
          </div>
          
          <div className="form-group">
            <label>System Instructions</label>
            <textarea 
              value={config.system_prompt} 
              onChange={e => setConfig(prev => ({ ...prev, system_prompt: e.target.value }))}
              className="form-textarea"
              rows={5}
            />
          </div>
          
          <button className="btn-primary" style={{width: '100%'}} onClick={() => setActiveTab('chat')}>Save and Close</button>
          <button 
            className="btn-secondary" 
            style={{width: '100%', marginTop: 8}}
            onClick={() => {
              localStorage.removeItem('agent_config');
              setConfig(DEFAULT_CONFIG);
            }}
          >Reset to Defaults (LM Studio)</button>
        </div>
      </div>

      {/* RAG Library Panel */}
      <div className={`config-sidebar ${activeTab === 'rag' ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <h2>RAG Knowledge Base</h2>
          <button className="nav-btn" onClick={() => setActiveTab('chat')} style={{width: 32, height: 32}}><X size={18} /></button>
        </div>
        <div className="sidebar-content">
          <p style={{fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16}}>
            Upload text, markdown, or PDF files to provide the agent with local document context.
          </p>

          <div 
            className="dropzone" 
            onClick={() => ragInputRef.current?.click()}
          >
            <FileText size={32} className="dropzone-icon" />
            <span>Click to upload documents</span>
            <span style={{fontSize: 11, color: 'var(--text-muted)'}}>(PDF, TXT, MD)</span>
            <input 
              type="file" 
              ref={ragInputRef}
              onChange={handleRagFileUpload}
              multiple 
              accept=".txt,.md,.pdf,.json,.py,.js"
              style={{display: 'none'}} 
            />
          </div>

          {isUploadingRag && (
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginTop: 12, fontSize: 13, color: 'var(--accent-cyan)'}}>
              <Loader size={16} className="tool-spinner" />
              <span>Parsing and Indexing document...</span>
            </div>
          )}

          <div style={{marginTop: 24}}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12}}>
              <span style={{fontWeight: 700, fontSize: 14}}>Indexed Documents ({ragStatus.documents.length})</span>
              {ragStatus.documents.length > 0 && (
                <button onClick={handleClearRag} className="doc-remove-btn" title="Clear library">
                  <Trash2 size={15} />
                </button>
              )}
            </div>

            <div className="document-list">
              {ragStatus.documents.length === 0 ? (
                <span style={{fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', display: 'block', padding: 20}}>No documents uploaded yet.</span>
              ) : (
                ragStatus.documents.map((doc, idx) => (
                  <div key={idx} className="document-item">
                    <div className="doc-info">
                      <FileText size={15} className="doc-icon" />
                      <span className="doc-name" title={doc}>{doc}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div style={{marginTop: 20, padding: 12, background: 'rgba(0,0,0,0.2)', borderRadius: 8, fontSize: 12}}>
            <div>Total Chunks: <strong>{ragStatus.total_chunks}</strong></div>
            <div style={{marginTop: 4}}>Embeddings Computed: <strong>{ragStatus.indexed_chunks} / {ragStatus.total_chunks}</strong></div>
          </div>
        </div>
      </div>

      {/* MCP Servers Panel */}
      <div className={`config-sidebar ${activeTab === 'mcp' ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <h2>MCP Servers</h2>
          <button className="nav-btn" onClick={() => setActiveTab('chat')} style={{width: 32, height: 32}}><X size={18} /></button>
        </div>
        <div className="sidebar-content">
          <p style={{fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16}}>
            Connect to Model Context Protocol (MCP) servers running locally over standard input/output (stdio).
          </p>

          <form onSubmit={handleAddMcpServer} style={{display: 'flex', flexDirection: 'column', gap: 12, background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-dark)', padding: 16, borderRadius: 12, marginBottom: 20}}>
            <h3 style={{fontSize: 13, fontWeight: 700}}>Connect Stdio MCP Server</h3>
            
            <div className="form-group" style={{marginBottom: 0}}>
              <label>Server Name</label>
              <input 
                type="text" 
                value={newMcpServer.name} 
                onChange={e => setNewMcpServer(prev => ({ ...prev, name: e.target.value }))}
                className="form-input"
                placeholder="e.g. postgres-db"
              />
            </div>

            <div className="form-group" style={{marginBottom: 0}}>
              <label>Executable Command</label>
              <input 
                type="text" 
                value={newMcpServer.command} 
                onChange={e => setNewMcpServer(prev => ({ ...prev, command: e.target.value }))}
                className="form-input"
                placeholder="npx or python or node"
              />
            </div>

            <div className="form-group" style={{marginBottom: 0}}>
              <label>Arguments (comma-separated)</label>
              <input 
                type="text" 
                value={newMcpServer.args} 
                onChange={e => setNewMcpServer(prev => ({ ...prev, args: e.target.value }))}
                className="form-input"
                placeholder="-y, @modelcontextprotocol/server-postgres"
              />
            </div>

            <div className="form-group" style={{marginBottom: 0}}>
              <label>Environment Variables (JSON)</label>
              <textarea 
                value={newMcpServer.env} 
                onChange={e => setNewMcpServer(prev => ({ ...prev, env: e.target.value }))}
                className="form-input"
                style={{fontFamily: 'var(--font-mono)', fontSize: 11}}
                placeholder='{ "DB_URL": "postgresql://..." }'
                rows={2}
              />
            </div>

            {mcpError && <div style={{color: '#ef4444', fontSize: 12}}>{mcpError}</div>}

            <button type="submit" className="btn-primary" style={{padding: '8px 16px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6}}>
              <Plus size={16} /> Add Server
            </button>
          </form>

          <div>
            <h3 style={{fontSize: 14, fontWeight: 700, marginBottom: 12}}>Active Connections ({mcpServers.length})</h3>
            {mcpServers.length === 0 ? (
              <span style={{fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', display: 'block', padding: 20}}>No MCP servers configured.</span>
            ) : (
              mcpServers.map((server, idx) => (
                <div key={idx} className="mcp-server-card">
                  <div className="mcp-server-card-header">
                    <span className="mcp-server-name">{server.name}</span>
                    <span className={`mcp-connection-status ${server.connected ? '' : 'failed'}`}>
                      <span className={`status-dot ${server.connected ? 'connected' : 'disconnected'}`} style={{width: 6, height: 6}} />
                      {server.connected ? 'Connected' : 'Offline'}
                    </span>
                  </div>
                  <div className="mcp-cmd-text">{server.command} {server.args.join(' ')}</div>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                    <span className="mcp-tools-badge">{server.tools_count} tools imported</span>
                    <button onClick={() => handleRemoveMcpServer(server.name)} className="doc-remove-btn" title="Remove server">
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* 3. Central Chat Space */}
      <div className="chat-container">
        {/* Top bar with model context details */}
        <header className="top-bar">
          <div className="connection-status">
            <span className={`status-dot ${wsStatus}`} />
            <span style={{textTransform: 'capitalize'}}>{wsStatus}</span>
          </div>

          <div style={{display: 'flex', alignItems: 'center', gap: 12}}>
            <div className="model-badge">
              {config.model_name}
            </div>

            <button 
              onClick={() => setIsVoiceOutputEnabled(!isVoiceOutputEnabled)} 
              className="nav-btn" 
              style={{width: 38, height: 38, color: isVoiceOutputEnabled ? 'var(--accent-cyan)' : 'var(--text-muted)'}}
              title={isVoiceOutputEnabled ? 'Mute Assistant Voice' : 'Unmute Assistant Voice'}
            >
              {isVoiceOutputEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            </button>
          </div>
        </header>

        {/* Central Workspace (Messages list or Welcome screen) */}
        <main className="messages-list">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <h1>Personal Agent</h1>
              <p>
                An interactive sandbox powered by local models. It features step-by-step thinking visualization, multi-turn tool loops, local document search, and MCP server extensions.
              </p>
              
              <div className="features-grid">
                <div className="feature-card">
                  <h3><MessageSquare size={16} /> Voice & Vision Enabled</h3>
                  <p>Speak to the agent using Speech-to-Text and upload images to analyze them using multimodal models.</p>
                </div>
                <div className="feature-card">
                  <h3><Cpu size={16} /> Loop Tool Execution</h3>
                  <p>The agent runs terminal scripts, writes files, runs Python code, and searches the web until the task is complete.</p>
                </div>
                <div className="feature-card">
                  <h3><Database size={16} /> Doc Search (RAG)</h3>
                  <p>Index local files and let the agent query them using vector search for accurate context retrieval.</p>
                </div>
                <div className="feature-card">
                  <h3><Terminal size={16} /> Model Context Protocol</h3>
                  <p>Add external MCP servers (Postgres DB, Memory, GitHub, etc.) to inject custom toolboxes.</p>
                </div>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`message-wrapper ${msg.role}`}>
                <span className="message-sender">{msg.role === 'user' ? 'User' : 'Agent'}</span>
                
                {msg.images && msg.images.length > 0 && (
                  <div className="message-images">
                    {msg.images.map((img, index) => (
                      <img 
                        key={index} 
                        src={img.base64 ? `data:${img.mime_type};base64,${img.base64}` : img.url || ''} 
                        className="message-image-preview" 
                        alt={msg.role === 'user' ? 'User Upload' : 'Assistant Image'} 
                      />
                    ))}
                  </div>
                )}

                <div className="message-bubble">
                  {/* Thinking visualization (DeepSeek R1 accordion) */}
                  {msg.role === 'assistant' && msg.thinking !== undefined && (msg.thinking.length > 0 || msg.id === streamingMessageId) && (
                    <div className="thinking-container">
                      <div className="thinking-header" onClick={() => toggleThinkingCollapse(msg.id)}>
                        <div className="thinking-status">
                          {msg.id === streamingMessageId && !msg.thinking.endsWith('</think>') ? (
                            <>
                              <span className="thinking-pulse-dot" />
                              <span>Thinking process...</span>
                            </>
                          ) : (
                            <span>Reasoning Process</span>
                          )}
                        </div>
                        {msg.thinkingCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
                      </div>
                      
                      {!msg.thinkingCollapsed && (
                        <div className="thinking-body">
                          {msg.thinking}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Tool Call Log visualizer — search_web calls get a distinct "search plan" look */}
                  {msg.role === 'assistant' && msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div className="tool-logs-container">
                      {msg.toolCalls.map((tc) => {
                        const isSearch = tc.name === 'search_web';
                        const isNoInternet = !!tc.output?.startsWith('NO_INTERNET:');
                        return (
                        <div key={tc.id} className={`tool-log-item${isSearch ? ' search-step' : ''}`}>
                          <div className="tool-log-header" onClick={() => toggleToolCallCollapse(msg.id, tc.id)}>
                            <div className="tool-status-badge">
                              {tc.status === 'running' ? (
                                <span className="tool-spinner" />
                              ) : isNoInternet ? (
                                <WifiOff size={13} style={{color: '#f59e0b'}} />
                              ) : tc.status === 'error' ? (
                                <span style={{color: '#ef4444'}}>✖</span>
                              ) : (
                                <span style={{color: '#10b981'}}>✔</span>
                              )}
                              {isSearch ? (
                                <span style={{display: 'flex', alignItems: 'center', gap: 4}}>
                                  <Search size={12} />
                                  Search{tc.searchRound ? ` #${tc.searchRound}` : ''}: <strong>{tc.arguments?.query || ''}</strong>
                                </span>
                              ) : (
                                <span>Tool Call: <strong>{tc.name}</strong></span>
                              )}
                            </div>
                            {tc.isCollapsed ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
                          </div>

                          {!tc.isCollapsed && (
                            <div className="tool-log-body">
                              {!isSearch && (
                                <>
                                  <div>Arguments:</div>
                                  <pre><code>{JSON.stringify(tc.arguments, null, 2)}</code></pre>
                                </>
                              )}
                              {tc.output && (
                                <div style={{marginTop: isSearch ? 0 : 8}}>
                                  <div>{isNoInternet ? 'Status:' : 'Output:'}</div>
                                  <pre><code>{tc.output}</code></pre>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Main text content */}
                  <div className="markdown-content">
                    {msg.content ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                          a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    ) : (
                      msg.id === streamingMessageId ? <Loader size={16} className="tool-spinner" /> : ''
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </main>

        {/* Input panel at bottom */}
        <footer className="input-panel">
          <form onSubmit={handleSubmitMessage} className="input-container-wrapper">
            {/* Attachment pre-previews */}
            {attachments.length > 0 && (
              <div className="attachment-previews">
                {attachments.map((att, idx) => (
                  <div key={idx} className="attachment-thumbnail">
                    <img src={`data:${att.mime_type};base64,${att.base64}`} alt="Thumbnail" />
                    <button 
                      type="button" 
                      onClick={() => removeAttachment(idx)} 
                      className="remove-attachment-btn"
                    >
                      <X size={10} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="input-row">
              <button 
                type="button" 
                onClick={() => fileInputRef.current?.click()} 
                className="input-action-btn"
                title="Attach Image"
              >
                <ImageIcon size={20} />
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleImageSelect} 
                  accept="image/*" 
                  multiple 
                  style={{display: 'none'}} 
                />
              </button>

              <button 
                type="button" 
                onClick={handleGenerateImage} 
                className="input-action-btn"
                title="Generate image from prompt"
                disabled={!inputText.trim()}
              >
                <Plus size={16} />
              </button>

              <button 
                type="button" 
                onClick={toggleRecording} 
                className={`input-action-btn ${isRecording ? 'active-record' : ''}`}
                title={isRecording ? 'Stop Voice Recording' : 'Voice Input (Speech to Text)'}
              >
                {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
              </button>

              {isRecording ? (
                <div className="soundwave-container" style={{flex: 1}}>
                  <div className="soundwave-bar" />
                  <div className="soundwave-bar" />
                  <div className="soundwave-bar" />
                  <div className="soundwave-bar" />
                  <div className="soundwave-bar" />
                  <span style={{fontSize: 12, color: 'var(--accent-pink)', marginLeft: 8}}>Listening...</span>
                </div>
              ) : (
                <textarea
                  value={inputText}
                  onChange={e => setInputText(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmitMessage();
                    }
                  }}
                  onPaste={handlePaste}
                  className="chat-input"
                  placeholder={wsStatus === 'connected' ? "Type a prompt, paste a screenshot, or attach an image..." : "Disconnected from agent..."}
                  disabled={wsStatus !== 'connected'}
                />
              )}

              {streamingMessageId ? (
                <button
                  type="button"
                  onClick={handleStopGeneration}
                  className="send-btn stop-btn"
                  title="Stop generating"
                >
                  <Square size={14} fill="currentColor" />
                </button>
              ) : (
                <button
                  type="submit"
                  className="send-btn"
                  disabled={wsStatus !== 'connected' || (!inputText.trim() && attachments.length === 0)}
                >
                  <Send size={16} />
                </button>
              )}
            </div>
          </form>
          
          <div style={{fontSize: 11, color: 'var(--text-muted)'}}>
            Personal Agent — Local Chat Agent Sandbox. Make changes, run code, iterate freely.
          </div>
        </footer>
      </div>
    </div>
  );
}
