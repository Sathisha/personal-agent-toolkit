export interface ImageAttachment {
  mime_type: string;
  base64?: string;
  url?: string;
}

export interface ToolCallState {
  id: string;
  name: string;
  arguments: any;
  output?: string;
  status: 'running' | 'done' | 'error';
  isCollapsed?: boolean;
  // Set only for search_web calls — which attempt number this is within the
  // current turn's research loop, so the UI can show it as "Search #N".
  searchRound?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  images?: ImageAttachment[];
  thinking?: string;
  thinkingCollapsed?: boolean;
  toolCalls?: ToolCallState[];
}

export interface AppConfig {
  api_type: 'ollama' | 'lmstudio' | 'openai';
  api_url: string;
  model_name: string;
  api_key: string;
  system_prompt: string;
  use_tools: boolean;
  embedding_type: 'ollama' | 'openai' | 'lmstudio';
  embedding_url: string;
  embedding_model: string;
}

export interface McpServer {
  name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
  connected: boolean;
  tools_count: number;
}
