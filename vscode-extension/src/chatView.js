/**
 * AI Orchestrator Chat View
 * =========================
 * A simple chat UI rendered in a VS Code WebviewView (sidebar).
 *
 * This is intentionally dependency-free to keep the extension lightweight.
 */

const crypto = require('crypto');

class ChatViewProvider {
    static viewType = 'ai-orchestrator.chatView';

    /**
     * @param {import('vscode').ExtensionContext} context
     * @param {{ query: Function, clearHistory: Function, conversationHistory: Array<{role: string, content: any}> }} orchestrator
     * @param {(overrides?: object) => object} getConfiguredQueryOptions
     */
    constructor(context, orchestrator, getConfiguredQueryOptions) {
        this._context = context;
        this._orchestrator = orchestrator;
        this._getConfiguredQueryOptions = getConfiguredQueryOptions;
        this._view = null;
        this._busy = false;
    }

    refresh() {
        this._postHistory();
    }

    /**
     * @param {import('vscode').WebviewView} webviewView
     */
    resolveWebviewView(webviewView) {
        this._view = webviewView;

        const { webview } = webviewView;
        webview.options = {
            enableScripts: true,
            localResourceRoots: [this._context.extensionUri],
        };

        webview.html = this._getHtml(webview);

        webview.onDidReceiveMessage(async (message) => {
            const type = message?.type;
            if (type === 'ready') {
                this._postHistory();
                return;
            }

            if (type === 'clear') {
                this._orchestrator.clearHistory();
                this._postHistory();
                return;
            }

            if (type === 'userMessage') {
                const text = typeof message?.text === 'string' ? message.text.trim() : '';
                if (!text) return;

                if (this._busy) {
                    this._postStatus('Busy: wait for the current response to finish.');
                    return;
                }

                this._busy = true;
                this._postStatus('Thinking...');

                try {
                    const response = await this._orchestrator.query(
                        text,
                        this._getConfiguredQueryOptions()
                    );

                    this._postMessage({
                        type: 'assistantMessage',
                        content: response?.content || '',
                        meta: {
                            model: response?.model || '',
                            latencyMs: response?.latencyMs || 0,
                            inputTokens: response?.usage?.inputTokens || 0,
                            outputTokens: response?.usage?.outputTokens || 0,
                        },
                    });
                    this._postStatus('');
                } catch (error) {
                    const msg = String(error?.message || error || 'Unknown error');
                    this._postMessage({
                        type: 'errorMessage',
                        content: msg,
                    });
                    this._postStatus('');
                } finally {
                    this._busy = false;
                }
            }
        });
    }

    _postHistory() {
        const history = Array.isArray(this._orchestrator?.conversationHistory)
            ? this._orchestrator.conversationHistory
            : [];

        const sanitized = history
            .filter((msg) => msg && typeof msg === 'object')
            .map((msg) => ({
                role: String(msg.role || ''),
                content: typeof msg.content === 'string' ? msg.content : String(msg.content ?? ''),
            }))
            .filter((msg) => msg.role && msg.content !== undefined);

        this._postMessage({
            type: 'setHistory',
            history: sanitized,
        });
    }

    _postStatus(status) {
        this._postMessage({
            type: 'status',
            status: typeof status === 'string' ? status : '',
        });
    }

    _postMessage(message) {
        if (!this._view) return;
        this._view.webview.postMessage(message);
    }

    _getHtml(webview) {
        const nonce = crypto.randomBytes(16).toString('hex');
        const csp = [
            `default-src 'none';`,
            `img-src ${webview.cspSource} https: data:;`,
            `style-src ${webview.cspSource} 'unsafe-inline';`,
            `script-src 'nonce-${nonce}';`,
        ].join(' ');

        // Keep the UI simple but readable. Avoid external resources for security.
        return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Orchestrator Chat</title>
  <style>
    :root {
      --bg: var(--vscode-sideBar-background);
      --fg: var(--vscode-sideBar-foreground);
      --muted: var(--vscode-descriptionForeground);
      --border: var(--vscode-panel-border);
      --inputBg: var(--vscode-input-background);
      --inputFg: var(--vscode-input-foreground);
      --buttonBg: var(--vscode-button-background);
      --buttonFg: var(--vscode-button-foreground);
      --buttonBgHover: var(--vscode-button-hoverBackground);
      --errorFg: var(--vscode-errorForeground);
      --shadow: rgba(0, 0, 0, 0.15);
    }

    html, body {
      height: 100%;
    }

    body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: var(--vscode-font-family);
      font-size: 13px;
      line-height: 1.4;
      display: flex;
      flex-direction: column;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 10px;
      border-bottom: 1px solid var(--border);
      gap: 8px;
    }

    .title {
      font-weight: 600;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      user-select: none;
    }

    .actions {
      display: flex;
      gap: 6px;
    }

    button {
      border: 1px solid transparent;
      background: var(--buttonBg);
      color: var(--buttonFg);
      padding: 5px 9px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      box-shadow: 0 1px 0 var(--shadow);
    }

    button:hover {
      background: var(--buttonBgHover);
    }

    button.secondary {
      background: transparent;
      color: var(--fg);
      border-color: var(--border);
      box-shadow: none;
    }

    button.secondary:hover {
      background: color-mix(in srgb, var(--inputBg) 80%, transparent);
    }

    .messages {
      flex: 1;
      overflow: auto;
      padding: 12px 10px 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .msg {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 10px;
      background: color-mix(in srgb, var(--inputBg) 60%, transparent);
    }

    .msg.user {
      align-self: flex-end;
      background: color-mix(in srgb, var(--buttonBg) 12%, var(--inputBg));
    }

    .role {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      user-select: none;
    }

    .content {
      white-space: pre-wrap;
      word-break: break-word;
    }

    .code {
      margin-top: 8px;
      border-radius: 8px;
      padding: 8px;
      background: color-mix(in srgb, #000 12%, var(--inputBg));
      border: 1px solid var(--border);
      overflow-x: auto;
      font-family: var(--vscode-editor-font-family);
      font-size: 12px;
      white-space: pre;
    }

    .meta {
      margin-top: 8px;
      font-size: 11px;
      color: var(--muted);
      user-select: none;
    }

    .status {
      min-height: 18px;
      padding: 0 10px 8px 10px;
      color: var(--muted);
      font-size: 12px;
      user-select: none;
    }

    .composer {
      border-top: 1px solid var(--border);
      padding: 10px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: end;
    }

    textarea {
      width: 100%;
      resize: vertical;
      min-height: 44px;
      max-height: 220px;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--inputBg);
      color: var(--inputFg);
      font-family: var(--vscode-font-family);
      font-size: 13px;
      line-height: 1.4;
      outline: none;
    }

    textarea:focus {
      border-color: color-mix(in srgb, var(--buttonBg) 35%, var(--border));
    }

    .hint {
      padding: 0 10px 10px 10px;
      font-size: 11px;
      color: var(--muted);
      user-select: none;
    }

    .error {
      color: var(--errorFg);
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="title">AI Orchestrator</div>
    <div class="actions">
      <button id="clear" class="secondary" title="Clear conversation">Clear</button>
    </div>
  </div>

  <div id="messages" class="messages" role="log" aria-live="polite"></div>
  <div id="status" class="status"></div>

  <div class="composer">
    <textarea id="input" placeholder="Ask AI Orchestrator..." aria-label="Prompt"></textarea>
    <button id="send">Send</button>
  </div>
  <div class="hint">Enter to send. Shift+Enter for a newline.</div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();

    const messagesEl = document.getElementById('messages');
    const statusEl = document.getElementById('status');
    const inputEl = document.getElementById('input');
    const sendEl = document.getElementById('send');
    const clearEl = document.getElementById('clear');

    function splitCodeFences(text) {
      // Very small markdown-ish renderer: split by triple-backtick fences.
      const fence = String.fromCharCode(96).repeat(3);
      const parts = [];
      let i = 0;
      while (i < text.length) {
        const start = text.indexOf(fence, i);
        if (start === -1) {
          parts.push({ type: 'text', value: text.slice(i) });
          break;
        }
        if (start > i) {
          parts.push({ type: 'text', value: text.slice(i, start) });
        }
        const end = text.indexOf(fence, start + fence.length);
        if (end === -1) {
          parts.push({ type: 'text', value: text.slice(start) });
          break;
        }
        const codeBlock = text.slice(start + fence.length, end);
        parts.push({ type: 'code', value: codeBlock.replace(/^\\n/, '') });
        i = end + fence.length;
      }
      return parts;
    }

    function scrollToBottom() {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderMessage(role, content, meta) {
      const msgEl = document.createElement('div');
      msgEl.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');

      const roleEl = document.createElement('div');
      roleEl.className = 'role';
      roleEl.textContent = role === 'user' ? 'You' : (role === 'assistant' ? 'Assistant' : role);
      msgEl.appendChild(roleEl);

      const parts = splitCodeFences(String(content || ''));
      for (const part of parts) {
        if (part.type === 'code') {
          const pre = document.createElement('pre');
          pre.className = 'code';
          pre.textContent = part.value;
          msgEl.appendChild(pre);
          continue;
        }
        if (part.value) {
          const contentEl = document.createElement('div');
          contentEl.className = 'content';
          contentEl.textContent = part.value;
          msgEl.appendChild(contentEl);
        }
      }

      if (meta && meta.model) {
        const metaEl = document.createElement('div');
        metaEl.className = 'meta';
        const tokens = (meta.inputTokens || 0) + '/' + (meta.outputTokens || 0);
        const latency = meta.latencyMs ? (meta.latencyMs + 'ms') : '';
        metaEl.textContent = meta.model + ' | ' + tokens + ' tokens | ' + latency;
        msgEl.appendChild(metaEl);
      }

      messagesEl.appendChild(msgEl);
      scrollToBottom();
    }

    function renderHistory(history) {
      messagesEl.textContent = '';
      for (const msg of history || []) {
        renderMessage(msg.role, msg.content);
      }
    }

    function setStatus(text) {
      statusEl.textContent = String(text || '');
    }

    function addError(text) {
      const msgEl = document.createElement('div');
      msgEl.className = 'msg assistant';
      const roleEl = document.createElement('div');
      roleEl.className = 'role';
      roleEl.textContent = 'Error';
      msgEl.appendChild(roleEl);
      const errEl = document.createElement('div');
      errEl.className = 'error';
      errEl.textContent = String(text || 'Unknown error');
      msgEl.appendChild(errEl);
      messagesEl.appendChild(msgEl);
      scrollToBottom();
    }

    function sendPrompt() {
      const text = (inputEl.value || '').trim();
      if (!text) return;

      renderMessage('user', text);
      inputEl.value = '';
      inputEl.focus();
      setStatus('Sending...');

      vscode.postMessage({ type: 'userMessage', text });
    }

    sendEl.addEventListener('click', sendPrompt);
    clearEl.addEventListener('click', () => {
      vscode.postMessage({ type: 'clear' });
    });

    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendPrompt();
      }
    });

    window.addEventListener('message', (event) => {
      const msg = event.data || {};
      if (msg.type === 'setHistory') {
        renderHistory(msg.history || []);
        setStatus('');
        return;
      }
      if (msg.type === 'assistantMessage') {
        setStatus('');
        renderMessage('assistant', msg.content || '', msg.meta || null);
        return;
      }
      if (msg.type === 'errorMessage') {
        setStatus('');
        addError(msg.content || '');
        return;
      }
      if (msg.type === 'status') {
        setStatus(msg.status || '');
        return;
      }
    });

    vscode.postMessage({ type: 'ready' });
    inputEl.focus();
  </script>
</body>
</html>`;
    }
}

module.exports = { ChatViewProvider };
