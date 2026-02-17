/**
 * AI Orchestrator Chat View
 * =========================
 * Sidebar chat experience with:
 * - Progressive (streaming-style) rendering
 * - Context attachments (selection/file)
 * - Apply edits and diff preview actions
 */

const crypto = require('crypto');
const path = require('path');
const vscode = require('vscode');

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
        this._attachments = [];
        this._assistantMessages = new Map();
        this._idCounter = 0;
        this._pendingDraft = '';
    }

    refresh() {
        this._postHistory();
        this._postAttachments();
    }

    setDraft(text) {
        const draft = typeof text === 'string' ? text : '';
        this._pendingDraft = draft;
        if (this._view) {
            this._postMessage({ type: 'setDraft', text: draft });
        }
    }

    async seedSelectionDraft(defaultPrompt = 'Please help with this selected code.') {
        const attachment = this._createSelectionAttachment();
        if (!attachment) {
            return false;
        }
        this._attachments = [attachment];
        this._postAttachments();
        this.setDraft(defaultPrompt);
        return true;
    }

    _nextId(prefix) {
        this._idCounter += 1;
        return `${prefix}-${Date.now()}-${this._idCounter}`;
    }

    _truncate(text, maxChars = 18000) {
        if (typeof text !== 'string') {
            return '';
        }
        if (text.length <= maxChars) {
            return text;
        }
        const omitted = text.length - maxChars;
        return `${text.slice(0, maxChars)}\n\n[...truncated ${omitted} chars...]`;
    }

    _createSelectionAttachment() {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.selection.isEmpty) {
            return null;
        }

        const selected = editor.document.getText(editor.selection);
        if (!selected.trim()) {
            return null;
        }

        const relativePath = vscode.workspace.asRelativePath(editor.document.uri, false);
        const startLine = editor.selection.start.line + 1;
        const endLine = editor.selection.end.line + 1;
        const fileLabel = relativePath || path.basename(editor.document.uri.fsPath);

        return {
            id: this._nextId('att'),
            kind: 'selection',
            label: `${fileLabel}:${startLine}-${endLine}`,
            languageId: editor.document.languageId || '',
            content: this._truncate(selected),
        };
    }

    _createFileAttachment() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return null;
        }

        const content = editor.document.getText();
        if (!content.trim()) {
            return null;
        }

        const relativePath = vscode.workspace.asRelativePath(editor.document.uri, false);
        const fileLabel = relativePath || path.basename(editor.document.uri.fsPath);

        return {
            id: this._nextId('att'),
            kind: 'file',
            label: fileLabel,
            languageId: editor.document.languageId || '',
            content: this._truncate(content),
        };
    }

    _composePromptWithAttachments(prompt, attachments) {
        if (!Array.isArray(attachments) || attachments.length === 0) {
            return prompt;
        }

        const blockSections = attachments.map((attachment, index) => {
            const lang = attachment.languageId || '';
            const heading = `Attachment ${index + 1} (${attachment.kind}): ${attachment.label}`;
            return `${heading}\n${String.fromCharCode(96).repeat(3)}${lang}\n${attachment.content}\n${String.fromCharCode(96).repeat(3)}`;
        });

        return [
            prompt,
            '',
            'Use the following attached context when answering. If attachment context conflicts with assumptions, trust the attachments.',
            '',
            ...blockSections,
        ].join('\n');
    }

    _extractCodeBlocks(text) {
        const blocks = [];
        if (typeof text !== 'string' || !text) {
            return blocks;
        }

        const pattern = /```([^\n`]*)\n?([\s\S]*?)```/g;
        let match;
        while ((match = pattern.exec(text)) !== null) {
            blocks.push({
                language: (match[1] || '').trim().toLowerCase(),
                code: (match[2] || '').replace(/^\n/, '').trimEnd(),
            });
        }

        return blocks;
    }

    _extractCodeForEditorAction(text) {
        const blocks = this._extractCodeBlocks(text);
        if (blocks.length === 0) {
            return typeof text === 'string' ? text.trim() : '';
        }

        const nonDiff = blocks.find((block) => block.language !== 'diff' && block.language !== 'patch');
        if (nonDiff && nonDiff.code) {
            return nonDiff.code;
        }

        const diffBlock = blocks[0];
        if (diffBlock && (diffBlock.language === 'diff' || diffBlock.language === 'patch')) {
            const plusLines = diffBlock.code
                .split('\n')
                .filter((line) => line.startsWith('+') && !line.startsWith('+++'))
                .map((line) => line.slice(1));
            if (plusLines.length > 0) {
                return plusLines.join('\n');
            }
        }

        return blocks[0].code || '';
    }

    async _applyResponseToEditor(messageId, mode = 'replace') {
        const content = this._assistantMessages.get(messageId);
        if (!content) {
            throw new Error('Could not find that assistant response.');
        }

        const code = this._extractCodeForEditorAction(content);
        if (!code) {
            throw new Error('No code block or editable content found in the response.');
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            throw new Error('No active editor to apply changes to.');
        }

        const selection = editor.selection;
        const shouldReplace = mode === 'replace' && !selection.isEmpty;

        await editor.edit((editBuilder) => {
            if (shouldReplace) {
                editBuilder.replace(selection, code);
                return;
            }
            editBuilder.insert(selection.active, code);
        });
    }

    async _previewResponseDiff(messageId, mode = 'replace') {
        const content = this._assistantMessages.get(messageId);
        if (!content) {
            throw new Error('Could not find that assistant response.');
        }

        const code = this._extractCodeForEditorAction(content);
        if (!code) {
            throw new Error('No code block or editable content found in the response.');
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            throw new Error('No active editor to preview a diff against.');
        }

        const originalDoc = editor.document;
        const originalText = originalDoc.getText();
        const selection = editor.selection;

        const startOffset = originalDoc.offsetAt(selection.start);
        const endOffset = mode === 'replace' && !selection.isEmpty
            ? originalDoc.offsetAt(selection.end)
            : startOffset;

        const proposedText = `${originalText.slice(0, startOffset)}${code}${originalText.slice(endOffset)}`;
        const proposedDoc = await vscode.workspace.openTextDocument({
            language: originalDoc.languageId,
            content: proposedText,
        });

        await vscode.commands.executeCommand(
            'vscode.diff',
            originalDoc.uri,
            proposedDoc.uri,
            'AI Orchestrator Diff Preview'
        );
    }

    async _copyResponse(messageId) {
        const content = this._assistantMessages.get(messageId);
        if (!content) {
            throw new Error('Could not find that assistant response.');
        }
        await vscode.env.clipboard.writeText(content);
    }

    _postHistory() {
        const history = Array.isArray(this._orchestrator?.conversationHistory)
            ? this._orchestrator.conversationHistory
            : [];

        this._assistantMessages.clear();

        const sanitized = history
            .filter((msg) => msg && typeof msg === 'object')
            .map((msg) => {
                const role = String(msg.role || '');
                const content = typeof msg.content === 'string' ? msg.content : String(msg.content ?? '');
                if (role === 'assistant') {
                    const messageId = this._nextId('assistant');
                    this._assistantMessages.set(messageId, content);
                    return { role, content, messageId };
                }
                return { role, content };
            })
            .filter((msg) => msg.role && msg.content !== undefined);

        this._postMessage({
            type: 'setHistory',
            history: sanitized,
        });
    }

    _postAttachments() {
        this._postMessage({
            type: 'attachments',
            attachments: this._attachments.map((attachment) => ({
                id: attachment.id,
                kind: attachment.kind,
                label: attachment.label,
            })),
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
                this._postAttachments();
                if (this._pendingDraft) {
                    this._postMessage({ type: 'setDraft', text: this._pendingDraft });
                }
                return;
            }

            if (type === 'clear') {
                this._orchestrator.clearHistory();
                this._assistantMessages.clear();
                this._postHistory();
                return;
            }

            if (type === 'attachSelection') {
                const attachment = this._createSelectionAttachment();
                if (!attachment) {
                    this._postStatus('No active editor selection to attach.');
                    return;
                }
                this._attachments.push(attachment);
                this._postAttachments();
                this._postStatus(`Attached selection: ${attachment.label}`);
                return;
            }

            if (type === 'attachFile') {
                const attachment = this._createFileAttachment();
                if (!attachment) {
                    this._postStatus('No active file to attach.');
                    return;
                }
                this._attachments.push(attachment);
                this._postAttachments();
                this._postStatus(`Attached file: ${attachment.label}`);
                return;
            }

            if (type === 'removeAttachment') {
                const id = String(message?.id || '');
                this._attachments = this._attachments.filter((attachment) => attachment.id !== id);
                this._postAttachments();
                return;
            }

            if (type === 'clearAttachments') {
                this._attachments = [];
                this._postAttachments();
                this._postStatus('Cleared attachments.');
                return;
            }

            if (type === 'applyResponse') {
                try {
                    await this._applyResponseToEditor(String(message?.messageId || ''), String(message?.mode || 'replace'));
                    this._postStatus('Applied AI edit to the active editor.');
                } catch (error) {
                    this._postStatus(String(error?.message || error || 'Failed to apply edit.'));
                }
                return;
            }

            if (type === 'previewResponseDiff') {
                try {
                    await this._previewResponseDiff(String(message?.messageId || ''), String(message?.mode || 'replace'));
                    this._postStatus('Opened diff preview.');
                } catch (error) {
                    this._postStatus(String(error?.message || error || 'Failed to preview diff.'));
                }
                return;
            }

            if (type === 'copyResponse') {
                try {
                    await this._copyResponse(String(message?.messageId || ''));
                    this._postStatus('Copied response to clipboard.');
                } catch (error) {
                    this._postStatus(String(error?.message || error || 'Failed to copy response.'));
                }
                return;
            }

            if (type === 'userMessage') {
                const text = typeof message?.text === 'string' ? message.text.trim() : '';
                if (!text) return;

                if (this._busy) {
                    this._postStatus('Busy: wait for the current response to finish.');
                    return;
                }

                const attachmentIds = Array.isArray(message?.attachmentIds)
                    ? message.attachmentIds.map((id) => String(id))
                    : [];
                const selectedAttachments = attachmentIds.length > 0
                    ? this._attachments.filter((attachment) => attachmentIds.includes(attachment.id))
                    : [...this._attachments];

                const promptToSend = this._composePromptWithAttachments(text, selectedAttachments);
                const messageId = this._nextId('assistant');
                this._assistantMessages.set(messageId, '');

                this._busy = true;
                this._postStatus('Thinking...');
                this._postMessage({ type: 'assistantStreamStart', messageId });

                let sawChunks = false;

                try {
                    const response = await this._orchestrator.query(
                        promptToSend,
                        this._getConfiguredQueryOptions({
                            stream: true,
                            onStreamChunk: (chunk) => {
                                if (!chunk) return;
                                sawChunks = true;
                                const previous = this._assistantMessages.get(messageId) || '';
                                const next = previous + chunk;
                                this._assistantMessages.set(messageId, next);
                                this._postMessage({
                                    type: 'assistantStreamDelta',
                                    messageId,
                                    chunk,
                                });
                            },
                        })
                    );

                    const finalContent = sawChunks
                        ? (this._assistantMessages.get(messageId) || response?.content || '')
                        : (response?.content || '');
                    this._assistantMessages.set(messageId, finalContent);

                    this._postMessage({
                        type: 'assistantStreamDone',
                        messageId,
                        content: finalContent,
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

    _getHtml(webview) {
        const nonce = crypto.randomBytes(16).toString('hex');
        const csp = [
            `default-src 'none';`,
            `img-src ${webview.cspSource} https: data:;`,
            `style-src ${webview.cspSource} 'unsafe-inline';`,
            `script-src 'nonce-${nonce}';`,
        ].join(' ');

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
      --chipBg: color-mix(in srgb, var(--inputBg) 75%, transparent);
      --shadow: rgba(0, 0, 0, 0.15);
    }

    html, body { height: 100%; }

    body {
      margin: 0;
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
      padding: 10px;
      border-bottom: 1px solid var(--border);
      gap: 8px;
    }

    .title {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      user-select: none;
    }

    .actions { display: flex; gap: 6px; }

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

    button:hover { background: var(--buttonBgHover); }

    button.secondary {
      background: transparent;
      color: var(--fg);
      border-color: var(--border);
      box-shadow: none;
    }

    button.secondary:hover {
      background: color-mix(in srgb, var(--inputBg) 80%, transparent);
    }

    .attachbar {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
    }

    .attachments {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 0 10px 8px 10px;
      border-bottom: 1px solid var(--border);
    }

    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--chipBg);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      color: var(--muted);
    }

    .chip button {
      border: none;
      background: transparent;
      color: inherit;
      font-size: 12px;
      padding: 0;
      box-shadow: none;
      cursor: pointer;
      line-height: 1;
    }

    .messages {
      flex: 1;
      overflow: auto;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .msg {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 10px;
      background: color-mix(in srgb, var(--inputBg) 60%, transparent);
      align-self: flex-start;
      max-width: 100%;
    }

    .msg.user {
      align-self: flex-end;
      background: color-mix(in srgb, var(--buttonBg) 14%, var(--inputBg));
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

    .msg-actions {
      margin-top: 8px;
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }

    .msg-actions button {
      padding: 4px 8px;
      font-size: 11px;
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

  <div class="attachbar">
    <button id="attachSelection" class="secondary" title="Attach selected editor text">Attach Selection</button>
    <button id="attachFile" class="secondary" title="Attach active file">Attach File</button>
    <button id="clearAttachments" class="secondary" title="Clear attachments">Clear Attachments</button>
  </div>
  <div id="attachments" class="attachments"></div>

  <div id="messages" class="messages" role="log" aria-live="polite"></div>
  <div id="status" class="status"></div>

  <div class="composer">
    <textarea id="input" placeholder="Ask AI Orchestrator..." aria-label="Prompt"></textarea>
    <button id="send">Send</button>
  </div>
  <div class="hint">Enter to send. Shift+Enter for newline. Inline actions do not replace VS Code's default AI features.</div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();

    const messagesEl = document.getElementById('messages');
    const statusEl = document.getElementById('status');
    const inputEl = document.getElementById('input');
    const sendEl = document.getElementById('send');
    const clearEl = document.getElementById('clear');
    const attachSelectionEl = document.getElementById('attachSelection');
    const attachFileEl = document.getElementById('attachFile');
    const clearAttachmentsEl = document.getElementById('clearAttachments');
    const attachmentsEl = document.getElementById('attachments');

    const assistantNodes = new Map();
    let attachments = [];

    function splitCodeFences(text) {
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
        parts.push({ type: 'code', value: codeBlock.replace(/^\n/, '') });
        i = end + fence.length;
      }
      return parts;
    }

    function scrollToBottom() {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function setStatus(text) {
      statusEl.textContent = String(text || '');
    }

    function buildMessageContainer(role) {
      const msgEl = document.createElement('div');
      msgEl.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');

      const roleEl = document.createElement('div');
      roleEl.className = 'role';
      roleEl.textContent = role === 'user' ? 'You' : 'Assistant';
      msgEl.appendChild(roleEl);

      const bodyEl = document.createElement('div');
      bodyEl.className = 'content';
      msgEl.appendChild(bodyEl);

      const metaEl = document.createElement('div');
      metaEl.className = 'meta';
      msgEl.appendChild(metaEl);

      return { msgEl, bodyEl, metaEl };
    }

    function renderMessage(role, content, meta, messageId) {
      const { msgEl, bodyEl, metaEl } = buildMessageContainer(role);

      const parts = splitCodeFences(String(content || ''));
      bodyEl.textContent = '';
      for (const part of parts) {
        if (part.type === 'code') {
          const pre = document.createElement('pre');
          pre.className = 'code';
          pre.textContent = part.value;
          msgEl.appendChild(pre);
          continue;
        }
        if (part.value) {
          const chunk = document.createElement('div');
          chunk.className = 'content';
          chunk.textContent = part.value;
          msgEl.appendChild(chunk);
        }
      }

      if (meta && meta.model) {
        const tokens = (meta.inputTokens || 0) + '/' + (meta.outputTokens || 0);
        const latency = meta.latencyMs ? (meta.latencyMs + 'ms') : '';
        metaEl.textContent = meta.model + ' | ' + tokens + ' tokens | ' + latency;
      } else {
        metaEl.textContent = '';
      }

      if (role === 'assistant' && messageId) {
        msgEl.dataset.messageId = messageId;
        const actionsEl = document.createElement('div');
        actionsEl.className = 'msg-actions';
        actionsEl.innerHTML = [
          '<button data-action="apply" title="Replace current selection (or insert at cursor)">Apply</button>',
          '<button data-action="diff" title="Open side-by-side diff preview">Diff</button>',
          '<button data-action="copy" title="Copy full response">Copy</button>',
        ].join('');
        msgEl.appendChild(actionsEl);
      }

      messagesEl.appendChild(msgEl);
      scrollToBottom();
      return msgEl;
    }

    function renderHistory(history) {
      assistantNodes.clear();
      messagesEl.textContent = '';
      for (const msg of history || []) {
        const messageId = msg.messageId || '';
        const node = renderMessage(msg.role, msg.content, null, messageId);
        if (msg.role === 'assistant' && messageId) {
          assistantNodes.set(messageId, { node, content: String(msg.content || '') });
        }
      }
    }

    function ensureAssistantNode(messageId) {
      const existing = assistantNodes.get(messageId);
      if (existing) {
        return existing;
      }
      const { msgEl, bodyEl, metaEl } = buildMessageContainer('assistant');
      msgEl.dataset.messageId = messageId;
      bodyEl.textContent = '';
      messagesEl.appendChild(msgEl);
      scrollToBottom();
      const node = {
        node: msgEl,
        content: '',
        bodyEl,
        metaEl,
      };
      assistantNodes.set(messageId, node);
      return node;
    }

    function startAssistantStream(messageId) {
      const node = ensureAssistantNode(messageId);
      node.bodyEl.textContent = '';
      node.content = '';
      node.metaEl.textContent = '';
      const existingActions = node.node.querySelector('.msg-actions');
      if (existingActions) {
        existingActions.remove();
      }
      scrollToBottom();
    }

    function appendAssistantStream(messageId, chunk) {
      const node = ensureAssistantNode(messageId);
      node.content += String(chunk || '');
      node.bodyEl.textContent = node.content;
      scrollToBottom();
    }

    function finalizeAssistantStream(messageId, fullContent, meta) {
      const content = String(fullContent || '');
      const node = assistantNodes.get(messageId);
      if (node && node.node && node.node.parentElement) {
        node.node.remove();
      }
      assistantNodes.delete(messageId);
      const finalNode = renderMessage('assistant', content, meta || null, messageId);
      assistantNodes.set(messageId, { node: finalNode, content });
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

    function renderAttachments(nextAttachments) {
      attachments = Array.isArray(nextAttachments) ? nextAttachments : [];
      attachmentsEl.textContent = '';
      for (const attachment of attachments) {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.textContent = attachment.kind + ': ' + attachment.label;

        const removeBtn = document.createElement('button');
        removeBtn.textContent = 'x';
        removeBtn.title = 'Remove attachment';
        removeBtn.dataset.attachmentId = attachment.id;
        chip.appendChild(removeBtn);
        attachmentsEl.appendChild(chip);
      }
    }

    function renderUserMessage(text, attachmentIds) {
      const selected = attachments.filter((a) => attachmentIds.includes(a.id));
      const attachmentSummary = selected.length > 0
        ? ('\n\n[Attachments]\n' + selected.map((a) => '- ' + a.kind + ': ' + a.label).join('\n'))
        : '';
      renderMessage('user', String(text || '') + attachmentSummary);
    }

    function sendPrompt() {
      const text = (inputEl.value || '').trim();
      if (!text) return;

      const attachmentIds = attachments.map((attachment) => attachment.id);
      renderUserMessage(text, attachmentIds);

      inputEl.value = '';
      inputEl.focus();
      setStatus('Sending...');

      vscode.postMessage({ type: 'userMessage', text, attachmentIds });
    }

    sendEl.addEventListener('click', sendPrompt);

    clearEl.addEventListener('click', () => {
      vscode.postMessage({ type: 'clear' });
    });

    attachSelectionEl.addEventListener('click', () => {
      vscode.postMessage({ type: 'attachSelection' });
    });

    attachFileEl.addEventListener('click', () => {
      vscode.postMessage({ type: 'attachFile' });
    });

    clearAttachmentsEl.addEventListener('click', () => {
      vscode.postMessage({ type: 'clearAttachments' });
    });

    attachmentsEl.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLButtonElement)) {
        return;
      }
      const id = target.dataset.attachmentId;
      if (!id) {
        return;
      }
      vscode.postMessage({ type: 'removeAttachment', id });
    });

    messagesEl.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLButtonElement)) {
        return;
      }
      const action = target.dataset.action;
      if (!action) {
        return;
      }
      const parent = target.closest('[data-message-id]');
      const messageId = parent && parent.getAttribute('data-message-id');
      if (!messageId) {
        return;
      }

      if (action === 'apply') {
        vscode.postMessage({ type: 'applyResponse', messageId, mode: 'replace' });
        return;
      }
      if (action === 'diff') {
        vscode.postMessage({ type: 'previewResponseDiff', messageId, mode: 'replace' });
        return;
      }
      if (action === 'copy') {
        vscode.postMessage({ type: 'copyResponse', messageId });
      }
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

      if (msg.type === 'attachments') {
        renderAttachments(msg.attachments || []);
        return;
      }

      if (msg.type === 'assistantStreamStart') {
        startAssistantStream(String(msg.messageId || ''));
        return;
      }

      if (msg.type === 'assistantStreamDelta') {
        appendAssistantStream(String(msg.messageId || ''), String(msg.chunk || ''));
        return;
      }

      if (msg.type === 'assistantStreamDone') {
        finalizeAssistantStream(
          String(msg.messageId || ''),
          String(msg.content || ''),
          msg.meta || null
        );
        setStatus('');
        return;
      }

      if (msg.type === 'assistantMessage') {
        renderMessage('assistant', msg.content || '', msg.meta || null, msg.messageId || '');
        setStatus('');
        return;
      }

      if (msg.type === 'errorMessage') {
        addError(msg.content || 'Unknown error');
        setStatus('');
        return;
      }

      if (msg.type === 'status') {
        setStatus(msg.status || '');
        return;
      }

      if (msg.type === 'setDraft') {
        inputEl.value = String(msg.text || '');
        inputEl.focus();
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
