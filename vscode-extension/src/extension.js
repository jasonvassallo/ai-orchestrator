/**
 * AI Orchestrator VS Code Extension
 * ==================================
 * Provides intelligent multi-model AI assistance in VS Code.
 * 
 * Security Features:
 * - No hardcoded API keys
 * - Secure credential storage using VS Code SecretStorage API
 * - Input validation and sanitization
 * - Audit logging
 * 
 * @author AI Orchestrator Team
 * @version 2.1.0
 */

const vscode = require('vscode');
const https = require('https');
const { spawn } = require('child_process');
const MODELS = require('../model-catalog.json');

/**
 * Secure credential manager using VS Code's built-in SecretStorage
 * Never stores API keys in settings or code
 */
class SecureCredentialManager {
    constructor(context) {
        this.secrets = context.secrets;
        this.context = context;
    }

    /**
     * Get API key for a provider
     * @param {string} provider - Provider name (e.g., 'openai', 'anthropic')
     * @returns {Promise<string|undefined>}
     */
    async getApiKey(provider) {
        const key = await this.secrets.get(`ai-orchestrator.${provider.toLowerCase()}`);
        
        // Fallback to environment variable
        if (!key) {
            const envVar = `${provider.toUpperCase()}_API_KEY`;
            return process.env[envVar];
        }
        
        return key;
    }

    /**
     * Store API key securely
     * @param {string} provider - Provider name
     * @param {string} apiKey - The API key to store
     */
    async setApiKey(provider, apiKey) {
        // Validate API key format (basic security check)
        if (!apiKey || apiKey.length < 10) {
            throw new Error('Invalid API key: too short');
        }
        
        await this.secrets.store(`ai-orchestrator.${provider.toLowerCase()}`, apiKey);
    }

    /**
     * Delete stored API key
     * @param {string} provider - Provider name
     */
    async deleteApiKey(provider) {
        await this.secrets.delete(`ai-orchestrator.${provider.toLowerCase()}`);
    }

    /**
     * Check if a provider has credentials configured
     * @param {string} provider - Provider name
     * @returns {Promise<boolean>}
     */
    async hasCredentials(provider) {
        const key = await this.getApiKey(provider);
        return !!key;
    }
}

/**
 * Input validator for security
 */
class InputValidator {
    static MAX_PROMPT_LENGTH = 500000;

    static SUSPICIOUS_PATTERNS = [
        /<\s*script\b/i,
        /javascript\s*:/i,
        /__proto__/,
        /\{\{.*\}\}/,
    ];

    /**
     * Validate user prompt
     * @param {string} prompt - The prompt to validate
     * @returns {{valid: boolean, error?: string}}
     */
    static validatePrompt(prompt) {
        if (!prompt || typeof prompt !== 'string') {
            return { valid: false, error: 'Invalid prompt: must be a non-empty string' };
        }

        if (prompt.length > this.MAX_PROMPT_LENGTH) {
            return { valid: false, error: `Prompt exceeds maximum length of ${this.MAX_PROMPT_LENGTH}` };
        }

        // Log suspicious patterns but don't block (could be legitimate code)
        for (const pattern of this.SUSPICIOUS_PATTERNS) {
            if (pattern.test(prompt)) {
                console.warn('Potentially suspicious pattern detected in prompt');
            }
        }

        return { valid: true };
    }

    /**
     * Sanitize text for logging (remove potential sensitive data)
     * @param {string} text - Text to sanitize
     * @param {number} maxLen - Maximum length
     * @returns {string}
     */
    static sanitizeForLogging(text, maxLen = 100) {
        if (!text) return '';
        
        let sanitized = text.substring(0, maxLen);
        // Redact anything that looks like an API key
        sanitized = sanitized.replace(
            /(sk-|api[_-]?key|bearer\s+)[a-zA-Z0-9\-_]{20,}/gi,
            '[REDACTED]'
        );
        
        return sanitized + (text.length > maxLen ? '...' : '');
    }
}

/**
 * Rate limiter with token bucket algorithm
 */
class RateLimiter {
    constructor() {
        this.requests = new Map(); // provider -> {timestamps: [], tokens: []}
        this.limits = {
            requestsPerMinute: 60,
            tokensPerMinute: 100000,
        };
    }

    /**
     * Check if request is allowed and wait if necessary
     * @param {string} provider - Provider name
     * @param {number} estimatedTokens - Estimated tokens for this request
     * @returns {Promise<boolean>}
     */
    async checkAndWait(provider, estimatedTokens = 1000) {
        const now = Date.now();
        const windowStart = now - 60000;

        let state = this.requests.get(provider);
        if (!state) {
            state = { timestamps: [], tokens: [] };
            this.requests.set(provider, state);
        }

        // Clean old entries
        state.timestamps = state.timestamps.filter(t => t > windowStart);
        state.tokens = state.tokens.filter(t => t.time > windowStart);

        // Check request limit
        if (state.timestamps.length >= this.limits.requestsPerMinute) {
            const waitTime = state.timestamps[0] - windowStart;
            await this.sleep(waitTime);
            return this.checkAndWait(provider, estimatedTokens);
        }

        // Check token limit
        const currentTokens = state.tokens.reduce((sum, t) => sum + t.count, 0);
        if (currentTokens + estimatedTokens > this.limits.tokensPerMinute) {
            const waitTime = state.tokens[0]?.time - windowStart || 1000;
            await this.sleep(waitTime);
            return this.checkAndWait(provider, estimatedTokens);
        }

        // Record this request
        state.timestamps.push(now);
        state.tokens.push({ time: now, count: estimatedTokens });

        return true;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Task classifier using keyword matching
 */
class TaskClassifier {
    static PATTERNS = {
        code: [
            /\b(code|program|script|function|class|implement|debug|refactor)\b/i,
            /\b(python|javascript|typescript|java|rust|go)\b/i,
            /\b(api|database|sql|algorithm)\b/i,
        ],
        reasoning: [
            /\b(prove|theorem|derive|logical|why|how|analyze)\b/i,
            /\b(research|investigate|complex|multi-step)\b/i,
        ],
        creative: [
            /\b(write|story|poem|essay|creative|fiction)\b/i,
            /\b(blog|article|content|marketing)\b/i,
        ],
        summarization: [
            /\b(summarize|summary|tldr|brief|condense)\b/i,
        ],
        math: [
            /\b(calculate|compute|solve|equation|integral)\b/i,
        ],
        local: [
            /\b(private|confidential|offline|local|sensitive)\b/i,
        ],
    };

    /**
     * Classify a prompt into task types
     * @param {string} prompt - The prompt to classify
     * @returns {Array<{type: string, confidence: number}>}
     */
    static classify(prompt) {
        const scores = {};
        const promptLower = prompt.toLowerCase();

        for (const [taskType, patterns] of Object.entries(this.PATTERNS)) {
            let score = 0;
            for (const pattern of patterns) {
                const matches = promptLower.match(pattern);
                if (matches) {
                    score += matches.length * 0.3;
                }
            }
            if (score > 0) {
                scores[taskType] = Math.min(score, 1.0);
            }
        }

        if (Object.keys(scores).length === 0) {
            scores['general'] = 0.5;
        }

        return Object.entries(scores)
            .map(([type, confidence]) => ({ type, confidence }))
            .sort((a, b) => b.confidence - a.confidence);
    }
}

/**
 * Provider implementations
 */
class OpenAIProvider {
    constructor(credentialManager) {
        this.credentialManager = credentialManager;
        this.name = 'openai';
    }

    async complete(messages, modelId, options = {}) {
        const apiKey = await this.credentialManager.getApiKey('openai');
        if (!apiKey) {
            throw new Error('OpenAI API key not configured. Run "AI Orchestrator: Configure Credentials"');
        }

        const body = JSON.stringify({
            model: modelId,
            messages: messages,
            max_tokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7,
        });

        return new Promise((resolve, reject) => {
            const req = https.request({
                hostname: 'api.openai.com',
                port: 443,
                path: '/v1/chat/completions',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Length': Buffer.byteLength(body),
                },
                timeout: 120000,
            }, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const response = JSON.parse(data);
                        if (response.error) {
                            reject(new Error(response.error.message));
                        } else {
                            resolve({
                                content: response.choices[0].message.content,
                                usage: {
                                    inputTokens: response.usage.prompt_tokens,
                                    outputTokens: response.usage.completion_tokens,
                                },
                            });
                        }
                    } catch (e) {
                        reject(new Error(`Failed to parse response: ${e.message}`));
                    }
                });
            });

            req.on('error', reject);
            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });
            req.write(body);
            req.end();
        });
    }
}

class AnthropicProvider {
    constructor(credentialManager) {
        this.credentialManager = credentialManager;
        this.name = 'anthropic';
    }

    async complete(messages, modelId, options = {}) {
        const apiKey = await this.credentialManager.getApiKey('anthropic');
        if (!apiKey) {
            throw new Error('Anthropic API key not configured. Run "AI Orchestrator: Configure Credentials"');
        }

        // Extract system message
        let system = '';
        const chatMessages = [];
        for (const msg of messages) {
            if (msg.role === 'system') {
                system = msg.content;
            } else {
                chatMessages.push(msg);
            }
        }

        const bodyObj = {
            model: modelId,
            max_tokens: options.maxTokens || 4096,
            messages: chatMessages,
        };
        if (system) {
            bodyObj.system = system;
        }

        const body = JSON.stringify(bodyObj);

        return new Promise((resolve, reject) => {
            const req = https.request({
                hostname: 'api.anthropic.com',
                port: 443,
                path: '/v1/messages',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': apiKey,
                    'anthropic-version': '2023-06-01',
                    'Content-Length': Buffer.byteLength(body),
                },
                timeout: 120000,
            }, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const response = JSON.parse(data);
                        if (response.error) {
                            reject(new Error(response.error.message));
                        } else {
                            resolve({
                                content: response.content[0].text,
                                usage: {
                                    inputTokens: response.usage.input_tokens,
                                    outputTokens: response.usage.output_tokens,
                                },
                            });
                        }
                    } catch (e) {
                        reject(new Error(`Failed to parse response: ${e.message}`));
                    }
                });
            });

            req.on('error', reject);
            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });
            req.write(body);
            req.end();
        });
    }
}

class MistralProvider {
    constructor(credentialManager) {
        this.credentialManager = credentialManager;
        this.name = 'mistral';
    }

    async complete(messages, modelId, options = {}) {
        const apiKey = await this.credentialManager.getApiKey('mistral');
        if (!apiKey) {
            throw new Error('Mistral API key not configured. Run "AI Orchestrator: Configure API Credentials"');
        }

        const body = JSON.stringify({
            model: modelId,
            messages: messages,
            max_tokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7,
        });

        return new Promise((resolve, reject) => {
            const req = https.request({
                hostname: 'api.mistral.ai',
                port: 443,
                path: '/v1/chat/completions',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Length': Buffer.byteLength(body),
                },
                timeout: 120000,
            }, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const response = JSON.parse(data);
                        if (response.error) {
                            reject(new Error(response.error.message || 'Mistral API error'));
                            return;
                        }
                        resolve({
                            content: response.choices?.[0]?.message?.content || '',
                            usage: {
                                inputTokens: response.usage?.prompt_tokens || 0,
                                outputTokens: response.usage?.completion_tokens || 0,
                            },
                        });
                    } catch (e) {
                        reject(new Error(`Failed to parse response: ${e.message}`));
                    }
                });
            });

            req.on('error', reject);
            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });
            req.write(body);
            req.end();
        });
    }
}

const MLX_BRIDGE_SCRIPT = `
import asyncio
import json
import sys
from src.orchestrator import AIOrchestrator

payload = json.loads(sys.stdin.read() or "{}")

async def run() -> None:
    orchestrator = AIOrchestrator(
        prefer_local=True,
        cost_optimize=False,
        incognito=True,
    )
    response = await orchestrator.query(
        prompt=payload.get("prompt", ""),
        system_prompt=payload.get("system_prompt") or None,
        model_override=payload.get("model_override"),
        max_tokens=int(payload.get("max_tokens", 4096)),
        temperature=float(payload.get("temperature", 0.7)),
    )
    if not response.success:
        print(json.dumps({"ok": False, "error": response.error or "Unknown local model error"}))
        return

    usage = response.usage if isinstance(response.usage, dict) else {}
    print(
        json.dumps(
            {
                "ok": True,
                "content": response.content,
                "usage": {
                    "inputTokens": int(usage.get("input_tokens", 0)),
                    "outputTokens": int(usage.get("output_tokens", 0)),
                },
            }
        )
    )

asyncio.run(run())
`;

class MlxProvider {
    constructor() {
        this.name = 'mlx';
    }

    _resolveRuntimeConfig() {
        const config = vscode.workspace.getConfiguration('ai-orchestrator');
        const pythonExecutable = (config.get('pythonExecutable', 'python3') || 'python3').trim();

        let projectPath = (config.get('pythonProjectPath', '') || '').trim();
        if (!projectPath) {
            projectPath = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
        }

        if (!projectPath) {
            throw new Error(
                'No project path configured for MLX runtime. Set "ai-orchestrator.pythonProjectPath" in settings.'
            );
        }

        return { pythonExecutable, projectPath };
    }

    _stringifyContent(content) {
        if (typeof content === 'string') {
            return content;
        }
        if (Array.isArray(content)) {
            return content
                .map((item) => {
                    if (typeof item === 'string') {
                        return item;
                    }
                    if (item && typeof item === 'object' && item.type === 'text') {
                        return item.text || '';
                    }
                    try {
                        return JSON.stringify(item);
                    } catch {
                        return String(item);
                    }
                })
                .join('\n');
        }
        if (content === null || content === undefined) {
            return '';
        }
        return String(content);
    }

    _extractPrompt(messages) {
        const nonSystem = messages.filter((msg) => msg.role !== 'system');
        if (nonSystem.length === 0) {
            return '';
        }
        return nonSystem
            .map((msg) => `${String(msg.role || 'user').toUpperCase()}: ${this._stringifyContent(msg.content)}`)
            .join('\n\n');
    }

    _extractSystemPrompt(messages) {
        const systemMessage = messages.find((msg) => msg.role === 'system');
        if (!systemMessage) {
            return '';
        }
        return this._stringifyContent(systemMessage.content);
    }

    async complete(messages, modelId, options = {}) {
        const { pythonExecutable, projectPath } = this._resolveRuntimeConfig();
        const payload = {
            prompt: this._extractPrompt(messages),
            system_prompt: this._extractSystemPrompt(messages),
            model_override: modelId,
            max_tokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7,
        };

        return new Promise((resolve, reject) => {
            const child = spawn(
                pythonExecutable,
                ['-c', MLX_BRIDGE_SCRIPT],
                {
                    cwd: projectPath,
                    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
                }
            );

            let stdout = '';
            let stderr = '';

            child.stdout.on('data', (chunk) => {
                stdout += String(chunk);
            });

            child.stderr.on('data', (chunk) => {
                stderr += String(chunk);
            });

            child.on('error', (err) => {
                reject(new Error(`Failed to start local MLX runtime: ${err.message}`));
            });

            child.on('close', (code) => {
                if (code !== 0) {
                    reject(
                        new Error(
                            `Local MLX runtime failed (exit ${code}). ${stderr.trim() || 'No stderr output.'}`
                        )
                    );
                    return;
                }

                let parsed;
                try {
                    parsed = JSON.parse(stdout.trim());
                } catch (error) {
                    reject(
                        new Error(
                            `Failed to parse local MLX response: ${error.message}. Raw output: ${stdout.trim()}`
                        )
                    );
                    return;
                }

                if (!parsed.ok) {
                    reject(new Error(parsed.error || 'Local MLX query failed.'));
                    return;
                }

                resolve({
                    content: parsed.content || '',
                    usage: {
                        inputTokens: parsed.usage?.inputTokens || 0,
                        outputTokens: parsed.usage?.outputTokens || 0,
                    },
                });
            });

            child.stdin.write(JSON.stringify(payload));
            child.stdin.end();
        });
    }
}

/**
 * Main AI Orchestrator class
 */
class AIOrchestrator {
    constructor(context) {
        this.credentialManager = new SecureCredentialManager(context);
        this.rateLimiter = new RateLimiter();
        this.providers = {
            openai: new OpenAIProvider(this.credentialManager),
            anthropic: new AnthropicProvider(this.credentialManager),
            mistral: new MistralProvider(this.credentialManager),
            mlx: new MlxProvider(),
        };
        this.conversationHistory = [];
    }

    /**
     * Select the best model for the given task types
     * @param {Array} taskTypes - Classified task types
     * @param {Object} options - Selection options
     * @returns {Object} Selected model
     */
    selectModel(taskTypes, options = {}) {
        const primaryTask = taskTypes[0]?.type || 'general';
        
        // Filter models by task type
        const candidates = Object.entries(MODELS)
            .filter(([_, model]) => {
                if (!this.providers[model.provider]) {
                    return false;
                }
                if (options.preferLocal && model.provider !== 'mlx') {
                    return false;
                }
                return model.taskTypes.includes(primaryTask);
            })
            .map(([key, model]) => {
                let score = 0;
                
                // Task match score
                for (const { type, confidence } of taskTypes) {
                    if (model.taskTypes.includes(type)) {
                        score += confidence * 10;
                    }
                }
                
                // Cost optimization
                if (options.costOptimize) {
                    score -= model.costPer1kInput * 100;
                }
                
                return { key, model, score };
            })
            .sort((a, b) => b.score - a.score);

        if (candidates.length === 0) {
            // Fallback to GPT-4o
            return { key: 'gpt-4o', model: MODELS['gpt-4o'] };
        }

        return { key: candidates[0].key, model: candidates[0].model };
    }

    /**
     * Send a query to the AI
     * @param {string} prompt - User prompt
     * @param {Object} options - Query options
     * @returns {Promise<Object>} Response
     */
    async query(prompt, options = {}) {
        const startTime = Date.now();

        // Validate input
        const validation = InputValidator.validatePrompt(prompt);
        if (!validation.valid) {
            throw new Error(validation.error);
        }

        // Classify task
        const taskTypes = TaskClassifier.classify(prompt);

        // Select model
        let model, modelKey;
        if (options.modelOverride && MODELS[options.modelOverride]) {
            modelKey = options.modelOverride;
            model = MODELS[modelKey];
        } else {
            const selection = this.selectModel(taskTypes, options);
            modelKey = selection.key;
            model = selection.model;
        }

        // Get provider
        const provider = this.providers[model.provider];
        if (!provider) {
            throw new Error(`Provider ${model.provider} not available`);
        }

        // Check rate limits
        await this.rateLimiter.checkAndWait(model.provider, options.maxTokens || 4096);

        // Build messages
        const messages = [];
        if (options.systemPrompt) {
            messages.push({ role: 'system', content: options.systemPrompt });
        }
        messages.push(...this.conversationHistory);
        messages.push({ role: 'user', content: prompt });

        // Execute request
        const response = await provider.complete(messages, model.modelId, {
            maxTokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7,
        });

        // Update history
        this.conversationHistory.push({ role: 'user', content: prompt });
        this.conversationHistory.push({ role: 'assistant', content: response.content });

        return {
            content: response.content,
            model: model.name,
            provider: model.provider,
            usage: response.usage,
            latencyMs: Date.now() - startTime,
        };
    }

    clearHistory() {
        this.conversationHistory = [];
    }
}

// Extension activation
let orchestrator = null;
let outputChannel = null;

function activate(context) {
    console.log('AI Orchestrator extension activated');

    outputChannel = vscode.window.createOutputChannel('AI Orchestrator');
    orchestrator = new AIOrchestrator(context);

    const getConfiguredQueryOptions = (overrides = {}) => {
        const config = vscode.workspace.getConfiguration('ai-orchestrator');
        const selectedModel = (config.get('selectedModel', '') || '').trim();
        const options = {
            preferLocal: config.get('preferLocal', false),
            costOptimize: config.get('costOptimize', false),
            maxTokens: config.get('maxTokens', 4096),
            temperature: config.get('temperature', 0.7),
            systemPrompt: config.get('systemPrompt', ''),
        };

        if (selectedModel && MODELS[selectedModel]) {
            const selectedProvider = MODELS[selectedModel].provider;
            if (orchestrator.providers[selectedProvider]) {
                options.modelOverride = selectedModel;
            } else {
                outputChannel.appendLine(
                    `[Warning] Selected model "${selectedModel}" uses unsupported provider "${selectedProvider}". Falling back to auto-routing.`
                );
            }
        }

        return { ...options, ...overrides };
    };

    // Register commands
    const commands = [
        vscode.commands.registerCommand('ai-orchestrator.query', async () => {
            const prompt = await vscode.window.showInputBox({
                prompt: 'Enter your prompt',
                placeHolder: 'Ask anything...',
            });

            if (!prompt) return;

            try {
                outputChannel.show();
                outputChannel.appendLine(`\n[Query] ${InputValidator.sanitizeForLogging(prompt, 200)}`);

                const response = await orchestrator.query(
                    prompt,
                    getConfiguredQueryOptions()
                );

                outputChannel.appendLine(`[${response.model}] (${response.latencyMs}ms)`);
                outputChannel.appendLine('-'.repeat(60));
                outputChannel.appendLine(response.content);
                outputChannel.appendLine('-'.repeat(60));
                outputChannel.appendLine(
                    `Tokens: ${response.usage.inputTokens} in / ${response.usage.outputTokens} out`
                );

                // Show in info message
                vscode.window.showInformationMessage(
                    `Response from ${response.model}`,
                    'View Output'
                ).then(selection => {
                    if (selection === 'View Output') {
                        outputChannel.show();
                    }
                });
            } catch (error) {
                vscode.window.showErrorMessage(`AI Orchestrator Error: ${error.message}`);
                outputChannel.appendLine(`[Error] ${error.message}`);
            }
        }),

        vscode.commands.registerCommand('ai-orchestrator.configureCredentials', async () => {
            const providers = ['OpenAI', 'Anthropic', 'Mistral'];
            const selected = await vscode.window.showQuickPick(providers, {
                placeHolder: 'Select provider to configure',
            });

            if (!selected) return;

            const apiKey = await vscode.window.showInputBox({
                prompt: `Enter API key for ${selected}`,
                password: true,
                placeHolder: 'API key (will be stored securely)',
            });

            if (!apiKey) return;

            try {
                await orchestrator.credentialManager.setApiKey(selected.toLowerCase(), apiKey);
                vscode.window.showInformationMessage(`${selected} API key saved securely`);
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to save API key: ${error.message}`);
            }
        }),

        vscode.commands.registerCommand('ai-orchestrator.selectModel', async () => {
            const config = vscode.workspace.getConfiguration('ai-orchestrator');
            const currentModel = (config.get('selectedModel', '') || '').trim();
            const models = [
                {
                    label: 'Automatic',
                    description: 'Use auto-routing based on task classification',
                    detail: 'No forced model override',
                    key: '',
                },
                ...Object.entries(MODELS)
                    .filter(([_, model]) => Boolean(orchestrator.providers[model.provider]))
                    .map(([key, model]) => ({
                        label: model.name,
                        description: `${model.provider} - ${model.strengths.join(', ')}`,
                        detail: `$${model.costPer1kInput}/1k input, $${model.costPer1kOutput}/1k output`,
                        key: key,
                    })),
            ];

            const selected = await vscode.window.showQuickPick(models, {
                placeHolder: 'Select a model',
            });

            if (selected) {
                // Store selection in workspace settings
                await config.update('selectedModel', selected.key, vscode.ConfigurationTarget.Global);
                if (!selected.key) {
                    const current = currentModel ? ` (was ${currentModel})` : '';
                    vscode.window.showInformationMessage(`Model selection reset to Automatic${current}`);
                } else {
                    vscode.window.showInformationMessage(`Selected model: ${selected.label}`);
                }
            }
        }),

        vscode.commands.registerCommand('ai-orchestrator.clearHistory', () => {
            orchestrator.clearHistory();
            vscode.window.showInformationMessage('Conversation history cleared');
        }),

        vscode.commands.registerCommand('ai-orchestrator.explainCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            const selection = editor.document.getText(editor.selection);
            if (!selection) {
                vscode.window.showWarningMessage('No code selected');
                return;
            }

            try {
                outputChannel.show();
                outputChannel.appendLine('\n[Explain Code]');

                const response = await orchestrator.query(
                    `Explain the following code:\n\n\`\`\`\n${selection}\n\`\`\``,
                    getConfiguredQueryOptions({
                        systemPrompt: 'You are a helpful coding assistant. Explain code clearly and concisely.',
                    })
                );

                outputChannel.appendLine(response.content);
            } catch (error) {
                vscode.window.showErrorMessage(`Error: ${error.message}`);
            }
        }),

        vscode.commands.registerCommand('ai-orchestrator.improveCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            const selection = editor.document.getText(editor.selection);
            if (!selection) {
                vscode.window.showWarningMessage('No code selected');
                return;
            }

            try {
                outputChannel.show();
                outputChannel.appendLine('\n[Improve Code]');
                const configuredOptions = getConfiguredQueryOptions();

                const response = await orchestrator.query(
                    `Improve the following code for better performance, readability, and security. Provide the improved code with explanations:\n\n\`\`\`\n${selection}\n\`\`\``,
                    {
                        ...configuredOptions,
                        systemPrompt: 'You are an expert code reviewer. Improve code for performance, security, and readability.',
                        modelOverride: configuredOptions.modelOverride || 'claude-sonnet-4.5',
                    }
                );

                outputChannel.appendLine(response.content);
            } catch (error) {
                vscode.window.showErrorMessage(`Error: ${error.message}`);
            }
        }),
    ];

    context.subscriptions.push(...commands, outputChannel);
}

function deactivate() {
    console.log('AI Orchestrator extension deactivated');
}

module.exports = { activate, deactivate };
