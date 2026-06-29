const API_ENDPOINT = "%%API_ENDPOINT%%";
const REQUEST_TIMEOUT_MS = 30000;
const STAGE_TIMINGS_MS = [0, 800, 2000, 3400, 5000];

const chatContainer = document.getElementById('chat-container');
const welcome = document.getElementById('welcome');
const questionInput = document.getElementById('question-input');
const submitBtn = document.getElementById('submit-btn');
const commandFeedback = document.getElementById('command-feedback');
let hasAsked = false;
let submitInProgress = false;
let stageTimerIds = [];
let commandFeedbackTimerId = null;

function syncAppHeight() {
    if (typeof window === 'undefined') return;
    const root = document.getElementById('root');
    if (!root) return;
    const viewportHeight = window.visualViewport && window.visualViewport.height
        ? window.visualViewport.height
        : window.innerHeight;
    if (viewportHeight) {
        root.style.height = Math.round(viewportHeight) + 'px';
    }
    if (
        document.activeElement === questionInput &&
        typeof questionInput.scrollIntoView === 'function'
    ) {
        questionInput.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
}

if (typeof window !== 'undefined') {
    syncAppHeight();
    window.addEventListener('resize', syncAppHeight);
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', syncAppHeight);
        window.visualViewport.addEventListener('scroll', syncAppHeight);
    }
}

function scrollChatToBottom() {
    const chat = document.getElementById('chat-container');
    if (!chat) return;

    if (typeof requestAnimationFrame !== 'undefined') {
        requestAnimationFrame(function() {
            chat.scrollTop = chat.scrollHeight;
        });
    } else {
        chat.scrollTop = chat.scrollHeight;
    }
}

function scrollChatToMessageStart(message) {
    const chat = document.getElementById('chat-container');
    if (!chat || !message) return;

    const scroll = function() {
        if (
            typeof chat.getBoundingClientRect === 'function' &&
            typeof message.getBoundingClientRect === 'function'
        ) {
            const chatRect = chat.getBoundingClientRect();
            const messageRect = message.getBoundingClientRect();
            chat.scrollTop = Math.max(0, chat.scrollTop + messageRect.top - chatRect.top);
            return;
        }

        chat.scrollTop = Math.max(0, message.offsetTop || 0);
    };

    if (typeof requestAnimationFrame !== 'undefined') {
        requestAnimationFrame(scroll);
    } else {
        scroll();
    }
}

function copyText(text) {
    if (typeof navigator === 'undefined' || !navigator.clipboard || !navigator.clipboard.writeText) {
        return Promise.reject(new Error('Clipboard unavailable'));
    }
    return navigator.clipboard.writeText(text);
}

function isClearCommand(value) {
    return /^\s*\/\s*clear\s*$/i.test(String(value || ''));
}

function syncCommandInputStyle() {
    if (isClearCommand(questionInput.value)) {
        addClass(questionInput, 'command-input');
    } else {
        removeClass(questionInput, 'command-input');
    }
}

questionInput.addEventListener('input', syncCommandInputStyle);

function askQuestion(text) {
    questionInput.value = text;
    syncCommandInputStyle();
    handleSubmit(new Event('submit'));
}

function addMessage(type, content) {
    const div = document.createElement('div');
    div.className = 'message ' + type;
    div.textContent = content;
    chatContainer.appendChild(div);
    scrollChatToBottom();
    return div;
}

async function readResponseBody(resp) {
    const contentType = resp.headers.get('content-type') || '';
    if (contentType.indexOf('application/json') !== -1) {
        try {
            return await resp.json();
        } catch (err) {
            return null;
        }
    }

    try {
        const text = await resp.text();
        return text ? { raw: text } : null;
    } catch (err) {
        return null;
    }
}

function clearNode(container) {
    container.textContent = '';
}

function showCommandFeedback(message) {
    if (!commandFeedback) return;
    if (commandFeedbackTimerId !== null) clearTimeout(commandFeedbackTimerId);
    commandFeedback.textContent = message;
    commandFeedback.hidden = false;
    commandFeedbackTimerId = setTimeout(function () {
        commandFeedback.hidden = true;
        commandFeedbackTimerId = null;
    }, 1800);
}

function resetChat() {
    clearNode(chatContainer);
    hasAsked = false;
    if (welcome) {
        welcome.style.display = '';
        chatContainer.appendChild(welcome);
    }
    questionInput.value = '';
    syncCommandInputStyle();
    questionInput.focus();
    showCommandFeedback('Chat cleared');
}

function isValidHttpUrl(text) {
    try {
        const url = new URL(text);
        return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (err) {
        return false;
    }
}

function parseAWSService(url) {
    try {
        const match = url.match(/docs\.aws\.amazon\.com\/([^\/]+)\//);
        if (match && match[1]) {
            return match[1].toUpperCase();
        }
    } catch (e) {}
    return 'AWS';
}

function normalizeSourceRows(rawSources) {
    return (rawSources || []).map(function(source) {
        var rawUrl = (typeof source === 'string') ? source : ((source && typeof source === 'object' ? (source.url || source.source_url || '') : ''));
        var url = String(rawUrl).trim();
        var service = (source && typeof source === 'object' && source.service) || parseAWSService(url) || 'AWS';
        var score = (source && typeof source === 'object') ? source.score : undefined;
        return { url: url, service: service, score: score };
    });
}

function appendMaybeLink(parent, text) {
    if (!text) return;
    if (isValidHttpUrl(text)) {
        const a = document.createElement('a');
        a.href = text;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = text;
        parent.appendChild(a);
        return;
    }
    parent.appendChild(document.createTextNode(text));
}

function appendTextWithLinks(parent, text) {
    const parts = String(text).split(/(https?:\/\/[^\s<>"']+)/g);
    parts.forEach(function (part) {
        appendMaybeLink(parent, part);
    });
}

function appendTextWithInlineCode(parent, text) {
    const parts = String(text).split(/(`[^`]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g);
    parts.forEach(function (part) {
        if (!part) return;
        if (part.length > 1 && part[0] === '`' && part[part.length - 1] === '`') {
            const code = document.createElement('code');
            code.textContent = part.slice(1, -1);
            parent.appendChild(code);
        } else if (part.length > 3 && part.slice(0, 2) === '**' && part.slice(-2) === '**') {
            const strong = document.createElement('strong');
            appendTextWithLinks(strong, part.slice(2, -2));
            parent.appendChild(strong);
        } else if (part.length > 1 && part[0] === '*' && part[part.length - 1] === '*') {
            const em = document.createElement('em');
            appendTextWithLinks(em, part.slice(1, -1));
            parent.appendChild(em);
        } else {
            appendTextWithLinks(parent, part);
        }
    });
}

function appendParagraph(container, lines) {
    const text = lines.map(function (line) { return line.trim(); }).join(' ').trim();
    if (!text) return;
    const p = document.createElement('p');
    appendTextWithInlineCode(p, text);
    container.appendChild(p);
}

function splitTableRow(line) {
    let trimmed = line.trim();
    if (trimmed[0] === '|') trimmed = trimmed.slice(1);
    if (trimmed[trimmed.length - 1] === '|') trimmed = trimmed.slice(0, -1);
    return trimmed.split('|').map(function (cell) { return cell.trim(); });
}

function isPipeRow(line) {
    const trimmed = line.trim();
    return trimmed.indexOf('|') !== -1 && trimmed[0] === '|' && trimmed[trimmed.length - 1] === '|';
}

function isTableStart(lines, index) {
    if (index + 1 >= lines.length || !isPipeRow(lines[index]) || !isPipeRow(lines[index + 1])) {
        return false;
    }
    const headers = splitTableRow(lines[index]);
    const separators = splitTableRow(lines[index + 1]);
    return headers.length > 0 && headers.length === separators.length && separators.every(function (cell) {
        return /^:?-{3,}:?$/.test(cell.replace(/\s+/g, ''));
    });
}

function parseTable(lines, index, container) {
    const tableScroll = document.createElement('div');
    tableScroll.className = 'table-scroll';
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    splitTableRow(lines[index]).forEach(function (cell) {
        const th = document.createElement('th');
        appendTextWithInlineCode(th, cell);
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    let i = index + 2;
    while (i < lines.length && isPipeRow(lines[i])) {
        const row = document.createElement('tr');
        splitTableRow(lines[i]).forEach(function (cell) {
            const td = document.createElement('td');
            appendTextWithInlineCode(td, cell);
            row.appendChild(td);
        });
        tbody.appendChild(row);
        i++;
    }
    table.appendChild(tbody);
    tableScroll.appendChild(table);
    container.appendChild(tableScroll);
    return i;
}

function parseList(lines, index, container) {
    const ordered = /^\s*\d+\.\s+/.test(lines[index]);
    const list = document.createElement(ordered ? 'ol' : 'ul');
    let i = index;
    const pattern = ordered ? /^\s*\d+\.\s+(.+)$/ : /^\s*[-*]\s+(.+)$/;
    while (i < lines.length && pattern.test(lines[i])) {
        const match = lines[i].match(pattern);
        const li = document.createElement('li');
        appendTextWithInlineCode(li, match ? match[1] : lines[i].trim());
        list.appendChild(li);
        i++;
    }
    container.appendChild(list);
    return i;
}

function renderAnswer(container, rawText) {
    clearNode(container);
    addClass(container, 'answer-rendered');

    const lines = String(rawText || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
    let i = 0;
    let paragraph = [];

    function flushParagraph() {
        appendParagraph(container, paragraph);
        paragraph = [];
    }

    while (i < lines.length) {
        const line = lines[i];
        const trimmed = line.trim();

        if (!trimmed) {
            flushParagraph();
            i++;
            continue;
        }

        if (/^```/.test(trimmed)) {
            flushParagraph();
            i++;
            const codeLines = [];
            while (i < lines.length && !/^```/.test(lines[i].trim())) {
                codeLines.push(lines[i]);
                i++;
            }
            if (i < lines.length) i++;
            const pre = document.createElement('pre');
            const code = document.createElement('code');
            code.textContent = codeLines.join('\n');
            pre.appendChild(code);
            container.appendChild(pre);
            continue;
        }

        const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
            flushParagraph();
            const h = document.createElement('h' + (heading[1].length + 2));
            appendTextWithInlineCode(h, heading[2]);
            container.appendChild(h);
            i++;
            continue;
        }

        if (isTableStart(lines, i)) {
            flushParagraph();
            i = parseTable(lines, i, container);
            continue;
        }

        if (/^\s*(?:[-*]|\d+\.)\s+/.test(line)) {
            flushParagraph();
            i = parseList(lines, i, container);
            continue;
        }

        paragraph.push(line);
        i++;
    }

    flushParagraph();

    if (!container.children || container.children.length === 0) {
        appendParagraph(container, [String(rawText || '')]);
    }
}

async function handleSubmit(e) {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();
    const question = questionInput.value.trim();
    if (!question || submitInProgress) return;

    if (isClearCommand(question)) {
        resetChat();
        return;
    }

    if (!hasAsked) {
        document.getElementById('welcome').style.display = 'none';
        hasAsked = true;
    }
    submitInProgress = true;

    addMessage('user', question);
    questionInput.value = '';
    syncCommandInputStyle();

    submitBtn.disabled = true;
    const loadingMsg = addMessage('loading', '');
    loadingMsg.innerHTML = '<div class="loading-title"><span class="loading-dots">Processing RAG request</span></div><div class="loading-stages"><span>Question</span><span>Embed</span><span>Search</span><span>Generate</span><span>Sources</span></div>';
    const loadingStageNodes = Array.from(loadingMsg.querySelectorAll('.loading-stages span'));
    if (loadingStageNodes.length > 0) {
        addClass(loadingStageNodes[0], 'stage-active');
        STAGE_TIMINGS_MS.slice(1).forEach(function (delay, index) {
            const timerId = setTimeout(function () {
                loadingStageNodes.forEach(function (node, nodeIndex) {
                    if (nodeIndex === index + 1) {
                        addClass(node, 'stage-active');
                    } else {
                        removeClass(node, 'stage-active');
                    }
                });
            }, delay);
            stageTimerIds.push(timerId);
        });
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS);

    try {
        const headers = { 'Content-Type': 'application/json' };
        const resp = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ question: question }),
            signal: controller.signal,
        });

        const data = await readResponseBody(resp);
        stageTimerIds.forEach(clearTimeout);
        stageTimerIds = [];
        loadingMsg.remove();

        if (!resp.ok) {
            const serverError = data && typeof data === 'object' && data.error
                ? data.error
                : null;
            const rawBody = data && typeof data === 'object' && data.raw
                ? ': ' + data.raw
                : '';
            addMessage(
                'error',
                serverError
                    ? 'Backend error (' + resp.status + '): ' + serverError
                    : 'Backend error (' + resp.status + ')' + rawBody
            );
            return;
        }

        if (!data || typeof data !== 'object') {
            addMessage('error', 'Backend error: response was not valid JSON.');
            return;
        }

        if (data.error) {
            addMessage('error', 'Backend error (' + resp.status + '): ' + data.error);
            return;
        }

        if (typeof data.answer !== 'string' || data.answer.trim() === '') {
            addMessage('error', 'Backend error (' + resp.status + '): response did not include an answer.');
            return;
        }

        const rawRows = normalizeSourceRows(data.sources);
        const seen = new Set();
        const sourceRows = rawRows.filter(function(s) {
            if (!s.url || seen.has(s.url)) return false;
            seen.add(s.url);
            return true;
        });
        const sourceCount = sourceRows.length;
        const msgDiv = addMessage('assistant', '');

        const body = document.createElement('div');
        body.className = 'assistant-body';
        const answerDiv = document.createElement('div');
        answerDiv.className = 'answer';
        renderAnswer(answerDiv, data.answer);
        body.appendChild(answerDiv);
        msgDiv.appendChild(body);

        if (sourceRows.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            const strong = document.createElement('strong');
            strong.textContent = 'Sources:';
            sourcesDiv.appendChild(strong);

            const sourceList = document.createElement('div');
            sourceList.className = 'source-list';

            sourceRows.forEach(function (source, i) {
                const url = source.url;
                const sourceCard = document.createElement('div');
                sourceCard.className = 'source-card';

                const sourceIndex = document.createElement('span');
                sourceIndex.className = 'source-index';
                sourceIndex.textContent = String(i + 1).padStart(2, '0');
                sourceCard.appendChild(sourceIndex);

                const sourceService = document.createElement('span');
                sourceService.className = 'source-service';
                sourceService.textContent = source.service || 'AWS';
                sourceCard.appendChild(sourceService);

                if (isValidHttpUrl(url)) {
                    const a = document.createElement('a');
                    a.className = 'source-link';
                    a.href = url;
                    a.target = '_blank';
                    a.rel = 'noopener noreferrer';
                    a.textContent = url;
                    sourceCard.appendChild(a);
                } else {
                    const span = document.createElement('span');
                    span.className = 'source-link';
                    span.textContent = url;
                    sourceCard.appendChild(span);
                }

                sourceList.appendChild(sourceCard);
            });

            sourcesDiv.appendChild(sourceList);
            msgDiv.appendChild(sourcesDiv);
        }

        const answerActionsDiv = document.createElement('div');
        answerActionsDiv.className = 'answer-actions';

        const copyAnswerBtn = document.createElement('button');
        copyAnswerBtn.type = 'button';
        copyAnswerBtn.className = 'answer-action';
        copyAnswerBtn.setAttribute('data-action', 'copy-answer');
        copyAnswerBtn.textContent = 'Copy answer';
        answerActionsDiv.appendChild(copyAnswerBtn);

        const copySourcesBtn = document.createElement('button');
        copySourcesBtn.type = 'button';
        copySourcesBtn.className = 'answer-action';
        copySourcesBtn.setAttribute('data-action', 'copy-sources');
        copySourcesBtn.textContent = 'Copy sources';
        answerActionsDiv.appendChild(copySourcesBtn);

        msgDiv.appendChild(answerActionsDiv);

        scrollChatToMessageStart(msgDiv);

    } catch (err) {
        stageTimerIds.forEach(clearTimeout);
        stageTimerIds = [];
        loadingMsg.remove();
        if (err.name === 'AbortError') {
            addMessage('error', 'Request timed out after ' + (REQUEST_TIMEOUT_MS / 1000) + ' seconds. Please try again.');
        } else {
            addMessage('error', 'Network error: ' + err.message + '. Check that the API endpoint is configured.');
        }
    } finally {
        submitInProgress = false;
        clearTimeout(timeoutId);
        submitBtn.disabled = false;
        questionInput.focus();
    }
}

function removeClass(el, className) {
    if (el.classList && typeof el.classList.remove === 'function') {
        el.classList.remove(className);
    } else {
        el.className = el.className.replace(new RegExp('\\s*\\b' + className + '\\b', 'g'), '');
    }
}

function addClass(el, className) {
    if (el.classList && typeof el.classList.add === 'function') {
        el.classList.add(className);
    } else if (!new RegExp('\\b' + className + '\\b').test(el.className)) {
        el.className = (el.className + ' ' + className).trim();
    }
}

function toggleMobileMenu(toggleBtn) {
    const sidebar = typeof document.querySelector === 'function'
        ? document.querySelector('.sidebar')
        : null;
    if (!sidebar) return;
    const isOpen = !new RegExp('\\bmobile-menu-open\\b').test(sidebar.className || '');
    if (isOpen) {
        addClass(sidebar, 'mobile-menu-open');
    } else {
        removeClass(sidebar, 'mobile-menu-open');
    }
    if (toggleBtn && typeof toggleBtn.setAttribute === 'function') {
        toggleBtn.setAttribute('aria-expanded', String(isOpen));
    }
}

function closeMobileMenu() {
    const sidebar = typeof document.querySelector === 'function'
        ? document.querySelector('.sidebar')
        : null;
    if (sidebar) removeClass(sidebar, 'mobile-menu-open');
    const toggleBtn = typeof document.querySelector === 'function'
        ? document.querySelector('.mobile-menu-toggle')
        : null;
    if (toggleBtn && typeof toggleBtn.setAttribute === 'function') {
        toggleBtn.setAttribute('aria-expanded', 'false');
    }
}

function navigate(name, btnEl) {
    if (typeof document.querySelectorAll !== 'function') return;
    const sections = document.querySelectorAll('[data-section]');
    for (let i = 0; i < sections.length; i++) {
        removeClass(sections[i], 'section-active');
    }
    const target = typeof document.querySelector === 'function'
        ? document.querySelector('[data-section="' + name + '"]')
        : null;
    if (target) {
        addClass(target, 'section-active');
        if (typeof target.focus === 'function') {
            target.focus({ preventScroll: true });
        }
        const contentInner = target.parentNode;
        if (contentInner && typeof contentInner.scrollTop === 'number') {
            contentInner.scrollTop = 0;
        }
    }
    const buttons = document.querySelectorAll('.nav-btn');
    for (let i = 0; i < buttons.length; i++) {
        removeClass(buttons[i], 'active');
    }
    if (btnEl) {
        addClass(btnEl, 'active');
    }
    closeMobileMenu();
}

if (typeof window !== 'undefined' && typeof setInterval === 'function') {
    const pad = function (n) { return String(n).padStart(2, '0'); };
    const updateClock = function () {
        const now = new Date();
        const t = document.getElementById('clock-time');
        const d = document.getElementById('clock-date');
        if (t) t.textContent = pad(now.getHours()) + ':' + pad(now.getMinutes()) + ':' + pad(now.getSeconds());
        if (d) d.textContent = pad(now.getMonth() + 1) + '.' + pad(now.getDate()) + '.' + String(now.getFullYear()).slice(2);
    };
    updateClock();
    setInterval(updateClock, 1000);

    const typeText = '// Retrieval over AWS documentation';
    const typeEl = document.getElementById('typing-text');
    if (typeEl) {
        let i = 0;
        const it = setInterval(function () {
            i++;
            typeEl.textContent = typeText.slice(0, i);
            if (i >= typeText.length) clearInterval(it);
        }, 55);
    }
}

/* Event Listeners for Answer Actions */
chatContainer.addEventListener('click', function(e) {
    const button = e.target.closest('[data-action]');
    if (!button) return;
    const action = button.getAttribute('data-action');

    if (action === 'copy-answer') {
        const answerDiv = button.closest('.message.assistant')?.querySelector('.answer');
        if (answerDiv) {
            copyText(answerDiv.textContent).catch(function() {});
        }
    } else if (action === 'copy-sources') {
        const sourcesDiv = button.closest('.message.assistant')?.querySelector('.sources');
        if (sourcesDiv) {
            const urls = Array.from(sourcesDiv.querySelectorAll('a.source-link'))
                .map(function(a) { return a.href; })
                .join('\n');
            if (urls) {
                copyText(urls).catch(function() {});
            }
        }
    }
});

function initUI() {
    if (typeof document.querySelector !== 'function') return;

    var mobileToggle = document.querySelector('.mobile-menu-toggle');
    if (mobileToggle) {
        mobileToggle.addEventListener('click', function() {
            toggleMobileMenu(mobileToggle);
        });
    }

    var navBtns = document.querySelectorAll('.nav-btn');
    for (var i = 0; i < navBtns.length; i++) {
        (function(btn) {
            btn.addEventListener('click', function() {
                navigate(btn.getAttribute('data-nav'), btn);
            });
        })(navBtns[i]);
    }

    var exampleTriggers = document.querySelectorAll('.example-trigger');
    for (var j = 0; j < exampleTriggers.length; j++) {
        (function(btn) {
            btn.addEventListener('click', function() {
                askQuestion(btn.getAttribute('data-question'));
            });
        })(exampleTriggers[j]);
    }

    var form = document.querySelector('#input-area form');
    if (form) {
        form.addEventListener('submit', handleSubmit);
    }
}

initUI();

