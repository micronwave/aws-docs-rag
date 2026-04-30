import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "frontend" / "index.html"


def _read_html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _extract_script() -> str:
    html = _read_html()
    match = re.search(r"<script>(.*)</script>", html, re.S)
    assert match is not None
    return match.group(1)


def _run_frontend_case(scenario: dict) -> dict:
    import tempfile

    script = _extract_script()
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as handle:
        handle.write(script)
        script_path = handle.name

    node_code = r"""
const fs = require('node:fs');
const vm = require('node:vm');

const scriptPath = process.argv[1];
const scenario = JSON.parse(process.argv[2]);
const script = fs.readFileSync(scriptPath, 'utf8');
const innerHTMLAssignments = [];
const removedNodes = [];

class Node {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.className = '';
    this.style = {};
    this.value = '';
    this.disabled = false;
    this.href = '';
    this.target = '';
    this.rel = '';
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this._text = '';
    this._html = '';
    this.parentNode = null;
    this._eventListeners = {};
  }
  appendChild(child) {
    this.children.push(child);
    child.parentNode = this;
    return child;
  }
  remove() {
    if (this.parentNode) {
      this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
    }
    removedNodes.push({
      tagName: this.tagName,
      className: this.className,
      textContent: this.textContent,
      innerHTML: this.innerHTML,
    });
  }
  focus() {}
  addEventListener(type, listener) {
    if (!this._eventListeners[type]) {
      this._eventListeners[type] = [];
    }
    this._eventListeners[type].push(listener);
  }
  closest(selector) {
    let node = this;
    while (node) {
      if (node.getAttribute && node.getAttribute('data-action') === selector.replace('[data-action=', '').replace(']', '')) {
        return node;
      }
      node = node.parentNode;
    }
    return null;
  }
  getAttribute(name) {
    return this[name] || null;
  }
  setAttribute(name, value) {
    this[name] = value;
  }
  querySelectorAll(selector) {
    return [];
  }
  querySelector(selector) {
    return null;
  }
  set textContent(value) { this._text = value; }
  get textContent() { return this._text || this.children.map((child) => child.textContent || '').join(''); }
  set innerHTML(value) {
    innerHTMLAssignments.push({ tagName: this.tagName, className: this.className, value });
    this._html = value;
  }
  get innerHTML() { return this._html; }
}

const nodes = {
  'chat-container': new Node('div'),
  'question-input': new Node('input'),
  'submit-btn': new Node('button'),
  'clear-chat-btn': new Node('button'),
  'welcome': new Node('div'),
};

const document = {
  getElementById(id) {
    return nodes[id];
  },
  createElement(tagName) {
    return new Node(tagName);
  },
  createTextNode(text) {
    const node = new Node('#text');
    node.textContent = text;
    return node;
  },
};

class AbortController {
  constructor() {
    this.signal = {};
  }
  abort() {
    this.signal.aborted = true;
  }
}

const timers = [];
function setTimeout(fn, _ms) {
  timers.push(fn);
  return timers.length;
}
function clearTimeout(_id) {}

const fetch = async (_url, _options) => {
  if (scenario.fetch.kind === 'resolve') {
    return {
      ok: scenario.fetch.ok,
      status: scenario.fetch.status,
      headers: { get: () => scenario.fetch.contentType },
      json: async () => scenario.fetch.jsonValue,
      text: async () => scenario.fetch.textValue,
    };
  }
  const error = new Error(scenario.fetch.message);
  error.name = scenario.fetch.name;
  throw error;
};

const context = {
  console,
  document,
  AbortController,
  setTimeout,
  clearTimeout,
  fetch,
  Event: class { constructor(type) { this.type = type; } preventDefault() {} },
  JSON,
  Array,
  Set,
  String,
  Number,
  Boolean,
  Date,
  RegExp,
  URL,
  Promise,
  Math,
  navigator: {
    clipboard: {
      writeText: function(text) {
        return Promise.resolve();
      }
    }
  },
  globalThis: null,
};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(script, context);

(async () => {
  nodes['question-input'].value = scenario.question;
  await context.handleSubmit({ preventDefault() {} });

  function serialize(node) {
    return {
      tagName: node.tagName,
      className: node.className,
      textContent: node.textContent,
      innerHTML: node.innerHTML,
      href: node.href,
      target: node.target,
      rel: node.rel,
      children: (node.children || []).map(serialize),
    };
  }

  process.stdout.write(JSON.stringify({
    chat: serialize(nodes['chat-container']),
    welcomeHidden: nodes['welcome'].style.display,
    submitDisabled: nodes['submit-btn'].disabled,
    innerHTMLAssignments,
    removedNodes,
  }));
})().catch((err) => {
  process.stderr.write(err.stack || String(err));
  process.exit(1);
});
"""

    try:
        completed = subprocess.run(
            ["node", "-e", node_code, script_path, json.dumps(scenario)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)
    finally:
        Path(script_path).unlink(missing_ok=True)


def test_successful_answer_renders_sources_and_clickable_links():
    duplicate_url = "https://docs.aws.amazon.com/s3"
    long_url = "https://docs.aws.amazon.com/lambda/latest/dg/configuration-vpc.html?example=long-source-link"
    result = _run_frontend_case(
        {
            "question": "What is S3?",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": {
                    "answer": "S3 stores objects.",
                    "sources": [
                        {"url": duplicate_url},
                        {"source_url": duplicate_url},
                        long_url,
                        {"url": "not-a-url"},
                    ],
                },
            },
        }
    )

    assistant_nodes = [child for child in result["chat"]["children"] if child["className"] == "message assistant"]
    assert len(assistant_nodes) == 1
    assistant = assistant_nodes[0]
    assert "S3 stores objects." in assistant["textContent"]
    assert "3 sources" in assistant["textContent"]
    sources = [child for child in assistant["children"] if child["className"] == "sources"]
    assert len(sources) == 1
    source_nodes = list(_walk_nodes(sources[0]))
    cards = [node for node in source_nodes if node["className"] == "source-card"]
    assert len(cards) == 3
    anchors = [node for node in source_nodes if node["tagName"] == "a"]
    assert [anchor["href"] for anchor in anchors] == [duplicate_url, long_url]
    assert all(anchor["target"] == "_blank" for anchor in anchors)
    assert all(anchor["rel"] == "noopener noreferrer" for anchor in anchors)

    source_links = [node for node in source_nodes if node["className"] == "source-link"]
    assert any(node["textContent"] == long_url for node in source_links)
    invalid_source = [node for node in source_links if node["textContent"] == "not-a-url"]
    assert len(invalid_source) == 1
    assert invalid_source[0]["tagName"] == "span"
    assert result["welcomeHidden"] == "none"
    assert result["submitDisabled"] is False


def _walk_nodes(node: dict):
    yield node
    for child in node.get("children", []):
        yield from _walk_nodes(child)


def test_source_count_updates_for_two_unique_valid_urls():
    result = _run_frontend_case(
        {
            "question": "Compare services.",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": {
                    "answer": "S3 and Lambda are different services.",
                    "sources": [
                        {"url": "https://docs.aws.amazon.com/s3"},
                        {"source_url": "https://docs.aws.amazon.com/lambda"},
                    ],
                },
            },
        }
    )

    assistant_nodes = [child for child in result["chat"]["children"] if child["className"] == "message assistant"]
    assert len(assistant_nodes) == 1
    assert "2 sources" in assistant_nodes[0]["textContent"]


def test_successful_answer_with_no_sources_still_renders_zero_source_count():
    result = _run_frontend_case(
        {
            "question": "No sources?",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": {
                    "answer": "The answer still renders.",
                    "sources": [],
                },
            },
        }
    )

    assistant_nodes = [child for child in result["chat"]["children"] if child["className"] == "message assistant"]
    assert len(assistant_nodes) == 1
    assert "The answer still renders." in assistant_nodes[0]["textContent"]
    assert "0 sources" in assistant_nodes[0]["textContent"]
    assert [child for child in assistant_nodes[0]["children"] if child["className"] == "sources"] == []


def test_successful_answer_renders_supported_markdown_subset():
    result = _run_frontend_case(
        {
            "question": "Show formats.",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": {
                    "answer": (
                        "# Heading\n\n"
                        "- bullet with `inline code`\n\n"
                        "1. step\n\n"
                        "| Service | Use |\n"
                        "|---|---|\n"
                        "| S3 | Object storage |\n\n"
                        "```\naws s3 ls\n```\n\n"
                        "Read https://docs.aws.amazon.com/s3"
                    ),
                    "sources": [],
                },
            },
        }
    )

    nodes = list(_walk_nodes(result["chat"]))
    tag_names = [node["tagName"] for node in nodes]
    assert "h3" in tag_names
    assert "ul" in tag_names
    assert "ol" in tag_names
    assert "table" in tag_names
    assert "pre" in tag_names
    assert "code" in tag_names

    answer_nodes = [node for node in nodes if "answer-rendered" in node["className"]]
    assert len(answer_nodes) == 1
    assert "# Heading" not in answer_nodes[0]["textContent"]
    assert "Heading" in answer_nodes[0]["textContent"]
    assert "bullet with inline code" in answer_nodes[0]["textContent"]
    assert "aws s3 ls" in answer_nodes[0]["textContent"]

    anchors = [node for node in nodes if node["tagName"] == "a"]
    assert len(anchors) == 1
    assert anchors[0]["href"] == "https://docs.aws.amazon.com/s3"
    assert anchors[0]["rel"] == "noopener noreferrer"


def test_answer_renderer_treats_raw_html_as_text_not_markup():
    result = _run_frontend_case(
        {
            "question": "Try html.",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": {
                    "answer": "<script>alert(1)</script>",
                    "sources": [],
                },
            },
        }
    )

    nodes = list(_walk_nodes(result["chat"]))
    answer_nodes = [node for node in nodes if "answer-rendered" in node["className"]]
    assert len(answer_nodes) == 1
    assert "<script>alert(1)</script>" in answer_nodes[0]["textContent"]
    assert "script" not in [node["tagName"] for node in nodes]
    assert all("<script>alert(1)</script>" not in item["value"] for item in result["innerHTMLAssignments"])


def test_server_error_displays_useful_message():
    result = _run_frontend_case(
        {
            "question": "What is S3?",
            "fetch": {
                "kind": "resolve",
                "ok": False,
                "status": 503,
                "contentType": "application/json",
                "jsonValue": {"error": "backend down"},
            },
        }
    )

    error_nodes = [child for child in result["chat"]["children"] if child["className"] == "message error"]
    assert len(error_nodes) == 1
    assert "Backend error (503): backend down" in error_nodes[0]["textContent"]


def test_invalid_json_response_is_handled():
    result = _run_frontend_case(
        {
            "question": "What is S3?",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": None,
            },
        }
    )

    error_nodes = [child for child in result["chat"]["children"] if child["className"] == "message error"]
    assert len(error_nodes) == 1
    assert "response was not valid JSON" in error_nodes[0]["textContent"]


def test_network_timeout_and_error_are_handled():
    timeout_result = _run_frontend_case(
        {
            "question": "What is S3?",
            "fetch": {
                "kind": "reject",
                "name": "AbortError",
                "message": "aborted",
            },
        }
    )

    timeout_errors = [child for child in timeout_result["chat"]["children"] if child["className"] == "message error"]
    assert "Request timed out after 30 seconds" in timeout_errors[0]["textContent"]

    network_result = _run_frontend_case(
        {
            "question": "What is S3?",
            "fetch": {
                "kind": "reject",
                "name": "TypeError",
                "message": "fetch failed",
            },
        }
    )

    network_errors = [child for child in network_result["chat"]["children"] if child["className"] == "message error"]
    assert "Network error: fetch failed" in network_errors[0]["textContent"]


def test_loading_state_uses_staged_labels_and_is_removed_after_success():
    result = _run_frontend_case(
        {
            "question": "What is S3?",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": {
                    "answer": "S3 stores objects.",
                    "sources": [],
                },
            },
        }
    )

    loading_nodes = [node for node in result["removedNodes"] if node["className"] == "message loading"]
    assert len(loading_nodes) == 1
    loading_html = loading_nodes[0]["innerHTML"]
    for label in ["Question", "Embed", "Search", "Generate", "Sources"]:
        assert label in loading_html
    assert "Processing RAG request" in loading_html
    assert all(child["className"] != "message loading" for child in result["chat"]["children"])


def test_initial_section_visibility_uses_active_section_model():
    html = _read_html()
    script = _extract_script()
    normalized = re.sub(r"\s+", "", html)

    assert '<divdata-section="ask"class="section-active">' in normalized
    assert '<divdata-section="about"class="section-active">' not in normalized
    assert '<divdata-section="architecture"class="section-active">' not in normalized
    assert "[data-section]{display:none;}" in normalized
    assert "[data-section].section-active{animation:sectionFade.2seaseforwards;}" in normalized
    assert '[data-section="about"],[data-section="architecture"]{display:block;}' not in normalized
    assert "section-active" in script
    assert "sections[i].style.display" not in script
    assert "target.style.display" not in script


def test_favicon_visible_letter_is_r():
    html = _read_html()
    favicon = re.search(r'<link rel="icon" href="([^"]+)">', html)

    assert favicon is not None
    payload = favicon.group(1)
    assert ">R</text>" in payload
    assert ">A</text>" not in payload


def test_destinations_include_page_kickers():
    html = _read_html()

    expected = {
        "ask": "Interactive Query",
        "about": "System Overview",
        "architecture": "Request Flow",
    }
    for section, kicker in expected.items():
        pattern = (
            r'<div data-section="' + re.escape(section) + r'"[^>]*>\s*'
            r'<div class="page-kicker">' + re.escape(kicker) + r"</div>"
        )
        assert re.search(pattern, html) is not None


def test_answer_actions_exist_with_sources():
    result = _run_frontend_case(
        {
            "question": "What is S3?",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": {
                    "answer": "S3 stores objects.",
                    "sources": [
                        {"url": "https://docs.aws.amazon.com/s3"},
                        {"url": "https://docs.aws.amazon.com/s3/latest/userguide/"},
                    ],
                },
            },
        }
    )

    assistant_nodes = [child for child in result["chat"]["children"] if child["className"] == "message assistant"]
    assert len(assistant_nodes) == 1
    assistant = assistant_nodes[0]
    
    # Verify answer-actions div exists
    answer_actions = [child for child in assistant["children"] if child["className"] == "answer-actions"]
    assert len(answer_actions) == 1, "answer-actions div should exist"
    
    # Verify both action buttons exist
    action_buttons = list(_walk_nodes(answer_actions[0]))
    copy_answer_buttons = [node for node in action_buttons if node.get("className") == "answer-action" and node.get("textContent") == "Copy answer"]
    copy_sources_buttons = [node for node in action_buttons if node.get("className") == "answer-action" and node.get("textContent") == "Copy sources"]
    assert len(copy_answer_buttons) >= 1, "Copy answer button should exist"
    assert len(copy_sources_buttons) >= 1, "Copy sources button should exist"


def test_answer_actions_exist_without_sources():
    result = _run_frontend_case(
        {
            "question": "What is S3?",
            "fetch": {
                "kind": "resolve",
                "ok": True,
                "status": 200,
                "contentType": "application/json",
                "jsonValue": {
                    "answer": "S3 stores objects.",
                    "sources": [],
                },
            },
        }
    )

    assistant_nodes = [child for child in result["chat"]["children"] if child["className"] == "message assistant"]
    assert len(assistant_nodes) == 1
    assistant = assistant_nodes[0]
    
    # Verify answer-actions div exists even without sources
    answer_actions = [child for child in assistant["children"] if child["className"] == "answer-actions"]
    assert len(answer_actions) == 1, "answer-actions div should exist even when no sources"
    
    # Verify both action buttons exist
    action_buttons = list(_walk_nodes(answer_actions[0]))
    copy_answer_buttons = [node for node in action_buttons if node.get("className") == "answer-action" and node.get("textContent") == "Copy answer"]
    copy_sources_buttons = [node for node in action_buttons if node.get("className") == "answer-action" and node.get("textContent") == "Copy sources"]
    assert len(copy_answer_buttons) >= 1, "Copy answer button should exist even without sources"
    assert len(copy_sources_buttons) >= 1, "Copy sources button should exist even without sources"


def test_static_template_invariants():
    html = _read_html()
    assert html.count("%%API_ENDPOINT%%") == 1
    assert len(re.findall(r"<script>", html)) == 1
    assert len(re.findall(r"</script>", html)) == 1
