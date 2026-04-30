import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "frontend" / "index.html"


def _extract_script() -> str:
    html = HTML_PATH.read_text(encoding="utf-8")
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
  }
  focus() {}
  set textContent(value) { this._text = value; }
  get textContent() { return this._text || this.children.map((child) => child.textContent || '').join(''); }
  set innerHTML(value) { this._html = value; }
  get innerHTML() { return this._html; }
}

const nodes = {
  'chat-container': new Node('div'),
  'question-input': new Node('input'),
  'submit-btn': new Node('button'),
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
  Promise,
  Math,
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
                        {"source_url": "https://docs.aws.amazon.com/s3"},
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
    sources = [child for child in assistant["children"] if child["className"] == "sources"]
    assert len(sources) == 1
    anchors = [child for child in sources[0]["children"] if child["tagName"] == "a"]
    assert len(anchors) == 1
    assert anchors[0]["href"] == "https://docs.aws.amazon.com/s3"
    assert anchors[0]["target"] == "_blank"
    assert result["welcomeHidden"] == "none"
    assert result["submitDisabled"] is False


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

