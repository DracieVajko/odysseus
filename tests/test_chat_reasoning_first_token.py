"""Reasoning-only SSE deltas must end the "waiting for first token" state.

A reasoning channel can arrive with an empty ``delta``. Harmony-format models
(gpt-oss) stream ``<|channel|>analysis<|message|>`` markers, which the backend
routes through ``_stream_delta_event(part, thinking=True)`` with no visible text
to attach, and servers that suppress reasoning send no text at all. When the
client gated its wait state on ``delta`` alone, the TTFT spinner kept cycling
"Still waiting for first token" / "Model pre-filling context" while the model
generated well over a thousand reasoning tokens.

``_providerEmittedOutput`` is the predicate that decision now hangs on, so this
test evaluates it directly against representative stream payloads instead of
asserting on source text: the bug is behavioural, and a source-string assertion
would pass again the moment someone reformatted the surrounding call.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

CHAT_JS = Path(__file__).resolve().parents[1] / "static" / "js" / "chat.js"

_HELPER_RE = re.compile(
    r"const\s+_providerEmittedOutput\s*=\s*\(json\)\s*=>\s*\((.*?)\);",
    re.DOTALL,
)


def _load_predicate():
    """Extract _providerEmittedOutput from chat.js and return a JS evaluator."""
    if not CHAT_JS.is_file():
        pytest.skip("static/js/chat.js not present")
    match = _HELPER_RE.search(CHAT_JS.read_text(encoding="utf-8"))
    assert match, "_providerEmittedOutput not found in chat.js"
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    return match.group(1), node


def _call_predicate(node, body, payloads):
    """Evaluate the predicate for each payload; return {payload_name: bool}."""
    cases = ",\n".join(
        f"  [{json.dumps(name)}, pred({json.dumps(payload)})]"
        for name, payload in payloads
    )
    script = (
        f"const pred = (json) => ({body});\n"
        f"const out = [\n{cases}\n];\n"
        "console.log(JSON.stringify(out));"
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return dict(json.loads(proc.stdout))


CASES = [
    # name, payload, expected
    ("plain_text", {"delta": "Ahoj"}, True),
    ("reasoning_only", {"delta": "", "thinking": True}, True),
    ("reasoning_no_delta_key", {"thinking": True}, True),
    ("harmony_marker", {"delta": "", "thinking": True}, True),
    ("empty", {"delta": ""}, False),
    ("thinking_false", {"delta": "", "thinking": False}, False),
    ("usage_event", {"type": "usage"}, False),
    ("tool_calls", {"type": "tool_calls", "calls": []}, False),
]


def test_predicate_treats_reasoning_as_provider_output():
    """Reasoning-only output counts; genuinely empty events do not."""
    body, node = _load_predicate()
    results = _call_predicate(node, body, [(c[0], c[1]) for c in CASES])

    for name, _payload, expected in CASES:
        assert results[name] is expected, f"{name}: expected {expected}, got {results[name]}"


def test_wait_state_gate_uses_the_predicate():
    """The stream loop must clear the wait timers via the predicate, not `json.delta`.

    A regression here is what produced the reported bug: `json.delta` is falsy
    for a reasoning-only event, so the gate never fired.
    """
    source = CHAT_JS.read_text(encoding="utf-8")
    assert "_providerEmittedOutput(json)" in source, (
        "stream loop no longer gates the wait state on _providerEmittedOutput"
    )
    # The stale gate would keep `json.delta` as the first disjunct.
    stale = re.search(
        r"if\s*\(\s*json\.delta\s*\|\|\s*json\.type\s*===\s*'final_response'",
        source,
    )
    assert stale is None, "wait-state gate still starts with bare `json.delta`"