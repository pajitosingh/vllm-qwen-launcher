# Qwen 3.6 Local Guide — Thinking, JSON, Agentic, Creative & Vision Use

A comprehensive guide to running Qwen 3.6 27B locally via vLLM, distilled from
production pipelines, benchmarks, and community research. Vision and temperature
guidance verified against official Qwen docs, HuggingFace model cards, vLLM
recipes, and community benchmarks (July 2026).

Full research sources at [`plans/RESEARCH-qwen-vision-thinking-claims.md`](plans/RESEARCH-qwen-vision-thinking-claims.md).

---

## Table of Contents

1. [Quick Facts](#1-quick-facts)
2. [Vision (Multimodal) Usage](#2-vision-multimodal-usage)
3. [Temperature Guide](#3-temperature-guide)
4. [Thinking Mode](#4-thinking-mode)
5. [Structured JSON Output](#5-structured-json-output)
6. [Agentic & Tool Calling](#6-agentic--tool-calling)
7. [Creative & Roleplay Use](#7-creative--roleplay-use)
8. [Chat Template Guide](#8-chat-template-guide)
9. [Common Pitfalls & Gotchas](#9-common-pitfalls--gotchas)
10. [Community Resources](#10-community-resources)

For vLLM server launch commands and client configuration, see
[`VLLM-GUIDE.md`](VLLM-GUIDE.md).

---

## 1. Quick Facts

Qwen 3.6 27B is a **dense** model (not MoE) — every parameter is active on every
token. This matters for structured decoding: logit-level constraint backends
(xgrammar, guidance) work effectively, unlike with MoE models where routing
happens before token masking. Licensed Apache 2.0. Max output is up to 80K
tokens (reliable to 64K).

### Recommended Hardware

**Dual 24 GB GPUs (RTX 3090/4090, 48 GB total)** is the sweet spot. This launcher
and all benchmarks were developed on this configuration:

- **FP8 (recommended):** ~27 GB model + ~13 GB KV-cache overhead = ~40 GB at full 262K context
- **AWQ 4-bit:** fits on a single 24 GB card (~17 GB) — great for budget setups
- **GPTQ 8-bit / AWQ 6-bit:** comfortable TP=2 on dual 24 GB (~28–34 GB)
- 48 GB total gives generous headroom for prefix caching, large batches, and long contexts

Single 24 GB GPU users: use AWQ 4-bit quantization. You'll still get excellent JSON
reliability and thinking quality — the 27B dense architecture shines at any quant.

### KV-Cache Quantization & Context Length

The KV-cache stores attention keys/values for all previous tokens. Its precision
directly affects how much context fits in VRAM:

| KV-Cache dtype | Context on 48 GB (FP8 model) | Quality | Best For |
|----------------|------------------------------|---------|----------|
| **auto / 16-bit** | ~148K tokens | Best — zero precision loss | Agentic coding, long-form reasoning |
| **fp8_e4m3 / fp8_e5m2** | ~262K tokens (full native) | Near-lossless — ~2× capacity | Max context when VRAM-constrained |

The launcher defaults to **148K at 16-bit KV-cache** for maximum quality in long-form
agentic coding. Switching to 8-bit KV-cache (`--kv-dtype fp8_e4m3`) roughly doubles
available context to Qwen 3.6's full native 262K — the tradeoff is a small precision
loss in cached attention values, generally unnoticeable in practice. This precision
loss may compound in long agentic coding sessions where cached attention values are
reused across many turns.

> **TODO:** The launcher script should auto-expand `--max-model-len` when 8-bit KV-cache
> is selected — to `min(model_max, 148K×2)` = 262K for Qwen 3.6 27B. Currently the user
> must manually set `--max-model-len 262144` alongside `--kv-dtype fp8_e4m3`.

**Context is ample, real workloads.** At 262K native, even long-running agentic
sessions with full conversation history stay well within the high-reliability zone.
For reference, a 20-turn game with full state context is ~76K tokens.

### At a Glance — Decision Tree

```
You need Qwen 3.6 27B to produce...

  JSON with a known schema?
    ├── YES → guided_json (xgrammar), temp=0.3–0.4, thinking ON  [§5]
    └── NO...

  Structured JSON but no schema?
    ├── YES → json_object, temp=0.6, thinking ON  [§5]
    └── NO...

  Creative narrative / roleplay?
    ├── YES → temp=0.8–0.9, thinking ON  [§7]
    └── NO...

  Classification / binary judgment?
    ├── YES → temp=0.6+ (see §3 for official tiers), thinking ON  [§3]
    └── NO...

  Tool calling / agentic workflow?
    └→ temp=0.6, thinking ON, qwen36-chat-optimized  [§6]

  Vision task (images, OCR, document parsing)?
    └→ See §2 for thinking ON vs OFF guidance by task type
```

### Minimal Working Example

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen3.6-27B-Local",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Respond in JSON."},
        {"role": "user", "content": "List three colors with hex codes."},
    ],
    temperature=0.6,           # Thinking-mode coding tier (see §3 for full guidance)
    max_tokens=4096,
    response_format={"type": "json_object"},
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},   # vLLM: MUST nest here
        "guided_json": {                                     # Add if you have a schema
            "type": "object",
            "properties": {
                "colors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "hex": {"type": "string"},
                        },
                        "required": ["name", "hex"],
                    },
                },
            },
            "required": ["colors"],
        },
        "guided_decoding_backend": "xgrammar",
    },
)

# Always extract from BOTH fields (vLLM #41132 workaround)
content = response.choices[0].message.content
reasoning = getattr(response.choices[0].message, "reasoning_content", None)

data = json.loads(content)
print(data)
```

> **The three most common mistakes in the example above:**
> 1. Forgetting `chat_template_kwargs` nesting — bare `enable_thinking` silently fails
> 2. Using `presence_penalty=1.5` — corrupts JSON structure
> 3. Not extracting from `reasoning_content` — JSON may land there instead of `content`
>
> **Note:** `guided_json` is deprecated in vLLM ≥0.12. See [§5](#5-structured-json-output)
> for migration to `structured_outputs`.

---

## 2. Vision (Multimodal) Usage

Qwen 3.6 27B includes a built-in vision encoder, making it a true multimodal model
capable of processing images and video alongside text. This section covers practical
vision usage, thinking-mode tradeoffs, and vLLM configuration.

### Thinking ON vs OFF for Vision — Task-Dependent

**There is no universal "best" setting for vision.** Head-to-head benchmarks on
Qwen3-VL-8B across 46 tasks (llm-stats.com) show:

| Metric | Thinking Wins | Instruct (Thinking OFF) Wins |
|--------|:---:|:---:|
| Total benchmarks | 30/46 | 14/46 |
| MMMU (multimodal reasoning) | **74.1** | 69.6 |
| MathVista | **81.4** | 77.2 |
| MMLU | **85.2** | 80.7 |
| OCRBench | 81.9 | **89.6** |
| DocVQA | 95.3 | **96.1** |
| ScreenSpot (GUI agent) | 93.6 | **94.4** |

**Practical rule:**

- **Thinking ON** for: STEM, math reasoning, spatial reasoning, medical/legal analysis,
  chart analysis, video understanding. The 2–4 point benchmark gain may offset the
  1.5–2× latency penalty.
- **Thinking OFF (Instruct)** for: OCR, document parsing, screen understanding,
  high-volume production pipelines. Faster (1.5–2×) and actually scores HIGHER on
  pure recognition tasks.

**Latency cost of thinking in vision:** ~1.5–2× wall-clock time and 2–3× more output
tokens compared to non-thinking mode.

### Known Gap — No Published A/B for Qwen3.6-27B Vision

No published A/B test exists comparing thinking ON vs OFF specifically for
**Qwen3.6-27B** on vision tasks. The benchmarks above are from Qwen3-VL-8B (a
dedicated VL model with separate Instruct/Thinking checkpoints). The HuggingFace
model card for Qwen3.6-27B runs all vision benchmarks in thinking mode only,
with no non-thinking vision numbers published.

**This is an open question worth benchmarking locally.** See [TODO.md](TODO.md) for
the deferred benchmark task.

### vLLM Vision Configuration

Key flags for vision workloads (from official vLLM recipe):

| Flag | Purpose |
|------|---------|
| `--limit-mm-per-prompt.video 0` | Image-only workloads — saves memory by disabling video slots |
| `--mm-encoder-tp-mode data` | Data-parallel vision encoder (better performance; encoder is small) |
| `--mm-processor-kwargs '{"max_pixels": 52144}'` | Resolution control (GitHub #25862) |
| `--language-model-only` | Skip vision encoder entirely for text-only workloads |
| `OMP_NUM_THREADS=1` | Prevents CPU contention when running multiple vLLM instances |

**`max_pixels` resolution control:** Controls the maximum number of pixels per image
processed by the vision encoder. Lower values = lower resolution but less memory and
fewer tokens. From vLLM GitHub #25862:
```bash
vllm serve Qwen/Qwen3.6-27B \
    --mm-processor-kwargs '{"max_pixels": 52144}'
```

### Image Format & Token Cost

Both JPEG and PNG are handled by the preprocessing pipeline. A full-page 390×844 JPEG
at default quality (~43KB) costs approximately **1.5k tokens** when read back by the
multimodal model. PNG is lossless and theoretically better for OCR/fine-grained text,
though no benchmarks confirm a practical difference.

### HTTP Transport Gotcha for Vision API Calls

**urllib times out on base64 payloads >500KB** for vision API calls. This affects
Python `urllib.request`-based HTTP clients when sending image data inline.

**Workaround:** Write the payload to a temp file, then pass to `curl -d @file` via
subprocess:

```python
import subprocess, tempfile, json

payload = json.dumps(request_body).encode()
with tempfile.NamedTemporaryFile(suffix=".json") as f:
    f.write(payload)
    f.flush()
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", "http://localhost:8000/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-d", f"@{f.name}"],
        capture_output=True, text=True, timeout=300
    )
```

See [§1 Quick Facts](#1-quick-facts) for architecture details, and the full research
report at [`plans/RESEARCH-qwen-vision-thinking-claims.md`](plans/RESEARCH-qwen-vision-thinking-claims.md).

---

## 3. Temperature Guide

### Official Temperature Recommendations

The official HuggingFace model card defines **three temperature tiers** depending on
operating mode. The earlier "0.6 floor" simplification is **partially correct** —
0.6 is specifically for precise coding in thinking mode, not a universal floor:

| Mode | Temperature | top_p | top_k | Source |
|------|:----------:|:-----:|:-----:|--------|
| Thinking (general) | **1.0** | 0.95 | 20 | HuggingFace model card |
| Thinking (precise coding) | **0.6** | 0.95 | 20 | HuggingFace model card |
| Non-thinking (Instruct) | **0.7** | 0.80 | 20 | HuggingFace model card |

### The Real Danger: Greedy Decoding

> **⚠️ Never use greedy decoding (temperature=0).** It causes endless repetitions
> in thinking mode — the official Qwen quickstart warns about this for all
> Qwen3-family models. The danger zone is temp=0 specifically; temperatures
> between 0.1–0.5 are fine when a grammar constraint enforces structure.

Community-confirmed:
- [lmstudio-community/Qwen3-32B-GGUF discussions](https://huggingface.co/lmstudio-community/Qwen3-32B-GGUF/discussions/1) — "thinking is never ending and propose same code again and again"
- [jan.ai Qwen3 settings guide](https://www.jan.ai/post/qwen3-settings) — "Greedy decoding breaks thinking mode — avoid it"

### Task-Specific Recommendations

| Use Case | Temperature | Notes |
|----------|------------|-------|
| Thinking mode — general tasks | **1.0** | Official recommendation for general thinking |
| Thinking mode — precise coding | **0.6** | Official recommendation for coding tasks |
| Non-thinking (Instruct) | **0.7** | Official recommendation for non-thinking mode |
| Structured JSON extraction | **0.3–0.4** | With `guided_json`/`structured_outputs` (xgrammar). Grammar enforces structure; lower temp is safe |
| Structured JSON extraction | **0.6** | Without `guided_json` (prompt-only enforcement). The 0.6 floor applies here |
| Mixed structured + creative | **0.6–0.8** | JSON schema enforces structure; temperature controls thought diversity |
| Creative narrative / brainstorming | **~0.9** | Sweet spot for creative tasks |
| Maximum creative diversity | **1.0** | Fully open-ended, maximum variety |
| Roleplay / character work | **0.8–0.95** | Enough heat for personality without incoherence |

### Key Nuances

- **With `guided_json`/`structured_outputs` (xgrammar):** 0.3–0.4 works well even
  below 0.6 — the grammar constraint enforces structure. The 0.6 floor applies when
  NOT using guided_json.
- **With reasoning ON + `json_schema`**, structured output stays solid even at higher
  temperatures. The schema constraint provides structural enforcement; temperature
  controls diversity of thought, not output structure.
- **xgrammar CFG enforcement** maintains 100% schema validity up to `temperature=0.95`
  (verified via `temperature_sweep_v2.py` benchmark on complex schemas).
- **Above 0.95**: intensity constraint (`ge=1`) is the first to break under
  temperature-driven randomness.
- **`presence_penalty=0.0` for structured JSON** — non-zero values (including the
  API default of 1.5) tend to corrupt JSON structure. Use 0.0 for any schema-constrained
  output.

### Other Sampling Parameters

| Parameter | Recommended | Notes |
|-----------|------------|-------|
| `top_p` | 0.80–0.95 | 0.95 for thinking, 0.80 for non-thinking (official) |
| `top_k` | 20 | Official recommendation across all modes |
| `repetition_penalty` | 1.0 | Default; raise only for specific repetition problems |
| `min_p` | 0.05 | For non-thinking mode |
| `presence_penalty` | **0.0** | Use 0.0 for structured JSON — the 1.5 default tends to corrupt structure |
| `frequency_penalty` | 0.0 | Default |

### Official Sampling Presets

**Thinking mode — general (official HF model card):**
```
temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0.0
```

**Thinking mode — precise coding (official HF model card):**
```
temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0
```

**Non-thinking mode (official HF model card):**
```
temperature=0.7, top_p=0.80, top_k=20, min_p=0.05, presence_penalty=0.0
```

> **Correction:** Earlier versions of this guide cited `presence_penalty=1.5` from
> Unsloth docs. This has been corrected to 0.0 based on web-verified findings —
> the official 1.5 default corrupts JSON structure (community-validated).

---

## 4. Thinking Mode

### The Rule: Thinking ON by Default

Qwen 3.6 was designed with thinking mode as the default. For structured extraction
tasks where you're using `json_schema` constraints, you may get cleaner output with
thinking off. See [§3](#3-temperature-guide) for the task-dependent tradeoffs.

### How to Configure (vLLM)

**Server-side flag:**
```bash
--reasoning-parser qwen3
```

**Client-side per-request (OpenAI-compatible API):**

```python
# Thinking ON (default, recommended)
response = client.chat.completions.create(
    model="Qwen3.6-27B-Local",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.6,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True}
    }
)

# Thinking OFF (only for simple extraction tasks)
response = client.chat.completions.create(
    model="Qwen3.6-27B-Local",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.3,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    }
)
```

### `chat_template_kwargs` Nesting

When calling vLLM's API, `enable_thinking` must be nested under
`chat_template_kwargs`:

```python
# CORRECT for vLLM Qwen:
extra_body={"chat_template_kwargs": {"enable_thinking": True}}

# WRONG — this is for raw transformers/Ollama, NOT vLLM:
extra_body={"enable_thinking": True}
```

The bare form silently fails on vLLM — the model appears to work but thinking is
never actually toggled.

### Server-Level Defaults

Disable thinking at the server level (all requests default to non-thinking):
```bash
--default-chat-template-kwargs '{"enable_thinking": false}'
```

Client `extra_body` overrides server defaults per-request.

### Thinking + Structured Output

**Thinking and `response_format` work together.** The reasoning trace goes to
`message.reasoning_content`, and the structured JSON goes to `message.content`.
No provider disables one for the other.

**Latency cost of thinking:** ~2–3× slower than non-thinking mode. For structured
extraction tasks where latency matters, turn thinking OFF (the schema constraint
provides enough structure).

**Critical: always extract from both fields:**
```python
content = message.content          # Where JSON lands with response_format
reasoning = getattr(message, "reasoning_content", None)  # Thinking trace
```

**vLLM bug workaround (GitHub #41132):** In some vLLM versions, structured output +
thinking mode may route JSON output to `reasoning_content` instead of `content`.
Check **both** the OpenAI SDK field (`reasoning_content`) and the raw JSON field
(`reasoning`) — depending on the vLLM version, the content may appear in either.

### Thinking Control via Prompt

The Qwen chat template also supports soft-switch keywords in user prompts:
- `/think` — enable thinking for this turn
- `/no_think` — disable thinking for this turn

These work alongside the `chat_template_kwargs` mechanism.

### "Preserved Thinking" (New in Qwen 3.6)

Qwen 3.6 supports preserving thinking content across conversation turns. In the
chat template, set `preserve_thinking: true` to keep reasoning traces in history.
This helps the model maintain coherent reasoning chains in multi-turn agentic
workflows.

---

## 5. Structured JSON Output

### The Three Mechanisms

| Mechanism | What It Does | Reliability | When to Use |
|-----------|-------------|-------------|-------------|
| `guided_json` / `structured_outputs` (vLLM) | Token-level constrained decoding via xgrammar/guidance | **~100%** Pydantic validity | Always, when available |
| `json_schema` (response_format) | OpenAI-compatible schema enforcement | High (vLLM) / N/A (DeepSeek API) | vLLM and OpenRouter |
| `json_object` (response_format) | "Output valid JSON" — no structural enforcement | ~25% Pydantic validity | Fallback only |

> **⚠️ `guided_json` is DEPRECATED in vLLM ≥0.12.** Use `structured_outputs` instead.
> Migration: `guided_json` → `{"structured_outputs": {"json": schema}}`
> The `guided_json` parameter still works (backward compatible) but will be removed
> in future releases.

### xgrammar vs Outlines Backend Selection

```bash
# xgrammar (default since v0.7.0, recommended)
--structured-outputs-config.backend xgrammar

# guidance (llguidance) — competitive alternative, near-zero compilation time
--structured-outputs-config.backend guidance
```

**Always explicitly set the backend.** Auto-selection changes between vLLM releases.

**xgrammar > outlines for nested schemas.** Outlines (FSM-based) fails on `anyOf` +
`enum` in nested structures (~13% failure rate). xgrammar (CFG-based) handles nesting
naturally. Prefer xgrammar for any schema with nested optional fields or complex
union types.

### Benchmark Results (Qwen 3.6 27B)

| Mode | Pydantic Validity | Notes |
|------|-------------------|-------|
| `guided_json` (xgrammar) | **100%** | Token-level constrained decoding |
| `json_object` only | ~25% | Enum drift, missing fields, schema mismatches |

**Verdict:** Never rely on `json_object` alone for critical schemas on Qwen.
Always use `guided_json`/`structured_outputs` or `response_format: json_schema`.

### Implementation: guided_json (Recommended — Pre-v0.12)

```python
from pydantic import BaseModel

class ActionResult(BaseModel):
    action: str
    success: bool
    narrative: str

schema_dict = ActionResult.model_json_schema()

response = client.chat.completions.create(
    model="Qwen3.6-27B-Local",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.6,
    response_format={"type": "json_object"},
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "guided_json": schema_dict,
        "guided_decoding_backend": "xgrammar",
    }
)
data = json.loads(response.choices[0].message.content)
```

### Implementation: structured_outputs (vLLM ≥0.12)

```python
response = client.chat.completions.create(
    model="Qwen3.6-27B-Local",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.6,
    response_format={"type": "json_object"},
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "structured_outputs": {"json": schema_dict},    # New API (vLLM ≥0.12)
        "guided_decoding_backend": "xgrammar",
    }
)
data = json.loads(response.choices[0].message.content)
```

### Implementation: json_schema (OpenAI-Compatible)

```python
response = client.chat.completions.create(
    model="Qwen3.6-27B-Local",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.6,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "action_result",
            "strict": True,
            "schema": ActionResult.model_json_schema(),
        },
    },
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
    }
)
```

### vLLM Issue #41132 — Structured Output + Thinking Routing

**Structured output + thinking mode may route JSON to `reasoning` field instead of
`content`.** This is a known vLLM bug (GitHub #41132). The JSON may appear in:
- `reasoning_content` (OpenAI SDK field)
- `reasoning` (raw JSON field)

Always check both fields alongside `content`:
```python
content = response.choices[0].message.content
reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
# Also check 'reasoning' in raw JSON if neither of the above has the JSON
```

### Schema Design Best Practices

1. **Avoid `anyOf` for Optional objects.** Pydantic's `Optional[ComplexModel]` generates
   `anyOf` wrappers that outlines/xgrammar struggle with. Use explicit sentinel booleans
   and flat fields instead:
   ```python
   # PROBLEMATIC:
   visual: Optional[TileVisual] = None
   # BETTER:
   visual_active: bool = False
   visual_frame: str = "front"
   visual_color: str = "#FFFFFF"
   ```

2. **Flatten deep nesting.** Promote nested optional fields with boolean sentinels.

3. **Add `maxItems` to all arrays.** Prevents memory explosion in the constraint engine.

4. **Include schema descriptions in the prompt.** vLLM does NOT inject schema information
   into the chat template. The model benefits from explicit format instructions in the
   system/user prompt, even with constrained decoding.

5. **Order fields logically.** Token-by-token generation follows field order.

6. **Always validate + retry.** Constrained decoding guarantees format, not correctness.

### Prefix Enum Values to Prevent Cross-Contamination

At higher temperatures (≥0.95), Qwen confuses enum values that share conceptual
overlap. The fix: **prefix all enum values with a namespace** so no value can be
valid in more than one enum.

```
❌ Before: "glitch" could be TileAnimation, TileTransition, or EntityMaterial
✅ After:  "anim_glitch", "trans_glitch", "glitch" — three distinct values
```

This makes cross-contamination impossible at the token level.

### Performance Cost of Grammar Constraints

`guided_json`/`structured_outputs` adds **30–80% overhead** to token generation speed
(community reports). This tradeoff is always worth it for pipeline reliability. Budget
for this in latency estimates.

### When to Use Two-Pass Approach

For creative tasks requiring both high-quality prose AND structured output:
1. **Pass 1:** Generate freely at temp 0.8–1.0, no constraints
2. **Pass 2:** Extract structured data from pass 1 at temp 0.2–0.3, with constraints

Constrained decoding demonstrably degrades content quality (BoundaryML benchmark:
93.63% free-form vs 91.37% constrained on BFCL). The two-pass approach preserves
quality while ensuring valid structure.

---

## 6. Agentic & Tool Calling

### Tool Calling Configuration

```bash
vllm serve Qwen/Qwen3.6-27B \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3
```

**Parser selection:**
- `hermes` — Generic, works with Qwen 3.6
- `qwen3_coder` — Original Qwen-specific parser (being phased out)
- `qwen3_xml` — Newer, more advanced (PR #25028), recommended for Qwen 3 coder models

### Chat Template Features for Agentic Work

The [`qwen36-chat-Pajito-optimized.jinja`](templates/qwen36-chat-Pajito-optimized.jinja) (bundled with this repo)
provides several agentic enhancements:

- **Auto-close thinking blocks before tool calls** — prevents the model from emitting
  tool calls inside unclosed thinking blocks (which would make the tool call invisible)
- **JSON string un-packing** — handles double-stringified tool arguments
- **Tool response deduplication** — merges consecutive tool responses into a single block
- **Configurable argument/response truncation** for context-window safety
- **`auto_disable_thinking_with_tools`** — suppresses thinking on tool-calling turns
  for lower latency

### Multi-Turn Agentic Patterns

**`reasoning_content` round-trip is NOT required for Qwen** (unlike DeepSeek).
The vLLM chat template handles thinking content in history transparently.

**Preserved thinking** (`preserve_thinking: true`) keeps reasoning traces in
conversation history, helping the model maintain coherent reasoning chains.

### Concurrency & Performance

**Concurrency guidance:** In testing with local Qwen 3.6 27B on vLLM, JSON
success was 100% at serial, dropping to ~90% at concurrency=3. Start with
2 parallel calls and tune for your specific workload, hardware, and context length.

**`max_tokens`** should have sufficient headroom for structured output tasks —
truncated mid-JSON means total response loss. No hard minimum; 8192 is a
reasonable starting point for most schemas, but complex outputs may need more.
The model supports up to 80K (reliable to 64K).

### Max Tokens for Agentic Output

- **Starting point: 8192** for typical structured output tasks (no hard minimum — context-dependent)
- **Up to 24K** for large data outputs
- **Reliable ceiling: 64K** (model supports up to 80K)

### MTP Speculative Decoding (Speed Boost)

Qwen 3.6 supports Multi-Token Prediction for faster generation:
```bash
--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

- For latency-focused serving at low concurrency: enable MTP-1, disable prefix caching
- For throughput-focused serving: disable MTP, enable prefix caching

### Prefix Caching for Multi-Turn Agents

```bash
--enable-prefix-caching
```

The `qwen36-chat-Pajito-optimized.jinja` normalizes message boundaries so identical system
prompts and tool definitions produce byte-identical token sequences. This gives
**100% prefix-cache hit rate** across multi-turn conversations and multi-user
sessions, dramatically reducing TTFT.

**Cache-busters to avoid:**
- Dynamic system prompts (changing per-turn)
- Naive context compaction that rewrites earlier messages
- Adding tool definitions dynamically

---

## 7. Creative & Roleplay Use

### Temperature for Creative Work

- **0.8–0.9** — Sweet spot for narrative generation, brainstorming, creative writing
- **0.9–0.95** — Roleplay with personality and variety
- **1.0** — Maximum diversity; may lose coherence

With thinking ON and `json_schema` enabled, creative output stays structurally
valid at higher temperatures.

### Thinking Mode for Creative Tasks

**Keep thinking ON for creative work.** The chain-of-thought deliberation helps
with:
- Mood and tone consistency
- Character voice maintenance
- Callback jokes and narrative coherence
- Long-range narrative planning

### Two-Pass for Creative + Structured

For use cases that need both creative quality AND structured output (e.g., an AI
Dungeon Master that outputs narrative in a JSON schema):

1. **Pass 1 (creative):** temp 0.9, thinking ON, no constraints → high-quality prose
2. **Pass 2 (structural):** temp 0.3, thinking OFF, `guided_json` → extract into schema

### Prompt Design Patterns

1. **One schema shape per call.** Never ask Qwen to produce multiple payload structures
   in one JSON response. Mid-size models lose track.

2. **Show, don't just tell.** Every non-trivial payload shape needs a concrete JSON
   example in the prompt. Written specs alone are ambiguous.

3. **Stay within output token budget.** For `max_tokens=4096`, target ≤12–15 objects.
   For `max_tokens=8192`, target ≤25–30. Truncated mid-JSON = total response loss.

4. **Normalize before validating.** Models frequently return `"Appropriate"` instead
   of `"appropriate"`. Normalize case, whitespace, and separators before Pydantic
   validation.

5. **Salvage valid entries.** Validate response entries individually. A batch of 15
   where entry #7 is bad should yield 14 valid entries, not 0.

### Character Voice & Consistency

Qwen 3.6 27B maintains good character consistency across multi-turn interactions
when:
- Thinking is ON (helps with voice maintenance)
- Temperature is 0.8–0.95 (enough heat for personality, not enough to break character)
- System prompt defines voice clearly with concrete examples
- Conversation history preserves thinking traces (`preserve_thinking: true`)

---

## 8. Chat Template Guide

### The qwen36-chat-Pajito-optimized.jinja (Flagship)

Bundled with this repo at [`templates/qwen36-chat-Pajito-optimized.jinja`](templates/qwen36-chat-Pajito-optimized.jinja).
Built on froggeric v20, merged with improvements from allanchan339 and Kim variants.

**Key features:**
- 100% prefix-cache hit rate (normalized message boundaries)
- Empty-think poisoning fix (suppresses empty thinking wrappers)
- Self-healing unclosed thinking blocks (newline-anchored detection, auto-injects closing tag before tool calls)
- JSON string un-packing for double-stringified tool arguments
- Quoted think-close bug fix (newline-anchored detection, adopted from froggeric v21.3)
- API-toggleable thinking via `chat_template_kwargs`
- Tool response deduplication
- Full vision content handling

### Toggleable Template Kwargs

Pass via `--default-chat-template-kwargs` or per-request `extra_body`:

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_thinking` | bool | `true` | Enable thinking blocks |
| `auto_disable_thinking_with_tools` | bool | `true` | Suppress thinking when tools present |
| `preserve_thinking` | bool | `true` | Keep thinking traces in history |
| `max_tool_arg_chars` | int | `0` | Truncate tool args (0=no limit) |
| `max_tool_response_chars` | int | `0` | Truncate tool responses (0=no limit) |

### Template Lineage

```
Official Qwen 3.6 tokenizer template
    ├── froggeric v19 (abolished "Empty Think" poisoning)
    ├── froggeric v20 "The Architect Patch" (structural overhaul, agentic loop fix)
    │       ├── allanchan339 enhanced (tool-calling fixes, string→JSON arg parsing)
    │       └── Kim merged (froggeric + allanchan + official additions, minja/C++ compat)
    └── qwen36-chat-Pajito-optimized.jinja ← THIS REPO'S FLAGSHIP (froggeric v20 + allanchan + Kim merges)
```

### Community Template Contributors

- **froggeric** ([HF: froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates))
  - v19 abolished "Empty Think" poisoning (empty thinking wrappers polluting KV-cache)
  - v20 "The Architect Patch" — deep agentic loop fixes, C++ inference engine compatibility
  - v21.3 — newline-anchored think-close detection (adopted in this repo's flagship template)
- **allanchan339** ([GitHub: allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix))
  - Enhanced tool-calling, JSON argument string→dict parsing
  - Maintains **separate templates for Qwen 3.5 vs 3.6** — templates are NOT interchangeable
  - ⚠️ **Warning:** Do NOT use Qwopus3.5-AWQ templates — format drift after 65K tokens
- **Kim** — Merged froggeric + allanchan with official tokenizer additions (minja/C++ compat)

> **⚠️ Qwen 3.5 vs 3.6 templates are NOT interchangeable.** Use version-specific
> templates. Mixing 3.5 templates with 3.6 models causes silent formatting errors.

---

## 9. Common Pitfalls & Gotchas

### 🔴 Critical — Will Break Things

| # | Pitfall | What Happens | Fix | See |
|---|---------|-------------|-----|-----|
| 1 | **Greedy decoding (temp=0)** | Endless repetitions, performance degradation — official warning | Use official tiers: 1.0 (general thinking), 0.6 (precise coding), 0.7 (non-thinking). With `guided_json`/xgrammar, 0.3–0.4 is also safe | [§3](#3-temperature-guide) |
| 2 | **`presence_penalty=1.5`** | Corrupts JSON structure — the API default is wrong for structured tasks | Use 0.0 for structured JSON | [§3](#3-temperature-guide) |
| 3 | **Bare `enable_thinking`** | Silently fails on vLLM | Nest under `chat_template_kwargs` | [§4](#4-thinking-mode) |
| 4 | **`anyOf` in Pydantic schemas** | Breaks outlines backend | Flatten with sentinel booleans, use xgrammar | [§5](#5-structured-json-output) |
| 5 | **Missing `--reasoning-parser qwen3`** | Thinking mode won't work | Always include in serve command — see [`VLLM-GUIDE.md`](VLLM-GUIDE.md) | [§4](#4-thinking-mode) |

### 🟡 Important — Will Degrade Quality

| # | Pitfall | What Happens | Fix | See |
|---|---------|-------------|-----|-----|
| 6 | **Concurrency too high** | JSON success drops 100%→90% at concurrency=3 | Start at 2 parallel calls, tune for your workload | [§6](#6-agentic--tool-calling) |
| 7 | **Not extracting `reasoning_content`** | JSON may land in wrong field — check `content`, `reasoning_content`, AND `reasoning` | Check all three fields (vLLM #41132) | [§4](#4-thinking-mode) |
| 8 | **Relying on `json_object` alone** | ~25% vs 100% validity with guided | Use `guided_json`/`structured_outputs` for critical schemas | [§5](#5-structured-json-output) |
| 9 | **Auto backend selection** | Changes between vLLM releases | Explicitly set `xgrammar` or `guidance` | [§5](#5-structured-json-output) |
| 10 | **Dynamic system prompts** | Breaks prefix caching | Keep system prompts stable | [§6](#6-agentic--tool-calling) |
| 11 | **Rewriting conversation history** | Invalidates prefix cache | Append only, never rewrite | [§6](#6-agentic--tool-calling) |
| 12 | **`max_tokens` too low** | Output truncated mid-structure — total response loss | Allow enough headroom (8192 is a reasonable starting point; complex schemas may need more) | [§6](#6-agentic--tool-calling) |
| 13 | **Schema not in prompt** | Model doesn't "see" field descriptions | Include format instructions in prompt | [§5](#5-structured-json-output) |

### 🔵 Model-Specific

| # | Pitfall | What Happens | Fix | See |
|---|---------|-------------|-----|-----|
| 14 | **3.6 vs 3.5 template mix-up** | Templates NOT interchangeable | Use version-specific templates | [§8](#8-chat-template-guide) |
| 15 | **Empty thinking blocks** | Pollutes KV-cache | qwen36-chat-optimized auto-suppresses | [§8](#8-chat-template-guide) |
| 16 | **Tool calls in thinking** | Tool call invisible to parser | qwen36-chat-optimized auto-closes thinking | [§8](#8-chat-template-guide) |
| 17 | **Whitespace in kwargs** | Unexpectedly toggles thinking | Be exact with parameter format | [§4](#4-thinking-mode) |

### ⚪ Infrastructure

| # | Pitfall | What Happens | Fix |
|---|---------|-------------|-----|
| 18 | **urllib timeout on vision payloads** | Base64 payloads >500KB cause HTTP timeout | Write to temp file, use `curl -d @file` via subprocess (see [§2](#2-vision-multimodal-usage)) |

---

## 10. Community Resources

### Official
- [Qwen 3.6 27B Model Card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen vLLM Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [Qwen Quickstart (official)](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)
- [vLLM Structured Outputs Docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [vLLM Recipes: Qwen 3.5 & 3.6](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM Qwen3.6-27B Recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- [SGLang Qwen3.6 Docs](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6)
- [Unsloth Qwen 3.6 Settings](https://unsloth.ai/docs/models/qwen3.6)
- [Alibaba Cloud Qwen Structured Output](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)

### Community Templates
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — v19/v20/v21.3
- [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)
- [Merged community template (fakezeta)](https://gist.github.com/fakezeta/9e8e039c60332fcb143c6e805558afe0)

### Benchmarks & Stats
- [llm-stats.com: Qwen3-VL-8B Instruct vs Thinking (46 benchmarks)](https://llm-stats.com/models/compare/qwen3-vl-8b-instruct-vs-qwen3-vl-8b-thinking)
- [club-3090 (noonghunna)](https://github.com/noonghunna/club-3090) — vLLM throughput benchmarks for Qwen-class models on 3090/4090

### Key GitHub Issues
- [vLLM #41132](https://github.com/vllm-project/vllm/issues/41132) — Structured output + thinking mode bug (JSON routed to `reasoning` field)
- [vLLM #18819](https://github.com/vllm-project/vllm/issues/18819) — Qwen3 structured output broken with `enable_thinking=False` (NOT applicable to Qwen 3.6 27B)
- [vLLM #39056](https://github.com/vllm-project/vllm/issues/39056) — XML tool_call inside thinking lost
- [vLLM PR #25028](https://github.com/vllm-project/vllm/pull/25028) — qwen3_xml tool call parser
- [vLLM #25862](https://github.com/vllm-project/vllm/issues/25862) — max_pixels usage example

### Research & Benchmarks
- [arxiv 2606.09395](https://arxiv.org/abs/2606.09395) — Empirical Study for Structured Output Control (MoE limitation finding, June 2026)
- [BoundaryML: Structured Outputs Create False Confidence](https://boundaryml.com/blog/structured-outputs-create-false-confidence)
- [Tianpan.co: JSON Mode Won't Save You](https://tianpan.co/blog/2026-04-09-structured-output-failures-production-llm)
- [Full research report: plans/RESEARCH-qwen-vision-thinking-claims.md](plans/RESEARCH-qwen-vision-thinking-claims.md)
