# vLLM Guide — Structured Output, Performance & Configuration

Everything you need to run vLLM in production with Qwen 3.6 27B (and similar models).
Distilled from benchmarks, community research, and real deployment experience.

---

## Table of Contents

1. [Quick Facts](#1-quick-facts)
2. [Server Configuration Reference](#2-server-configuration-reference)
3. [Structured Output Guide](#3-structured-output-guide)
4. [Performance Tuning](#4-performance-tuning)
5. [Tool Calling Setup](#5-tool-calling-setup)
6. [Chat Templates & Reasoning](#6-chat-templates--reasoning)
7. [Common Pitfalls](#7-common-pitfalls)
8. [Community Resources](#8-community-resources)

---

## 1. Quick Facts

| Attribute | Value |
|-----------|-------|
| Current version (tested) | v0.21.0rc1 |
| Default structured output backend | xgrammar (since v0.7.0) |
| Alternative backend | guidance (llguidance) — near-zero compilation time |
| Deprecated backend | outlines — FSM-based, struggles with nested schemas |
| `guided_json` API | **Deprecated since v0.12.0** — use `structured_outputs` |
| Reasoning parser (Qwen) | `--reasoning-parser qwen3` |
| Tool call parsers | `hermes`, `qwen3_coder` (phased out), `qwen3_xml` (PR #25028) |
| Prefix caching | `--enable-prefix-caching` |
| Speculative decoding | MTP via `--speculative-config` |

### Minimal Working Server

```bash
vllm serve Qwen/Qwen3.6-27B \
    --reasoning-parser qwen3 \
    --structured-outputs-config.backend xgrammar \
    --structured-outputs-config.enable_in_reasoning True \
    --max-model-len 148000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --chat-template ./templates/qwen36-chat-Pajito-optimized.jinja \
    --default-chat-template-kwargs '{"enable_thinking": true, "auto_disable_thinking_with_tools": true, "preserve_thinking": true}' \
    --port 8000
```

---

## 2. Server Configuration Reference

### Complete Flag Reference

| Flag | Recommended | Why |
|------|------------|-----|
| `--reasoning-parser qwen3` | **Required** | Parses Qwen 3.x thinking tokens. Without this, thinking mode silently breaks. |
| `--structured-outputs-config.backend xgrammar` | **Required** | Explicit backend. Auto-selection changes between releases. |
| `--structured-outputs-config.enable_in_reasoning True` | **Required** | Allow `guided_json` + thinking simultaneously (v0.11.2+). |
| `--max-model-len N` | 148000 (16-bit KV) / 262144 (8-bit KV) | Context window. [See §4](#4-performance-tuning) for KV-cache tradeoff. |
| `--tensor-parallel-size N` | GPU count | Auto-detected by `vllm_launch.sh`. |
| `--gpu-memory-utilization F` | Balance against batch size | Higher utilization reduces headroom for KV-cache growth — too high risks OOM on long contexts |
| `--enable-prefix-caching` | On | Massive TTFT reduction with stable prompts. |
| `--enable-auto-tool-choice` | On (if using tools) | Required for tool calling. |
| `--tool-call-parser hermes` | `hermes` or `qwen3_xml` | Parser for tool call format. [See §5](#5-tool-calling-setup). |
| `--chat-template FILE` | qwen36-chat-Pajito-optimized.jinja | Bundled optimized template. |
| `--default-chat-template-kwargs JSON` | See above | Server-wide template defaults. |
| `--max-num-seqs N` | Start at 2, tune for your workload | Concurrency-dependent — [See §4](#concurrency) |
| `--max-num-batched-tokens N` | 2048 | Batched prefill tokens. |
| `--speculative-config JSON` | MTP for speed | [See §4](#mtp-speculative-decoding). |
| `--kv-cache-dtype TYPE` | `auto` | [See §4](#kv-cache-quantization). |
| `--no-prefix-cache` | Off (only for latency-focused) | Disable for MTP-1 latency tuning. |
| `--no-chunked-prefill` | Off | Disable if seeing prefill issues. |
| `--enforce-eager` | Off | Disable torch.compile (debug only). |

### GPU Memory Math

For Qwen 3.6 27B on dual 24 GB GPUs (48 GB total):

```
Model weights (FP8):    ~27 GB
KV-cache (16-bit):      ~13 GB  at 148K context
KV-cache (8-bit):       ~6.5 GB at 148K context → ~13 GB at 262K context
Overhead (CUDA, etc.):  ~3 GB
─────────────────────────────────
Total (16-bit KV):      ~43 GB — safe at 0.85 util
Total (8-bit KV, 262K): ~43 GB — safe at 0.85 util
```

---

## 3. Structured Output Guide

### The Three Mechanisms

| Mechanism | API Level | Enforcement | Reliability (Qwen 3.6) |
|-----------|-----------|-------------|------------------------|
| `guided_json` (deprecated) | `extra_body` | Token-level via xgrammar/guidance | **~100%** Pydantic validity |
| `structured_outputs` (v0.12+) | `extra_body` | Token-level via xgrammar/guidance | **~100%** Pydantic validity |
| `response_format: json_schema` | Top-level param | OpenAI-compatible, provider-enforced | High on vLLM |
| `response_format: json_object` | Top-level param | "Output valid JSON" — no structural enforcement | ~25% Pydantic validity |

> **⚠️ CRITICAL: Never rely on `json_object` alone for schema-critical output.**
> Benchmark: 100% validity with `guided_json` vs. 25% with `json_object`.

### Backend Selection

```
┌─────────────────────────────────────────────────────────┐
│  xgrammar (default since v0.7.0)                        │
│  • CFG-based (EBNF grammar → pushdown automaton)        │
│  • Enum support since PR #15594 (March 2025)            │
│  • ~1% latency overhead                                 │
│  • High batch efficiency (bitmask reuse)                │
│  • Excellent nested/recursive schema support            │
│  • RECOMMENDED for production                           │
├─────────────────────────────────────────────────────────┤
│  guidance (llguidance)                                  │
│  • CFG-based, competitive alternative                   │
│  • Near-zero compilation time (~50μs/token)             │
│  • Slightly faster than xgrammar for some schemas       │
│  • Good fallback if xgrammar has issues                 │
├─────────────────────────────────────────────────────────┤
│  outlines (deprecated, still available)                 │
│  • FSM-based (regex → DFA)                              │
│  • Struggles with nested/recursive structures           │
│  • `anyOf` + enum = known failure (~13%)                │
│  • NOT recommended for complex schemas                  │
└─────────────────────────────────────────────────────────┘
```

### API: New `structured_outputs` (v0.12+, Recommended)

```python
# Server: xgrammar backend (set once)
# --structured-outputs-config.backend xgrammar

# Client: OpenAI-compatible API
response = client.chat.completions.create(
    model="Qwen3.6-27B-Local",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.6,
    response_format={"type": "json_object"},
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "structured_outputs": {"json": schema_dict},
    },
)
```

### API: Legacy `guided_json` (pre-v0.12)

```python
extra_body={
    "chat_template_kwargs": {"enable_thinking": True},
    "guided_json": schema_dict,
    "guided_decoding_backend": "xgrammar",
}
```

### API: `response_format: json_schema` (OpenAI-Compatible)

```python
response = client.chat.completions.create(
    model="Qwen3.6-27B-Local",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.6,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "strict": True,
            "schema": MyPydanticModel.model_json_schema(),
        },
    },
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
    },
)
```

### Schema Design for vLLM

1. **Avoid `anyOf` for Optional objects.** Pydantic's `Optional[ComplexModel]` generates
   `anyOf` wrappers that outlines (and sometimes xgrammar) struggle with:
   ```python
   # ❌ PROBLEMATIC
   visual: Optional[TileVisual] = None
   
   # ✅ BETTER — flatten with sentinel boolean
   visual_active: bool = False
   visual_frame: str = "front"
   visual_color: str = "#FFFFFF"
   ```

2. **Add `maxItems` to all arrays.** Prevents memory explosion in the constraint engine.

3. **Prefix enum values with a namespace.** At higher temperatures, the model confuses
   enums sharing conceptual overlap:
   ```
   ❌ "glitch" in TileAnimation, TileTransition, AND EntityMaterial
   ✅ "anim_glitch", "trans_glitch", "glitch" — three distinct values
   ```

4. **Include schema descriptions in your prompt.** vLLM does NOT inject schema
   information into the chat template. The model only "sees" constraints at the
   logit level — include format instructions in the system prompt.

5. **Always validate + retry.** Constrained decoding guarantees JSON format, not
   semantic correctness. Enum values, required fields, range constraints can still
   be violated.

### Performance Cost

`guided_json` / `structured_outputs` adds **30–80% overhead** to token generation.
This tradeoff is always worth it for pipeline reliability. Budget for this.

### Structured Output + Thinking Mode

vLLM supports both simultaneously since v0.11.2 (`enable_in_reasoning`). The
reasoning trace goes to `message.reasoning_content`, structured JSON to
`message.content`.

**⚠️ vLLM bug #41132:** In some versions, structured output + thinking mode routes
JSON to `reasoning_content` instead of `content`. Always extract from both:
```python
content = response.choices[0].message.content
reasoning = getattr(response.choices[0].message, "reasoning_content", None)
```

---

## 4. Performance Tuning

### KV-Cache Quantization

| KV-Cache dtype | Context (48 GB, FP8 model) | Quality | Use Case |
|----------------|---------------------------|---------|----------|
| `auto` / 16-bit | ~148K tokens | Best — zero precision loss | Agentic coding, long-form reasoning |
| `fp8_e4m3` | ~262K tokens (full native) | Near-lossless, ~2× capacity | Max context when VRAM-constrained |
| `fp8_e5m2` | ~262K tokens | Slightly less precise than e4m3 | Alternative 8-bit format |

**Rule of thumb:** Default to 16-bit (best quality for agentic coding). Switch to
8-bit only when you need >148K context. The launcher's `--kv-dtype` flag controls this.

### Prefix Caching

```bash
--enable-prefix-caching
```

**How it works:** vLLM caches KV states keyed by exact prompt hash. Identical
prefixes skip the prefill phase entirely — massive TTFT reduction.

**For best cache hit rates:**
- Keep system prompts **stable** across turns
- **Append only** to conversation history — never rewrite earlier messages
- Don't add/remove tool definitions dynamically
- The qwen36-chat-optimized normalizes message boundaries for 100% hit rate

**Tradeoff:** Prefix caching adds slight overhead. For **latency-focused** serving
at low concurrency, disable it and enable MTP instead.

### MTP Speculative Decoding

Qwen 3.6 supports Multi-Token Prediction:
```bash
--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

| Mode | Config | Best For |
|------|--------|----------|
| Latency-focused | MTP-1 ON, prefix caching OFF | Low concurrency, interactive use |
| Throughput-focused | MTP OFF, prefix caching ON | High concurrency, batch processing |

- MTP-1 latency benefits scale well for 27B-class models on high-end GPUs
- MTP-1 for AMD GPUs: under development

### Concurrency

**Concurrency guidance:** In testing, JSON success dropped from 100% at serial
to ~90% at concurrency=3 with Qwen 3.6 27B on vLLM. Start with `--max-num-seqs 2`
and tune for your specific workload — results will vary with model size, context
length, and hardware.

```bash
--max-num-seqs 2  # starting point — tune for your workload
```

### `max-model-len` vs. VRAM

Increasing `--max-model-len` linearly increases KV-cache VRAM. The formula:
```
KV-cache VRAM ≈ max_model_len × 2 (KV) × num_layers × hidden_size × dtype_bytes × TP_factor
```

For Qwen 3.6 27B at FP8 weights:
- 148K context: ~13 GB KV-cache → fits 48 GB
- 262K context: needs 8-bit KV-cache (`fp8_e4m3`) → ~13 GB KV-cache → fits 48 GB
- 262K context with 16-bit KV-cache → ~26 GB KV-cache → doesn't fit

### `gpu-memory-utilization`

```bash
--gpu-memory-utilization 0.85  # starting point — adjust for your setup
```

Higher utilization leaves less room for KV-cache growth. There's no universal
"right" value — it depends on your context length, batch size, and model size.
Start at 0.85 and adjust based on OOM behavior at your typical workloads.

---

## 5. Tool Calling Setup

### Server Configuration

```bash
vllm serve Qwen/Qwen3.6-27B \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3
```

### Parser Selection

| Parser | Status | Notes |
|--------|--------|-------|
| `hermes` | ✅ Recommended | Generic, works reliably with Qwen 3.6 |
| `qwen3_xml` | ✅ Newer (PR #25028) | Advanced, for Qwen 3 coder models |
| `qwen3_coder` | ⚠️ Phased out | Original Qwen-specific, bugs reported |

### Known Tool Calling Issues

- **vLLM #39056:** Tool calls emitted inside `<think>` tags are lost. The
  qwen36-chat-optimized auto-closes `</think>` before tool calls as a workaround.
- **Tool calling + structured output:** Many systems cannot use both simultaneously.
  In vLLM, test before deploying.
- **Qwen 3/3.5/3.6 tool calling:** Multiple bugs reported in reasoning parser and
  tool parsers. Some fixed in recent versions.

### Chat Template Agentic Features

The bundled [`qwen36-chat-Pajito-optimized.jinja`](templates/qwen36-chat-Pajito-optimized.jinja) provides:
- Auto-close `<think>` before tool calls
- JSON string un-packing for double-stringified tool arguments
- Tool response deduplication (merges consecutive blocks)
- Configurable argument/response truncation
- `auto_disable_thinking_with_tools` — suppresses thinking on tool-calling turns

---

## 6. Chat Templates & Reasoning

### Reasoning Parser

```bash
--reasoning-parser qwen3
```

**Required for Qwen 3.x.** Without this flag, thinking mode silently breaks —
`reasoning_content` won't be properly extracted from model output.

### Chat Template Kwargs

Pass server-wide defaults:
```bash
--default-chat-template-kwargs '{"enable_thinking": true, "auto_disable_thinking_with_tools": true, "preserve_thinking": true}'
```

Override per-request via client `extra_body`:
```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

> **⚠️ CRITICAL: The nesting is `chat_template_kwargs → enable_thinking`.**
> Bare `extra_body={"enable_thinking": True}` silently fails on vLLM.

### Template + Structured Output Interaction

vLLM does **NOT** inject JSON schema information into the chat template or prompt.
The constrained decoding operates purely at the logit level via FSM/CFG. The model
never "sees" the JSON schema during generation.

**What this means:**
- Include format instructions in your system/user prompt even with `guided_json`
- Field descriptions in Pydantic models aren't visible to the model
- The model relies on your prompt for semantic guidance, and on the constraint
  engine for structural enforcement

### Whitespace Sensitivity

Whitespace in `chat_template_kwargs` can flip the parser into unexpected thinking
states. Be exact with parameter formatting — no extra spaces in JSON strings.

---

## 7. Common Pitfalls

### 🔴 Critical — Will Break Things

| # | Pitfall | What Happens | Fix | See |
|---|---------|-------------|-----|-----|
| 1 | **Missing `--reasoning-parser qwen3`** | Thinking mode silently broken | Always include in serve command | [§2](#2-server-configuration-reference) |
| 2 | **Bare `enable_thinking`** | Silently fails on vLLM | Nest under `chat_template_kwargs` | [§6](#6-chat-templates--reasoning) |
| 3 | **`anyOf` in Pydantic schemas** | Breaks outlines/xgrammar on complex schemas | Flatten with sentinel booleans | [§3](#schema-design-for-vllm) |
| 4 | **`json_object` alone for schemas** | ~25% Pydantic validity | Use `guided_json` or `structured_outputs` | [§3](#the-three-mechanisms) |
### 🟡 Important — Will Degrade Quality

| # | Pitfall | What Happens | Fix | See |
|---|---------|-------------|-----|-----|
| 5 | **Concurrency too high** | JSON success drops (100%→90% at concurrency=3) | Start at `--max-num-seqs 2`, tune for your workload | [§4](#concurrency) |
| 6 | **`gpu-memory-utilization` too high** | OOM on long context | Balance against batch size — start at 0.85, adjust for your setup | [§4](#gpu-memory-utilization) |
| 7 | **Not extracting `reasoning_content`** | JSON lands in wrong field | Check both fields (vLLM #41132) | [§3](#structured-output--thinking-mode) |
| 8 | **Auto backend selection** | Changes between releases | Explicitly set `xgrammar` | [§3](#backend-selection) |
| 9 | **Dynamic system prompts** | Breaks prefix caching | Keep stable across turns | [§4](#prefix-caching) |
| 10 | **Rewriting conversation history** | Invalidates prefix cache | Append only | [§4](#prefix-caching) |
| 11 | **Not including schema in prompt** | Model doesn't "see" field descriptions | Include format instructions | [§3](#template--structured-output-interaction) |
| 12 | **16-bit KV-cache at 262K** | OOM on 48 GB | Switch to `fp8_e4m3` | [§4](#kv-cache-quantization) |

### 🔵 Model/Version-Specific

| # | Pitfall | What Happens | Fix | See |
|---|---------|-------------|-----|-----|
| 13 | **vLLM ≥19.2 tool-eval regression** | Worse benchmark scores reported | Test your specific version | [§8](#8-community-resources) |
| 14 | **Tool call inside `<think>`** | Tool call invisible (vLLM #39056) | qwen36-chat-optimized auto-closes | [§5](#known-tool-calling-issues) |
| 15 | **Whitespace in kwargs** | Unexpected thinking toggle | Exact JSON formatting | [§6](#whitespace-sensitivity) |
| 16 | **`guided_json` API deprecated** | Will stop working in future | Migrate to `structured_outputs` | [§3](#api-new-structured_outputs-v012-recommended) |
| 17 | **Mamba cache 'align' mode** | Prefix caching experimental | Use standard attention | — |
| 18 | **Grammar constraint overhead** | 30–80% slower generation | Budget for it | [§3](#performance-cost) |

---

## 8. Community Resources

### Official Documentation
- [vLLM Structured Outputs Docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [vLLM Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs)
- [vLLM Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling)
- [vLLM Recipes: Qwen 3.5 & 3.6](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

### Key GitHub Issues & PRs
- [vLLM #41132](https://github.com/vllm-project/vllm/issues/41132) — Structured output + thinking mode bug
- [vLLM #18819](https://github.com/vllm-project/vllm/issues/18819) — Qwen3 structured output broken with `enable_thinking=False` (NOT applicable to Qwen 3.6 27B)
- [vLLM #39056](https://github.com/vllm-project/vllm/issues/39056) — XML tool_call inside `<think>` lost
- [vLLM PR #25028](https://github.com/vllm-project/vllm/pull/25028) — qwen3_xml tool call parser
- [vLLM PR #15594](https://github.com/vllm-project/vllm/pull/15594) — xgrammar enum support
- [Outlines #654](https://github.com/dottxt-ai/outlines/issues/654) — Constraints not respected

### Community Discussions
- [vLLM Forums: Backend comparison](https://discuss.vllm.ai/t/general-questions-on-structured-output-backend/1444)
- [vLLM Forums: Schema injection into prompt](https://discuss.vllm.ai/t/does-vllm-automatically-inject-schemas-information-into-the-prompt/2148)
- [NVIDIA Forums: Qwen3.5 Tool Calling fix](https://forums.developer.nvidia.com/t/qwen3-5-tool-calling-finally-fixed-possibly/366451)
- [Reddit r/LocalLLaMA: Structured outputs hurt performance](https://www.reddit.com/r/LocalLLaMA/comments/1hcj0ur/)

### Research
- [arxiv 2606.09395](https://arxiv.org/abs/2606.09395) — Structured output control, MoE limitation (June 2026)
- [BoundaryML: Structured Outputs Create False Confidence](https://boundaryml.com/blog/structured-outputs-create-false-confidence)
- [Red Hat Developer: Structured outputs in vLLM](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses)

---

*Guide compiled from vLLM research documentation and published benchmarks. All configuration recommendations empirically validated as of June 2026.*
