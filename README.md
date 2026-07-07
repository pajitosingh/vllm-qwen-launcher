# Qwen36 vLLM Toolkit — Launcher + Templates, Optimized for 2×24 GB

A portable vLLM launch script and **significantly improved chat template** for
**Qwen 3.6 27B** (also supports Qwen 3 / 3.5). Battle-tested on **dual 24 GB GPUs
(RTX 3090/4090, 48 GB total)** — the sweet spot for FP8, AWQ, or GPTQ quantized
models with tensor parallelism. Ships with the flagship `qwen36-chat-Pajito-optimized.jinja`
template — now strictly better than froggeric v21 on all dimensions.

## What This Repo Provides

### 1. Optimized Chat Template (`templates/`)

#### `qwen36-chat-Pajito-optimized.jinja` (recommended)

The result of iterative refinement across multiple community templates for Qwen 3.6 27B.
Built on `froggeric v20` and merged with improvements from `allanchan339` and `Kim`
variants. Has three unique features no community template matches — even froggeric v21
(July 2026) hasn't caught up on these.

### Pajito vs Community Templates

| Feature | Pajito (v5) | froggeric v21 | allanchan339 |
|---------|:---------:|:------------:|:------------:|
| **`ensure_ascii=False`** (100% cache hit) | ✅ | ❌ — bare `tojson` | ✅ (tools) |
| **Self-healing unclosed `<think>`** | ✅ (inherited) | ❌ | ✅ (origin) |
| **JSON string un-packing** (`from_json`) | ✅ (inherited) | ❌ | ✅ (origin) |
| **Tool error detection** (deterministic) | ✅ opt-in | ⚠️ heuristic | ❌ |
| **`auto_disable_thinking_with_tools`** | ✅ | ❌ | ❌ |
| **Thinking-OFF safe gen prompt** | ✅ | ⚠️ (v21.2 partial) | ❌ open `<think>` |
| **`preserve_thinking` kwarg** | `true` | `true` (v21.1) | N/A |
| **Empty-think suppression** | ✅ (froggeric v19) | ✅ | ✅ |
| **Quoted `</think>` bug fix** |  ✅  | ✅ (v21.1) | — |
| **Vision / tool dedup** | ✅ | ✅ | ✅ |

**Heritage:** Self-healing think + JSON un-packing originated from allanchan339.
Empty-think suppression from froggeric v19, quoted think-close detection from v21.3.
Our unique additions: deterministic tool error detection, `auto_disable_thinking_with_tools`,
and thinking-OFF safe generation prompt.

**Key improvements over stock and community templates:**

1. **100% Prefix-Cache Hit Rate**
   The template normalizes message boundaries so that identical system prompts and tool definitions produce byte-identical token sequences. This means vLLM's `--enable-prefix-caching` gets full reuse across multi-turn conversations and multi-user sessions, dramatically reducing TTFT (time-to-first-token).

2. **Interleaved Thinking (M2.5-style)**
   Thinking blocks (`<think>...</think>`) are preserved in conversation history only when they contain actual reasoning content. Empty think blocks are suppressed, preventing KV-cache pollution from wasted tokens — a known Qwen 3 issue where the model emits empty `<think></think>` wrappers.

3. **Self-Healing Unclosed Thinking Blocks**
   When the model generates `<think>` but emits a tool call before closing with `</think>`, the template automatically injects the closing tag. Without this, the model's subsequent outputs get parsed as thinking content and are invisible to the user.

4. **Robust Tool Call Parsing**
   - Handles both mapping-type and string-type tool arguments
   - **JSON String Un-Packing**: When tool arguments arrive as double-stringified JSON strings (common when models serialize arguments through certain APIs), the template parses them back into key-value parameter pairs
   - Configurable truncation of tool arguments (`max_tool_arg_chars`) and tool responses (`max_tool_response_chars`) for context-window safety

5. **Thinking Auto-Disable with Tools**
   When `auto_disable_thinking_with_tools=true` (passed via `--default-chat-template-kwargs`), the template suppresses thinking output when tools are present. This reduces latency for pure tool-calling turns while keeping reasoning available for normal chat.

6. **API-Toggleable Thinking**
   Thinking can be toggled per-request via the `enable_thinking` kwarg in `chat_template_kwargs`, without restarting the server. The template handles both enabled and disabled states gracefully.

7. **Vision Content Handling**
   Full multi-modal content rendering for images and videos with proper `<|vision_start|>` / `<|vision_end|>` tokens, vision ID counting, and strict validation (e.g., system messages cannot contain images).

8. **Tool Response Deduplication**
   Consecutive tool responses are merged into a single `<|im_start|>user` block, reducing token overhead in multi-step agentic workflows.

9. **Deterministic Tool Error Detection** (opt-in, v4)
   When `enable_tool_error_warnings=true`, the template detects tool call failures using constant string patterns (JSON `"error"` keys and Python traceback/exception prefixes). After 1 error, it suggests retry. After ≥2 consecutive errors, it warns the model its approach is failing. Fully cache-deterministic — same input always produces same output. Off by default for zero performance impact when not needed.

#### Comparison / Reference Templates

The `templates/` directory also includes the templates that `qwen36-chat-Pajito-optimized.jinja` was derived from, for reference and comparison:

| Template | Source | Description |
|---|---|---|
| `qwen36-chat-Pajito-optimized.jinja` | This repo | Final merged template — recommended |
| `frogerric_chat_template.jinja` | community (froggeric v20) | The base template q36 was built on; most thoroughly tested community variant |
| `Kim-qwen3.6_27b_merged.jinja` | community (Kim) | Merges froggeric + allanchan + official tokenizer additions; minja/C++ compatible |
| `allanchan339_qwen3.6-enhanced.jinja` | community (allanchan339) | Enhanced tool-calling instructions and argument handling |
| `qwen3.6-enhanced.jinja` | community | Base enhanced template |
| `qwen3.6_27b_improved.jinja` | community | Harness-agnostic variant (vLLM, LM Studio, llama.cpp) |

### 2. The Launch Script (`vllm_launch.sh`)

A battle-tested bash script that wraps `vllm serve` with:

- **Interactive model menu** — auto-discovers all model subdirectories in your `models/` folder and presents them with quantization type, approximate size, max context, and recommended-seqs
- **Quantization-aware profiles** — automatically detects AWQ (4-bit, 6-bit, BF16-INT4), GPTQ (4-bit, 8-bit), FP8, Marlin, and native unquantized models. Falls back to `config.json` inspection if the model name doesn't match known patterns
- **Interactive overrides** — override context length (`--max-model-len`), max concurrent sequences (`--max-num-seqs`), KV-cache quantization (`auto` / `fp8_e4m3` / `fp8_e5m2`), and speculative decoding token count at launch time
- **Stale process cleanup** — detects leftover vLLM/worker processes from crashed runs, terminates them gracefully (SIGTERM → SIGKILL), and releases GPU VRAM
- **Automatic GPU detection** — counts available GPUs via `nvidia-smi` to set tensor-parallel size automatically (override with `--tp N`)
- **NVIDIA .so injection** — automatically adds all CUDA library `.so` directories from your venv to `LD_LIBRARY_PATH` (fixes the common "libcudnn.so not found" class of errors)
- **LMCache support** — optional CPU offloading of KV-cache via the `--lmcache` flag
- **Pluggable chat template** — defaults to the bundled `qwen36-chat-Pajito-optimized.jinja`, but accepts any `.jinja` file via `--chat-template`
- **Non-interactive mode** — use `--yes` / `-y` with `--model NAME` to script it in automation
- **Dry-run mode** — use `--dry-run` to print the exact `vllm serve` command without launching

## Quick Start

### Prerequisites

- Python venv with `vllm` installed (tested with vLLM ≥ 0.6.0)
- NVIDIA GPU(s) with sufficient VRAM (see model requirements below)
- `nvidia-smi` available on PATH

### Installation

```bash
git clone <repo-url> vllm-qwen-launcher
cd vllm-qwen-launcher
chmod +x vllm_launch.sh
```

### Set Up Models Directory

The script looks for model subdirectories inside `./models/` by default. Each subdirectory should be a HuggingFace model (containing `config.json`, `.safetensors` files, etc.):

```bash
mkdir -p models
# Either copy or symlink your model directories:
ln -s /path/to/Qwen3-27B-AWQ models/Qwen3-27B-AWQ
ln -s /path/to/Qwen3-27B-GPTQ models/Qwen3-27B-GPTQ
```

### Run

```bash
# Interactive mode (recommended for first-time use)
./vllm_launch.sh --auto-venv

# Or specify your venv explicitly
./vllm_launch.sh --venv /path/to/your/venv/bin/activate

# Non-interactive with a specific model
./vllm_launch.sh --model Qwen3-27B-AWQ --venv /path/to/venv/bin/activate --yes

# Preview the command without launching
./vllm_launch.sh --dry-run --model Qwen3-27B-AWQ
```

### Venv Auto-Detection

Use `--auto-venv` to let the script search common locations:
- `./.venv/bin/activate` (relative to the script)
- `~/vllm_project/.venv/bin/activate`
- `~/.venv/bin/activate`
- Any `bin/activate` found under `$HOME` (max depth 5)

If no venv is found, the script attempts to use whatever `vllm` is on `PATH`.

## CLI Reference

```
./vllm_launch.sh [OPTIONS]

--models-dir DIR       Directory containing model subdirectories (default: ./models)
--model NAME           Specific model subdirectory to launch (skip menu)
--venv PATH            Path to Python venv activate script (default: ~/.venv/bin/activate)
--auto-venv           Auto-detect venv from common locations
--tp N                Tensor parallel size (default: auto-detect GPU count)
--gpu-ids IDS         CUDA_VISIBLE_DEVICES value (default: all GPUs)
--port PORT           vLLM server port (default: 8000)
--host HOST           vLLM server host (default: 0.0.0.0)
--gpu-mem-util FRAC   GPU memory utilization (default: 0.95)
--chat-template FILE  Path to chat template .jinja (default: ./templates/qwen36-chat-Pajito-optimized.jinja)
--served-name NAME    --served-model-name value (default: model dir name)
--max-seqs N          Override max-num-seqs for all models
--max-model-len N     Override max context length
--kv-dtype TYPE       KV-cache dtype: auto|fp8_e4m3|fp8_e5m2 (default: auto)
--spec-tokens N       Speculative decoding tokens (default: 3, 0=disabled)
--no-prefix-cache      Disable prefix caching
--no-chunked-prefill Disable chunked prefill
--enforce-eager       Disable torch.compile
--lmcache             Enable LMCache CPU offloading
--lmcache-config FILE LMCache config YAML
--cache-dir DIR       Cache directory (default: ~/.cache/vllm-launcher)
--yes / -y            Accept all defaults, run non-interactively
--dry-run             Print the vllm command without launching
--help / -h           Show help
```

## How the Script Works (Step by Step)

1. **Venv Activation** — Sources your Python venv and injects NVIDIA `.so` paths into `LD_LIBRARY_PATH` to avoid CUDA library errors. Uses `--auto-venv` or `--venv PATH`.
2. **Model Discovery** — Scans `--models-dir` (default: `./models/`) for subdirectories containing model files.
3. **Profile Detection** — For each model, uses name heuristics + `config.json` inspection to determine quantization, approximate VRAM usage, and default max context length. If the model name contains recognizable quant markers (e.g., "AWQ", "GPTQ", "FP8", "6bit"), it selects the appropriate profile. Otherwise, it parses `config.json` for `quant_config` entries.
4. **Interactive Menu** — Displays all models with quant label, approximate size, context length, and a star (★) for recommended variants. Remembers your last-launched model and offers to relaunch it via Enter.
5. **Override Prompts** — Interactively offers to override context length, max sequences, KV-cache quantization, and speculative token count. All of these can also be set via CLI flags or skipped with `--yes`.
6. **Environment Setup** — Exports vLLM, NCCL, PyTorch, and Triton cache-related env vars. These are tuned for stability (disabling brittle vLLM compile cache in favor of PyTorch's FX graph cache). All are overridable via existing env vars.
7. **Pre-flight Cleanup** — Finds and gracefully terminates any leftover vLLM processes from previous crashed runs before the new launch starts.
8. **Launch** — Runs `vllm serve` with the assembled arguments.
9. **Cooldown** — After vLLM exits, waits 5 seconds for natural worker shutdown, then SIGTERM/SIGKILLs any surviving workers. Reports final GPU memory state.

## VRAM Requirements (27B class models)

These approximations are for a 27B parameter model with different quantizations.
**This launcher is optimized for dual 24 GB GPUs (RTX 3090/4090, 48 GB total)** —
the most common enthusiast setup for running 27B-class models locally.

| Quantization | Approx. VRAM | 2×24 GB (3090/4090) | Notes |
|---|---|---|---|
| 4-bit AWQ / GPTQ | ~14-22 GB | ✅ Single GPU | Fits on one 24 GB card |
| 6-bit AWQ | ~28 GB | ✅ TP=2 | Comfortable on 2×24 GB |
| 8-bit GPTQ | ~34 GB | ✅ TP=2 | Comfortable on 2×24 GB |
| **FP8** | **~27 GB** | **✅ TP=2** | **Recommended sweet spot** |

**Dual 24 GB GPUs with FP8 is the recommended configuration.** FP8 offers near-lossless
quality vs. BF16 at roughly half the VRAM, fitting comfortably in 48 GB with room
for KV-cache and prefix caching.

### KV-Cache Quantization & Context Length

The KV-cache stores attention keys/values — its precision directly affects how much
context fits in VRAM:

| KV-Cache dtype | Max Context (48 GB, FP8 model) | Quality |
|----------------|-------------------------------|---------|
| `auto` / 16-bit (default) | ~148K tokens | Best — zero precision loss. Ideal for agentic coding. |
| `fp8_e4m3` / `fp8_e5m2` | ~262K tokens (full native) | Near-lossless. ~2× context capacity. |

The launcher defaults to **148K at 16-bit KV-cache** for maximum reasoning quality.
Switch to 8-bit KV-cache with `--kv-dtype fp8_e4m3` and manually set
`--max-model-len 262144` to unlock Qwen 3.6's full native context window on the
same hardware. The quality tradeoff is generally negligible in practice.

## Using the Chat Template Without the Launcher

The `qwen36-chat-Pajito-optimized.jinja` can be used with any vLLM installation. Simply point vLLM at it:

```bash
vllm serve /path/to/model \
    --chat-template /path/to/qwen36-chat-Pajito-optimized.jinja \
    --default-chat-template-kwargs '{"enable_thinking": true, "auto_disable_thinking_with_tools": true, "preserve_thinking": true}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

### Toggleable Template Kwargs

These are passed via `--default-chat-template-kwargs` and can be overridden per-request:

- `enable_thinking` (bool, default: `true`) — Enables `<think>` blocks in assistant responses
- `auto_disable_thinking_with_tools` (bool, default: `true`) — Suppresses thinking when tools are present (speeds up tool-calling turns)
- `preserve_thinking` (bool, default: `true`) — Preserves thinking content in conversation history (set to `false` to strip thinking from history for cache efficiency)
- `max_tool_arg_chars` (int, default: `0`) — Truncates tool arguments longer than N characters (0 = no truncation)
- `max_tool_response_chars` (int, default: `0`) — Truncates tool responses longer than N characters (0 = no truncation)
- `enable_tool_error_warnings` (bool, default: `false`) — Opt-in: detects tool call errors deterministically and injects warnings to help the model self-correct in agentic loops. After 1 error: suggests retry. After ≥2 consecutive errors: warns about failing approach. Detection uses constant string patterns (not heuristics) for cache safety.

## Qwen 3.6 Local Guide — Quick Reference

The companion guide **[`QWEN-LOCAL-GUIDE.md`](QWEN-LOCAL-GUIDE.md)** covers everything you need to know about running Qwen 3.6 27B locally: temperature settings, thinking modes, structured JSON output, agentic/tool-calling patterns, creative/roleplay use, and common pitfalls — all distilled from battle-tested production pipelines.

### Quick Parameter Cheat Sheet

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Temperature (structured)** | 0.3–0.4 | With `guided_json` (xgrammar); 0.6 without |
| **Temperature (creative)** | 0.8–0.9 | Sweet spot for narrative/roleplay |
| **Temperature (floor)** | **≥0.6** | MANDATORY — below this, thinking degrades |
| **Thinking mode** | **ON** (default) | Critical for proper function; disable only for latency-sensitive extraction |
| **`presence_penalty`** | **0.0** | MANDATORY — non-zero destroys JSON output |
| **`top_p`** | 0.90–0.95 | Standard range |
| **`top_k`** | 20 | Unsloth recommendation |
| **Structured output** | `guided_json` (xgrammar) | ~100% Pydantic validity vs. ~25% with `json_object` alone |
| **Max concurrency** | 2 | Degradation at concurrency=3 |
| **`max_tokens`** | ≥8192 | Minimum for structured output |
| **Tool call parser** | `hermes` or `qwen3_xml` | With `--enable-auto-tool-choice` |
| **Reasoning parser** | `qwen3` | Required for thinking mode |

### Architecture Note

Qwen 3.6 27B is **dense** (not MoE). Every parameter is active on every token. This
means logit-level constrained decoding (xgrammar/guidance) works effectively — the MoE
limitation documented in [arxiv 2606.09395](https://arxiv.org/abs/2606.09395) does NOT
apply. This is what enables the 100% JSON reliability benchmarks.

### The Most Common Mistakes

1. **Temperature < 0.6** — thinking quality collapses
2. **`presence_penalty=1.5`** — destroys JSON output (use 0.0)
3. **Bare `enable_thinking` in `extra_body`** — silently fails on vLLM; must nest under `chat_template_kwargs`
4. **`anyOf` in Pydantic schemas** — breaks outlines backend; flatten with sentinel booleans
5. **Not extracting from `reasoning_content`** — JSON may land there in some vLLM versions

See **[`QWEN-LOCAL-GUIDE.md`](QWEN-LOCAL-GUIDE.md)** for the full guide with code examples,
benchmark data, model comparisons, and community resources.

## vLLM Guide — Quick Reference

The companion guide **[`VLLM-GUIDE.md`](VLLM-GUIDE.md)** covers everything about running
vLLM in production: structured output configuration, performance tuning, tool calling
setup, and common deployment pitfalls.

### Quick vLLM Configuration Cheat Sheet

| Flag | Recommended Value | Why |
|------|------------------|-----|
| `--reasoning-parser` | `qwen3` | **Required** for Qwen 3.x thinking |
| `--structured-outputs-config.backend` | `xgrammar` | Explicit backend — don't rely on auto |
| `--structured-outputs-config.enable_in_reasoning` | `True` | Allow guided_json + thinking (v0.11.2+) |
| `--max-model-len` | 148000 (16-bit KV) / 262144 (8-bit KV) | [See KV-cache tradeoff](#kv-cache-quantization--context-length) |
| `--gpu-memory-utilization` | 0.85 | Leave 15% headroom |
| `--max-num-seqs` | 2 | Qwen 27B concurrency limit |
| `--enable-prefix-caching` | On | Massive TTFT reduction with stable prompts |
| `--enable-auto-tool-choice` | On (if using tools) | Required for tool calling |
| `--tool-call-parser` | `hermes` (stable) or `qwen3_xml` (newer) | Tool call format parser |
| `--speculative-config` | MTP-1 for latency-focused | 160 t/s boost on RTX 6000 |
| `--kv-cache-dtype` | `auto` (16-bit) / `fp8_e4m3` (8-bit) | 8-bit = ~2× context capacity |

### Structured Output Quick Reference

| Mechanism | API | Reliability (Qwen 3.6) |
|-----------|-----|------------------------|
| `structured_outputs` (v0.12+) | `extra_body` | **~100%** Pydantic validity |
| `guided_json` (legacy) | `extra_body` | **~100%** (deprecated, migrate) |
| `response_format: json_schema` | Top-level param | High (vLLM/OpenRouter) |
| `response_format: json_object` | Top-level param | ~25% — **do not rely on** |

### Backend Selection

```
xgrammar (default since v0.7.0)     ← RECOMMENDED: CFG-based, enum support, ~1% overhead
guidance (llguidance)               ← Alternative: near-zero compilation, competitive speed
outlines (deprecated)               ← AVOID: FSM-based, struggles with nested schemas
```

### The Most Common vLLM Mistakes

1. **Missing `--reasoning-parser qwen3`** — thinking mode silently broken
2. **`guided_json` without schema in prompt** — vLLM doesn't inject schema into chat template
3. **`gpu-memory-utilization` too high** — OOM on long contexts (0.85 max)
4. **Dynamic system prompts** — breaks prefix caching; keep stable across turns
5. **`json_object` alone for critical schemas** — 25% validity vs 100% with `guided_json`

See **[`VLLM-GUIDE.md`](VLLM-GUIDE.md)** for the full guide with backend deep-dive,
GPU memory math, tool calling configuration, and 18 documented pitfalls.

## Roadmap

This is an active project — we aim to compile benchmarks and tests as we go.
Key areas planned:

- **Benchmark suite** — Vision throughput at 4/8/16-bit KV-cache quantization,
  MTP speculative decoding token sweep (1–5), and NVLink vs PCIe GPU interconnect
  comparison for Qwen 3.6 27B specifically
- **Health & verification tooling** — Smoke test and runtime probe scripts for
  validating vLLM deployments
- **Edge-case template fixes** — Quoted `</think>` tag handling (froggeric v21.1)
  and reasoning bypass hardening (v21.2)

Contributions and benchmark results welcome.

## License

MIT — see [LICENSE](LICENSE).
