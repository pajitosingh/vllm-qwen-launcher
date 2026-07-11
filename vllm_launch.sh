#!/bin/bash

# =============================================================================
# Qwen36 vLLM Toolkit — Launcher + Templates, Optimized for 2×24 GB
# Qwen 3.x model launcher
# A generalized, portable vLLM launch script with:
#   • Interactive model-selection menu (auto-discovers model subdirectories)
#   • Quantization-aware profile detection (AWQ, GPTQ, FP8, Marlin, native)
#   • Interactive overrides for context length, max-seqs, KV-cache dtype,
#     and speculative decoding tokens
#   • Stale-process cleanup (pre-flight + post-exit cooldown)
#   • Pluggable chat template (defaults to the bundled qwen36-chat-Pajito-optimized.jinja)
#
# USAGE:
#   ./vllm_launch.sh [OPTIONS]
#
# EXAMPLES:
#   # Fully interactive (prompts for everything)
#   ./vllm_launch.sh
#
#   # Point at a specific models directory, single GPU, custom port
#   ./vllm_launch.sh --models-dir /data/models --tp 1 --port 8000
#
#   # Non-interactive: pick model by name, accept all defaults
#   ./vllm_launch.sh --model Qwen3-27B-AWQ --yes
#
#   # Custom venv and chat template
#   ./vllm_launch.sh --venv /opt/vllm/.venv --chat-template ./my-template.jinja
#
#   # Let the script auto-find the venv in common locations
#   ./vllm_launch.sh --auto-venv
#
# OPTIONS:
#   --models-dir DIR     Directory containing model subdirectories
#                         (default: ./models)
#   --model NAME          Specific model subdirectory to launch (skip menu)
#   --venv PATH           Path to Python venv activate script
#                         (default: $HOME/.venv/bin/activate)
#   --auto-venv           Auto-detect venv from common locations
#   --tp N                Tensor parallel size (default: auto-detect GPU count)
#   --gpu-ids IDS         CUDA_VISIBLE_DEVICES value (default: all GPUs)
#   --port PORT           vLLM server port (default: 8000)
#   --host HOST           vLLM server host (default: 0.0.0.0)
#   --gpu-mem-util FRAC   GPU memory utilization (default: 0.95)
#   --chat-template FILE  Path to chat template .jinja file
#                         (default: ./templates/qwen36-chat-Pajito-optimized.jinja)
#   --served-name NAME    --served-model-name value (default: model dir name)
#   --max-seqs N          Override max-num-seqs for all models
#   --max-model-len N     Override max context length
#   --kv-dtype TYPE       KV-cache dtype: auto|fp8_e4m3|fp8_e5m2 (default: auto)
#   --spec-tokens N       Speculative decoding tokens (default: 3, 0=disabled)
#   --no-prefix-cache     Disable prefix caching
#   --no-chunked-prefill Disable chunked prefill
#   --enforce-eager       Disable torch.compile
#   --cache-dir DIR       Cache directory for vLLM/torch/triton (default: $HOME/.cache/vllm-launcher)
#   --yes / -y            Accept all defaults, run non-interactively
#   --dry-run             Print the vllm command without launching
#   --help / -h           Show this help
#
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
VENV_PATH="${VENV_PATH:-$HOME/.venv/bin/activate}"
AUTO_VENV=0
MODEL_NAME=""
TP_SIZE=""
GPU_IDS=""
PORT="8000"
HOST="0.0.0.0"
GPU_MEM_UTIL="0.95"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/templates/qwen36-chat-Pajito-optimized.jinja}"
SERVED_NAME=""
MAX_SEQS_OVERRIDE=""
MAX_MODEL_LEN_OVERRIDE=""
KV_DTYPE="auto"
SPEC_TOKENS="3"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/vllm-launcher}"
MAX_BATCHED_TOKENS="2048"
YES_MODE=0
DRY_RUN=0

# Performance env defaults (can be overridden)
USE_PREFIX_CACHE=1
USE_CHUNKED_PREFILL=1
ENFORCE_EAGER=0

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
show_help() {
    sed -n '2,/^# =\+/p' "$0" | sed 's/^# \?//' | head -80
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir)      MODELS_DIR="$2"; shift 2 ;;
        --model)           MODEL_NAME="$2"; shift 2 ;;
        --venv)            VENV_PATH="$2"; shift 2 ;;
        --auto-venv)       AUTO_VENV=1; shift ;;
        --tp)              TP_SIZE="$2"; shift 2 ;;
        --gpu-ids)         GPU_IDS="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --host)            HOST="$2"; shift 2 ;;
        --gpu-mem-util)    GPU_MEM_UTIL="$2"; shift 2 ;;
        --chat-template)   CHAT_TEMPLATE="$2"; shift 2 ;;
        --served-name)     SERVED_NAME="$2"; shift 2 ;;
        --max-seqs)        MAX_SEQS_OVERRIDE="$2"; shift 2 ;;
        --max-model-len)   MAX_MODEL_LEN_OVERRIDE="$2"; shift 2 ;;
        --kv-dtype)        KV_DTYPE="$2"; shift 2 ;;
        --spec-tokens)     SPEC_TOKENS="$2"; shift 2 ;;
        --no-prefix-cache) USE_PREFIX_CACHE=0; shift ;;
        --no-chunked-prefill) USE_CHUNKED_PREFILL=0; shift ;;
        --enforce-eager)   ENFORCE_EAGER=1; shift ;;
        --cache-dir)       CACHE_DIR="$2"; shift 2 ;;
        --yes|-y)          YES_MODE=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --help|-h)         show_help ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage."; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Colors (disabled if not a TTY)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    C_RESET=$(printf '%b' '\033[0m'); C_BOLD=$(printf '%b' '\033[1m')
    C_DIM=$(printf '%b' '\033[2m'); C_GREEN=$(printf '%b' '\033[32m')
    C_YELLOW=$(printf '%b' '\033[33m'); C_CYAN=$(printf '%b' '\033[36m')
    C_RED=$(printf '%b' '\033[31m')
else
    C_RESET=''; C_BOLD=''; C_DIM=''; C_GREEN=''; C_YELLOW=''; C_CYAN=''; C_RED=''
fi

print_header() {
    echo -e "${C_BOLD}========================================${C_RESET}"
    echo -e "${C_BOLD}      vLLM Model Launcher${C_RESET}"
    echo -e "${C_BOLD}========================================${C_RESET}"
}

gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        local total
        total=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {printf "%.0f", s/1024}')
        echo -e "${C_DIM}GPUs:${C_RESET} $(nvidia-smi -L | wc -l)x $(nvidia-smi -L | head -1 | cut -d':' -f2 | cut -d'(' -f1 | xargs) ${C_DIM}(~${total} GB used)${C_RESET}"
    fi
}

# ---------------------------------------------------------------------------
# Helper: Find all vLLM-related PIDs
# ---------------------------------------------------------------------------
find_vllm_pids() {
    {
        pgrep -f "vllm serve" 2>/dev/null || true
        pgrep "VLLM::Worker" 2>/dev/null || true
        pgrep -f "python.*vllm" 2>/dev/null | grep -v "grep" || true
    } | sort -u | grep -v "^$$$" || true
}

# ---------------------------------------------------------------------------
# 0. Venv activation
# ---------------------------------------------------------------------------
activate_venv() {
    local venv_to_use=""

    if [ "$AUTO_VENV" -eq 1 ]; then
        # Try common locations
        for candidate in \
            "$SCRIPT_DIR/../.venv/bin/activate" \
            "$SCRIPT_DIR/.venv/bin/activate" \
            "$HOME/vllm_project/.venv/bin/activate" \
            "$HOME/.venv/bin/activate" \
            "$HOME/.local/share/virtualenvs/vllm/.venv/bin/activate"; do
            if [ -f "$candidate" ]; then
                venv_to_use="$candidate"
                break
            fi
        done
        # Fallback: search for vllm in any venv
        if [ -z "$venv_to_use" ]; then
            local found
            found=$(find "$HOME" -maxdepth 5 -name "activate" -path "*/bin/activate" 2>/dev/null | head -1 || true)
            if [ -n "$found" ]; then
                venv_to_use="$found"
            fi
        fi
    else
        venv_to_use="$VENV_PATH"
    fi

    if [ -n "$venv_to_use" ] && [ -f "$venv_to_use" ]; then
        echo -e "${C_DIM}Activating venv: $venv_to_use${C_RESET}"
        source "$venv_to_use"
    else
        echo -e "${C_YELLOW}⚠  No venv found. Attempting to use system vllm...${C_RESET}"
    fi

    if ! command -v vllm &> /dev/null; then
        echo -e "${C_RED}❌ 'vllm' command not found. Install vLLM or specify a venv with --venv / --auto-venv.${C_RESET}"
        exit 1
    fi

    # Inject nvidia .so directories from venv (common fix for cuDNN/CuBLAS errors)
    local venv_base
    venv_base="$(dirname "$(dirname "${venv_to_use:-}")")" 2>/dev/null || ""
    if [ -n "$venv_base" ] && [ -d "$venv_base" ]; then
        for libdir in $(find "$venv_base/lib" -path "*/nvidia/*" -name "*.so*" -exec dirname {} \; 2>/dev/null | sort -u); do
            case ":$LD_LIBRARY_PATH:" in
                *":$libdir:"*) ;;
                *) export LD_LIBRARY_PATH="$libdir:$LD_LIBRARY_PATH" ;;
            esac
        done
    fi
}

activate_venv

# ---------------------------------------------------------------------------
# Resolve models directory
# ---------------------------------------------------------------------------
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${C_YELLOW}⚠  Models directory not found: $MODELS_DIR${C_RESET}"
    echo -e "${C_DIM}   Create it and place model subdirectories inside, or use --models-dir DIR.${C_RESET}"
    echo -e "${C_DIM}   Example: mkdir -p ./models && (symlink or copy model dirs into ./models/)${C_RESET}"
    if [ "$YES_MODE" -eq 0 ]; then
        read -rp "Create '$MODELS_DIR' now? [y/N] " mkreply
        if [[ "$mkreply" =~ ^[Yy] ]]; then
            mkdir -p "$MODELS_DIR"
            echo -e "${C_GREEN}✅ Created. Place model directories inside and re-run.${C_RESET}"
            exit 0
        fi
    fi
    exit 1
fi

cd "$MODELS_DIR" || { echo -e "${C_RED}❌ Cannot cd into $MODELS_DIR${C_RESET}"; exit 1; }

# PERSISTENCE: remember last launched model
CACHE_DIR="$(eval echo "$CACHE_DIR")"  # expand ~ if present
LAST_MODEL_FILE="$CACHE_DIR/last-model"
mkdir -p "$CACHE_DIR"

# Resolve tensor-parallel size
if [ -z "$TP_SIZE" ]; then
    if command -v nvidia-smi &> /dev/null; then
        TP_SIZE=$(nvidia-smi -L | wc -l)
    else
        TP_SIZE=1
    fi
fi

# ----------------------------------------------------------------------------
# 1. Model Registry — profiles per quantization type
#    Format: profile|label|approx_size_gb|default_ctx_len|recommended|use_spec|default_seqs
#    These are sensible defaults; override at runtime as needed.
# ----------------------------------------------------------------------------
declare -A MODEL_DB=(
    ["Qwen3-AWQ"]="awq|4-bit AWQ|21.9|160000|0|1|2"
    ["Qwen3-AWQ-6Bit"]="awq_6bit|6-bit AWQ|27.7|148000|0|1|2"
    ["Qwen3-AWQ-BF16-INT4"]="awq_bf16_int4|AWQ BF16-INT4|28.3|148000|1|1|2"
    ["Qwen3-GPTQ-8bit"]="gptq_8bit|8-bit GPTQ|33.6|148000|0|1|2"
)

detect_profile_heuristic() {
    local name; name=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    local dir="$2"
    if [[ "$name" == *"6bit"* || "$name" == *"6-bit"* ]]; then
        echo "awq_6bit|6-bit AWQ|27.7|148000|0|1|2"; return
    fi
    if [[ "$name" == *"awq"* && "$name" == *"bf16"* && "$name" == *"int4"* ]]; then
        echo "awq_bf16_int4|AWQ BF16-INT4|28.3|148000|1|1|2"; return
    fi
    if [[ "$name" == *"awq"* ]]; then
        echo "awq|4-bit AWQ|21.9|160000|0|1|2"; return
    fi
    if [[ "$name" == *"gptq"* && ( "$name" == *"8bit"* || "$name" == *"8-bit"* ) ]]; then
        echo "gptq_8bit|8-bit GPTQ|33.6|148000|0|1|2"; return
    fi
    if [[ "$name" == *"gptq"* ]]; then
        echo "gptq_4bit|4-bit GPTQ|14.0|148000|0|1|2"; return
    fi
    if [[ "$name" == *"fp8"* ]]; then
        echo "fp8|FP8|27.0|90000|0|1|2"; return
    fi
    if [[ "$name" == *"marlin"* ]]; then
        echo "marlin|Marlin|14.0|160000|0|1|2"; return
    fi
    if [ -f "$dir/config.json" ]; then
        local cfg="$dir/config.json"
        if grep -qi '"quant_config".*"fp8"' "$cfg" 2>/dev/null; then
            echo "fp8|FP8|27.0|148000|0|1|2"; return
        elif grep -qi '"quant_config".*"awq"' "$cfg" 2>/dev/null; then
            echo "awq|4-bit AWQ|21.9|160000|0|1|2"; return
        elif grep -qi '"quant_config".*"gptq"' "$cfg" 2>/dev/null; then
            echo "gptq_4bit|4-bit GPTQ|14.0|160000|0|1|2"; return
        fi
    fi
    echo "native|Native BF16/FP16|55.0|32768|0|0|1"
}

# ----------------------------------------------------------------------------
# 2. Build model list (auto-discover subdirectories)
# ----------------------------------------------------------------------------
shopt -s nullglob
folders=(*/)
shopt -u nullglob
if [ ${#folders[@]} -eq 0 ]; then
    echo -e "${C_RED}❌ No model directories found in: $MODELS_DIR${C_RESET}"
    echo -e "${C_DIM}   Each subdirectory should contain a HuggingFace model (config.json, safetensors, etc.)${C_RESET}"
    exit 1
fi
models=()
for f in "${folders[@]}"; do models+=("${f%/}"); done

# ----------------------------------------------------------------------------
# 3. Selection: direct (--model flag) or interactive menu
# ----------------------------------------------------------------------------
MODEL_PATH=""
META=""

if [ -n "$MODEL_NAME" ]; then
    # Direct selection
    if [ -d "$MODELS_DIR/$MODEL_NAME" ]; then
        MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
        META="${MODEL_DB[$MODEL_NAME]:-$(detect_profile_heuristic "$MODEL_NAME" "$MODEL_PATH")}"
        echo -e "✅ Selected: ${C_BOLD}$MODEL_NAME${C_RESET}"
    else
        echo -e "${C_RED}❌ Model directory not found: $MODELS_DIR/$MODEL_NAME${C_RESET}"
        echo -e "${C_DIM}   Available: ${models[*]}${C_RESET}"
        exit 1
    fi
else
    print_header
    gpu_status
    echo ""

    LAST_MODEL=""; [ -f "$LAST_MODEL_FILE" ] && LAST_MODEL=$(<"$LAST_MODEL_FILE")

    echo -e "${C_BOLD}Available Models:${C_RESET}"
    echo ""

    declare -a MODEL_NAMES; declare -a MODEL_PATHS; declare -a MODEL_PROFILES

    i=1
    for m in "${models[@]}"; do
        MODEL_NAMES+=("$m")
        MODEL_PATHS+=("$MODELS_DIR/$m")
        meta=""
        if [ -n "${MODEL_DB[$m]:-}" ]; then meta="${MODEL_DB[$m]}"
        else meta=$(detect_profile_heuristic "$m" "$MODELS_DIR/$m"); fi
        MODEL_PROFILES+=("$meta")
        IFS='|' read -r prof label size ctx rec spec profile_seqs <<< "$meta"
        rec_marker="   "; last_marker=""
        if [ "$rec" == "1" ]; then rec_marker=" ${C_YELLOW}★${C_RESET} "; fi
        if [ "$m" == "$LAST_MODEL" ]; then last_marker=" ${C_CYAN}(last)${C_RESET}"; fi
        printf "%s[%d]  %-40s %s | ~%s GB | %s ctx | %s seqs%s\n" \
            "$rec_marker" "$i" "$m" "$label" "$size" "$ctx" "$profile_seqs" "$last_marker"
        ((i++))
    done

    echo ""
    echo -e "${C_DIM}★ = recommended | Enter alone = relaunch last used${C_RESET}"
    if [ -n "$MAX_SEQS_OVERRIDE" ]; then
        echo -e "${C_YELLOW}⚠️  MAX_SEQS_OVERRIDE=$MAX_SEQS_OVERRIDE is active for all models.${C_RESET}"
    fi
    echo ""

    if [ "$YES_MODE" -eq 1 ] && [ -n "$LAST_MODEL" ]; then
        MODEL_NAME="$LAST_MODEL"
        echo -e "✅ Auto-selecting last used: ${C_BOLD}$MODEL_NAME${C_RESET}"
    elif [ "$YES_MODE" -eq 1 ]; then
        # Pick first model
        MODEL_NAME="${models[0]}"
        META="${MODEL_PROFILES[0]}"
        echo -e "✅ Auto-selecting first model: ${C_BOLD}$MODEL_NAME${C_RESET}"
    else
        while true; do
            read -rp "Choose [1-${#models[@]}], Enter for last, or 'q' to quit: " REPLY
            if [[ "$REPLY" == "q" || "$REPLY" == "Q" ]]; then echo "Exiting..."; exit 0; fi
            if [ -z "$REPLY" ] && [ -n "$LAST_MODEL" ]; then
                MODEL_NAME="$LAST_MODEL"
                for idx in "${!MODEL_NAMES[@]}"; do
                    if [ "${MODEL_NAMES[$idx]}" == "$LAST_MODEL" ]; then META="${MODEL_PROFILES[$idx]}"; break; fi
                done
                echo -e "\n✅ Relaunching last used: ${C_BOLD}$MODEL_NAME${C_RESET}"
                break
            fi
            if [[ "$REPLY" =~ ^[0-9]+$ ]] && [ "$REPLY" -ge 1 ] && [ "$REPLY" -le "${#models[@]}" ]; then
                idx=$((REPLY - 1)); MODEL_NAME="${MODEL_NAMES[$idx]}"; META="${MODEL_PROFILES[$idx]}"
                echo -e "\n✅ Selected: ${C_BOLD}$MODEL_NAME${C_RESET}"
                break
            fi
            echo "❌ Invalid selection."
        done
    fi
    MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
fi

echo "$MODEL_NAME" > "$LAST_MODEL_FILE"

# If META not set (from --model flag), detect it
if [ -z "$META" ]; then
    META="${MODEL_DB[$MODEL_NAME]:-$(detect_profile_heuristic "$MODEL_NAME" "$MODEL_PATH")}"
fi

# ----------------------------------------------------------------------------
# 4. Parse profile
# ----------------------------------------------------------------------------
IFS='|' read -r PROFILE QUANT_LABEL FOLDER_SIZE MAX_MODEL_LEN RECOMMENDED USE_SPEC PROFILE_MAX_SEQS <<< "$META"

# Apply command-line overrides
if [ -n "$MAX_SEQS_OVERRIDE" ]; then
    MAX_SEQS="$MAX_SEQS_OVERRIDE"
    SEQS_SOURCE="global override (--max-seqs)"
else
    MAX_SEQS="$PROFILE_MAX_SEQS"
    SEQS_SOURCE="profile default"
fi

# Override context length from CLI
if [ -n "$MAX_MODEL_LEN_OVERRIDE" ]; then
    MAX_MODEL_LEN="$MAX_MODEL_LEN_OVERRIDE"
fi

echo -e "${C_DIM}----------------------------------------${C_RESET}"
echo -e "Model:       ${C_BOLD}$MODEL_NAME${C_RESET}"
echo -e "Path:        $MODEL_PATH"
echo -e "Profile:     ${C_BOLD}$PROFILE${C_RESET}"
echo -e "Quant:       $QUANT_LABEL"
echo -e "Folder size: ~${FOLDER_SIZE} GB"
echo -e "Context:     ${C_BOLD}$MAX_MODEL_LEN${C_RESET}"
echo -e "GPU util:    ${C_BOLD}$GPU_MEM_UTIL${C_RESET}"
echo -e "Tensor-parallel: ${C_BOLD}$TP_SIZE${C_RESET}"
echo -e "Max seqs:    ${C_BOLD}$MAX_SEQS${C_RESET} ${C_DIM}($SEQS_SOURCE)${C_RESET}"
if [ "$USE_SPEC" == "1" ]; then echo -e "Speculative: ${C_GREEN}MTP enabled${C_RESET}"
else echo -e "Speculative: ${C_YELLOW}disabled${C_RESET}"; fi
echo -e "Chat template: ${C_CYAN}$CHAT_TEMPLATE${C_RESET}"
echo -e "${C_DIM}----------------------------------------${C_RESET}"

if [ "$PROFILE" == "gptq_8bit" ]; then
    echo -e "${C_YELLOW}⚠️  8-bit GPTQ can be tight on limited VRAM. Monitor for OOM.${C_RESET}"
fi
if [ "$PROFILE" == "native" ]; then
    echo -e "${C_RED}⚠️  Native (unquantized) weights may not fit in your VRAM.${C_RESET}"
    if [ "$YES_MODE" -eq 0 ]; then
        read -rp "Press Enter to continue anyway or Ctrl+C to abort... "
    fi
fi
echo ""

# ----------------------------------------------------------------------------
# 5. Interactive overrides (skipped in --yes mode)
# ----------------------------------------------------------------------------
if [ "$YES_MODE" -eq 0 ]; then
    echo -e "${C_BOLD}Context Length Override${C_RESET}"
    echo -e "Model default: ${C_CYAN}$MAX_MODEL_LEN${C_RESET}"
    read -rp "Press Enter to accept, or type a new value: " CTX_OVERRIDE
    if [[ -n "$CTX_OVERRIDE" && "$CTX_OVERRIDE" =~ ^[0-9]+$ ]]; then
        MAX_MODEL_LEN="$CTX_OVERRIDE"
        echo -e "→ Context overridden to: ${C_BOLD}$MAX_MODEL_LEN${C_RESET}"
    else
        echo -e "→ Using default: ${C_BOLD}$MAX_MODEL_LEN${C_RESET}"
    fi

    echo ""
    echo -e "${C_BOLD}Max Concurrent Sequences Override${C_RESET}"
    echo -e "Current value: ${C_CYAN}$MAX_SEQS${C_RESET} ${C_DIM}($SEQS_SOURCE)${C_RESET}"
    read -rp "Press Enter to accept, or type a new value: " SEQS_OVERRIDE
    if [[ -n "$SEQS_OVERRIDE" && "$SEQS_OVERRIDE" =~ ^[0-9]+$ ]]; then
        MAX_SEQS="$SEQS_OVERRIDE"
        SEQS_SOURCE="user override"
        echo -e "→ Max seqs overridden to: ${C_BOLD}$MAX_SEQS${C_RESET}"
    else
        echo -e "→ Using: ${C_BOLD}$MAX_SEQS${C_RESET}"
    fi

    if [ "$USE_SPEC" == "1" ]; then
        echo ""
        echo -e "${C_BOLD}Speculative Tokens Override${C_RESET}"
        echo -e "Current value: ${C_CYAN}$SPEC_TOKENS${C_RESET}"
        read -rp "Press Enter to accept, or type a new number (0 to disable): " SPEC_OVERRIDE
        if [[ -n "$SPEC_OVERRIDE" && "$SPEC_OVERRIDE" =~ ^[0-9]+$ ]]; then
            SPEC_TOKENS="$SPEC_OVERRIDE"
        fi
        if [ "$SPEC_TOKENS" -eq 0 ]; then
            echo -e "→ Speculative decoding: ${C_YELLOW}Disabled (tokens = 0)${C_RESET}"
            USE_SPEC=0
        else
            echo -e "→ Speculative tokens: ${C_BOLD}$SPEC_TOKENS${C_RESET}"
        fi
    fi

    echo ""
    echo -e "${C_BOLD}KV-Cache Quantization${C_RESET}"
    echo -e "  [1] auto      ${C_DIM}bf16/fp16 • 1× size • zero loss${C_RESET}"
    echo -e "  [2] fp8_e4m3   ${C_DIM}2× smaller • ~0% loss • recommended${C_RESET}"
    echo -e "  [3] fp8_e5m2   ${C_DIM}2× smaller • alternative fp8 format${C_RESET}"
    echo ""
    while true; do
        read -rp "Select [1-3] or Enter for default ($KV_DTYPE): " KV_REPLY
        if [ -z "$KV_REPLY" ]; then KV_DTYPE="$KV_DTYPE"; break; fi
        case "$KV_REPLY" in
            1) KV_DTYPE="auto"; break ;;
            2) KV_DTYPE="fp8_e4m3"; break ;;
            3) KV_DTYPE="fp8_e5m2"; break ;;
            *) echo "❌ Invalid choice." ;;
        esac
    done
    echo -e "→ KV-cache dtype: ${C_BOLD}$KV_DTYPE${C_RESET}"
    echo ""
fi

# ----------------------------------------------------------------------------
# 6. Environment & Launch Args
# ----------------------------------------------------------------------------
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# Cache directories
export VLLM_CACHE_DIR="$CACHE_DIR/vllm_cache"
export VLLM_TORCH_COMPILE_CACHE_DIR="$CACHE_DIR/torch_compile_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
mkdir -p "$VLLM_CACHE_DIR" "$VLLM_TORCH_COMPILE_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# Use PyTorch's more stable compile cache format
export VLLM_COMPILE_CACHE_SAVE_FORMAT="${VLLM_COMPILE_CACHE_SAVE_FORMAT:-unpacked}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export VLLM_COMPILE_DEPYF="${VLLM_COMPILE_DEPYF:-0}"

# Served model name
if [ -z "$SERVED_NAME" ]; then
    SERVED_NAME="$MODEL_NAME"
fi

# Chat template validation
if [ ! -f "$CHAT_TEMPLATE" ]; then
    echo -e "${C_YELLOW}⚠️  Chat template not found: $CHAT_TEMPLATE${C_RESET}"
    echo -e "${C_DIM}   Launching without --chat-template (vLLM will use the model's built-in template).${C_RESET}"
    CHAT_TEMPLATE=""
fi

# Build vLLM args array
VLLM_ARGS=(
    "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --tensor-parallel-size "$TP_SIZE"
    --max-num-seqs "$MAX_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --kv-cache-dtype "$KV_DTYPE"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --enable-auto-tool-choice
    --attention-backend FLASHINFER
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --default-chat-template-kwargs '{"enable_thinking": true, "auto_disable_thinking_with_tools": true, "preserve_thinking": true}'
    --trust-remote-code
    --host "$HOST"
    --port "$PORT"
    --override-generation-config '{"temperature":0.7, "top_p":0.95, "top_k":20, "min_p":0.0, "repetition_penalty":1.0, "presence_penalty":0.0}'
)

# Conditionally add optional flags
if [ -n "$CHAT_TEMPLATE" ]; then
    VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi
if [ "$USE_PREFIX_CACHE" -eq 1 ]; then
    VLLM_ARGS+=(--enable-prefix-caching)
fi
if [ "$USE_CHUNKED_PREFILL" -eq 1 ]; then
    VLLM_ARGS+=(--enable-chunked-prefill)
fi
if [ "$ENFORCE_EAGER" -eq 1 ]; then
    VLLM_ARGS+=(--enforce-eager)
fi

if [[ "$USE_SPEC" == "1" && "$SPEC_TOKENS" -gt 0 ]]; then
    VLLM_ARGS+=(
        --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $SPEC_TOKENS}"
    )
fi

# ----------------------------------------------------------------------------
# 7. Pre-flight: Clear stale workers from previous runs
# ----------------------------------------------------------------------------
echo -e "${C_BOLD}Pre-flight GPU Check:${C_RESET}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader | while read -r line; do
        echo -e "  $line"
    done
    echo ""
fi

OLD_PIDS=$(find_vllm_pids)
if [ -n "$OLD_PIDS" ]; then
    echo -e "${C_YELLOW}⚠️  Existing vLLM processes detected:${C_RESET}"
    ps -fp $OLD_PIDS 2>/dev/null || true
    echo ""
    echo -e "${C_DIM}Waiting 5 seconds for graceful termination...${C_RESET}"
    sleep 5

    OLD_PIDS=$(find_vllm_pids)
    if [ -n "$OLD_PIDS" ]; then
        echo -e "${C_RED}Force killing stale vLLM processes...${C_RESET}"
        echo "$OLD_PIDS" | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
    echo -e "${C_GREEN}✅ Stale processes cleared${C_RESET}"
    echo ""
fi

# ----------------------------------------------------------------------------
# 8. Launch vLLM
# ----------------------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${C_CYAN}[DRY RUN] Would execute:${C_RESET}"
    if [ -n "$GPU_IDS" ]; then
        echo "CUDA_VISIBLE_DEVICES=$GPU_IDS vllm serve ${VLLM_ARGS[*]}"
    else
        echo "vllm serve ${VLLM_ARGS[*]}"
    fi
    exit 0
fi

echo -e "${C_GREEN}✅ Launching vLLM on ${HOST}:${PORT}${C_RESET}"
echo -e "${C_DIM}   Tensor-parallel: $TP_SIZE${C_RESET}"
echo -e "${C_DIM}   Max seqs: $MAX_SEQS | Context: $MAX_MODEL_LEN | KV: $KV_DTYPE${C_RESET}"
echo -e "${C_DIM}   Press Ctrl+C ONCE, then wait for the cleanup message.${C_RESET}"
echo ""

if [ -n "$GPU_IDS" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_IDS" vllm serve "${VLLM_ARGS[@]}"
else
    vllm serve "${VLLM_ARGS[@]}"
fi
VLLM_EXIT_CODE=$?

echo ""
if [ $VLLM_EXIT_CODE -ne 0 ]; then
    echo -e "${C_RED}❌ vLLM exited with code $VLLM_EXIT_CODE${C_RESET}"
else
    echo -e "${C_GREEN}✅ vLLM main process exited${C_RESET}"
fi

# ----------------------------------------------------------------------------
# 9. Cooldown & worker cleanup
# ----------------------------------------------------------------------------
echo -e "${C_DIM}Waiting 5 seconds for workers to shut down naturally...${C_RESET}"
sleep 5

REMAINING=$(find_vllm_pids)
if [ -n "$REMAINING" ]; then
    echo -e "${C_YELLOW}Sending SIGTERM to remaining workers...${C_RESET}"
    echo "$REMAINING" | xargs -r kill -TERM 2>/dev/null || true
    sleep 3
fi

REMAINING=$(find_vllm_pids)
if [ -n "$REMAINING" ]; then
    echo -e "${C_RED}Force killing stubborn worker processes...${C_RESET}"
    echo "$REMAINING" | xargs -r kill -9 2>/dev/null || true
    sleep 1
fi

echo ""
echo -e "${C_BOLD}Final GPU Status:${C_RESET}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
fi
echo -e "\n${C_GREEN}✅ Cleanup complete. GPU VRAM released.${C_RESET}"
