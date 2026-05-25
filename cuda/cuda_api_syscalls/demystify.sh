#!/usr/bin/env bash
# cuda-demystify · cuda_api_syscalls
#
# 1. Always runs ltrace with full info (all lib calls + syscalls, timestamps,
#    per-thread) and saves the raw output to ltrace.txt.
# 2. If --func is given (repeatable), runs filter_ltrace_syscalls.py to extract
#    the syscall window for each invocation of the specified function(s).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

B='\033[1m'; C='\033[36m'; G='\033[32m'; R='\033[31m'; X='\033[0m'

usage() {
    printf "${B}Usage:${X} %s [--func FUNC ...] [--output-dir DIR] -- COMMAND [ARGS...]\n\n" "$(basename "$0")"
    printf "Always captures full ltrace output (all library calls + syscalls, timestamps,\n"
    printf "per-thread).  If --func is given, also writes syscalls_per_func.txt.\n\n"
    printf "${B}Options:${X}\n"
    printf "  --func FUNC, -f FUNC   CUDA API to filter (repeatable)\n"
    printf "                         e.g. -f cuInit -f cudaMalloc\n"
    printf "  --output-dir DIR, -o DIR\n"
    printf "                         output directory (default: \$PWD/.cuda-demystify/<ts>)\n"
    printf "  -h, --help\n\n"
    printf "${B}Output:${X}\n"
    printf "  ltrace.txt             raw ltrace (all lib calls + syscalls, always)\n"
    printf "  syscalls_per_func.txt  syscalls scoped per --func call (if --func given)\n\n"
    printf "${B}Prerequisites:${X}\n"
    printf "  ltrace : sudo apt install ltrace\n"
    exit "${1:-0}"
}

# ── argument parsing ──────────────────────────────────────────────────────────
FUNCS=()
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --func|-f)        FUNCS+=("$2");    shift 2 ;;
        --output-dir|-o)  OUTPUT_DIR="$2";  shift 2 ;;
        -h|--help)        usage 0 ;;
        --)               shift; break ;;
        -*)  printf "${R}error:${X} unknown option '%s'\n" "$1" >&2; usage 1 ;;
        *)   break ;;
    esac
done

[[ $# -eq 0 ]] && { printf "${R}error:${X} no command specified\n" >&2; usage 1; }

# ── resolve tools ─────────────────────────────────────────────────────────────
if ! command -v ltrace &>/dev/null; then
    printf "${R}error:${X} ltrace not found — install: sudo apt install ltrace\n" >&2
    exit 1
fi

# ── output directory ──────────────────────────────────────────────────────────
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$PWD/.cuda-demystify/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

LTRACE_OUT="$OUTPUT_DIR/ltrace.txt"

# ── banner ────────────────────────────────────────────────────────────────────
printf "\n${B}cuda-demystify  ·  cuda_api_syscalls${X}\n"
printf "  command    : %s\n" "$*"
printf "  func(s)    : %s\n" "${FUNCS[*]:-(not specified — raw ltrace only)}"
printf "  output dir : %s\n\n" "$OUTPUT_DIR"

# ── ltrace (always) ───────────────────────────────────────────────────────────
# -f    : follow all threads/forks (adds [pid XXXX] prefix per line)
# -S    : interleave SYS_xxx syscall lines with library call lines
# -ttt  : Unix timestamp with microseconds on every line
# -C    : demangle C++ symbols
# no -e : capture ALL dynamic library calls
printf "  ${C}[1/2]${X} running ltrace...\n"
ltrace -f -S -ttt -C -o "$LTRACE_OUT" "$@" || true
printf "  ${G}✓${X}  ltrace → %s\n" "$LTRACE_OUT"

# ── filter (only when --func given) ──────────────────────────────────────────
if [[ ${#FUNCS[@]} -gt 0 ]]; then
    FILTERED_OUT="$OUTPUT_DIR/syscalls_per_func.txt"
    FUNC_ARGS=()
    for func in "${FUNCS[@]}"; do FUNC_ARGS+=(--func "$func"); done

    printf "  ${C}[2/2]${X} filtering...\n"
    python3 "$SCRIPT_DIR/filter_ltrace_syscalls.py" \
        "${FUNC_ARGS[@]}" "$LTRACE_OUT" >"$FILTERED_OUT"
    printf "  ${G}✓${X}  filtered  → %s\n" "$FILTERED_OUT"
fi

printf "\n  output: ${B}%s${X}\n\n" "$OUTPUT_DIR"
