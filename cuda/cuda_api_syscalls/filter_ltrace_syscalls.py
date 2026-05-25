#!/usr/bin/env python3
"""
Parse ltrace -f -S -ttt output and, for each specified CUDA API function,
extract all its invocations together with every syscall made during each call.

Reports include line numbers into the original ltrace file so findings can be
cross-referenced easily.

ltrace -S interleaves library-call lines with SYS_xxx lines. When a library
function issues syscalls before returning, ltrace uses the unfinished/resumed
format so nested events appear in chronological order:

    [pid 1234] 1748000000.123456 cudaMalloc(0x7ffe..., 1024 <unfinished ...>
    [pid 1234] 1748000000.123500 SYS_mmap(0, 4096, ...)        = 0x7f...
    [pid 1234] 1748000000.124000 <... cudaMalloc resumed>) = 0

When the function returns before any syscall it is a single line:

    [pid 1234] 1748000000.125000 cudaMalloc(0x7ffe..., 512) = 0

Usage:
    python3 filter_ltrace_syscalls.py --func cuInit --func cudaMalloc ltrace.txt
    python3 filter_ltrace_syscalls.py -f cuInit ltrace.txt > report.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class Call:
    func: str
    pid: str
    entry_lineno: int
    entry_text: str
    exit_lineno: int | None = None
    exit_text: str | None = None
    syscalls: list[tuple[int, str]] = field(default_factory=list)


# ── line parsing ──────────────────────────────────────────────────────────────

# ltrace -f adds "[pid XXXX] " before every line (including main thread).
# ltrace -ttt adds a Unix timestamp (seconds.microseconds) after that.
# Full prefix:  [pid 1234] 1748000000.123456 <content>
# Without -f:   1748000000.123456 <content>
_PID_RE   = re.compile(r"^\[pid\s+(\d+)\]\s+")
_SYSCALL_RE = re.compile(r"\bSYS_\w+")


def extract_pid(line: str) -> str:
    m = _PID_RE.match(line)
    return m.group(1) if m else "main"


def strip_prefix(line: str) -> str:
    """Remove optional [pid NNN] and timestamp, return just the content."""
    s = _PID_RE.sub("", line)             # drop [pid NNN]
    s = re.sub(r"^\d+\.\d+\s+", "", s)   # drop Unix timestamp
    return s.strip()


# ── per-function regex patterns ───────────────────────────────────────────────

def make_patterns(func: str) -> dict[str, re.Pattern]:
    esc = re.escape(func)
    return {
        # func_name(... <unfinished ...>
        "unfinished": re.compile(rf"\b{esc}\s*\(.*<unfinished\b"),
        # <... func_name resumed> ...) = value
        "resumed":    re.compile(rf"<\.\.\.\s+{esc}\s+resumed>"),
        # func_name(...) = value   (no nested syscalls, single line)
        "immediate":  re.compile(rf"\b{esc}\s*\(.*\)\s*="),
    }


# ── main parser ───────────────────────────────────────────────────────────────

def parse(lines: list[str], funcs: list[str]) -> dict[str, list[Call]]:
    """
    Walk the ltrace output once and collect Call objects per function.

    Per-thread (pid) tracking ensures syscalls are attributed only to the
    function call that is currently in progress on the same thread.
    """
    patterns = {f: make_patterns(f) for f in funcs}

    # (pid, func_name) -> Call currently in progress on that thread
    pending: dict[tuple[str, str], Call] = {}
    # pid -> set of function names currently executing on that pid
    active: dict[str, set[str]] = defaultdict(set)

    results: dict[str, list[Call]] = {f: [] for f in funcs}

    for lineno, raw in enumerate(lines, 1):
        line    = raw.rstrip()
        pid     = extract_pid(line)
        content = strip_prefix(line)

        matched = False
        for func, pats in patterns.items():
            if pats["unfinished"].search(content):
                pending[(pid, func)] = Call(
                    func=func, pid=pid,
                    entry_lineno=lineno, entry_text=line,
                )
                active[pid].add(func)
                matched = True
                break

            if pats["resumed"].search(content):
                key = (pid, func)
                if key in pending:
                    call = pending.pop(key)
                    call.exit_lineno = lineno
                    call.exit_text   = line
                    results[func].append(call)
                    active[pid].discard(func)
                matched = True
                break

            if pats["immediate"].search(content) and "resumed" not in content:
                results[func].append(Call(
                    func=func, pid=pid,
                    entry_lineno=lineno, entry_text=line,
                    exit_lineno=lineno,
                ))
                matched = True
                break

        # Syscall lines: attribute to every function currently open on this pid.
        if not matched and _SYSCALL_RE.search(content):
            for func in active.get(pid, set()):
                key = (pid, func)
                if key in pending:
                    pending[key].syscalls.append((lineno, line))

    # Flush calls still open when the process was killed.
    for (pid, func), call in pending.items():
        call.exit_text = "(process ended before function returned)"
        results[func].append(call)

    return results


# ── report ────────────────────────────────────────────────────────────────────

def print_report(results: dict[str, list[Call]]) -> None:
    for func, calls in results.items():
        n = len(calls)
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  {func}()  —  {n} call{'s' if n != 1 else ''} found")
        print(sep)

        if not calls:
            print()
            print("  (not found in ltrace output)")
            print("  If the function is statically linked, ltrace cannot intercept")
            print("  it through the PLT. Check bpftrace.txt instead.")
            continue

        for i, call in enumerate(calls, 1):
            if call.exit_lineno is None or call.exit_lineno == call.entry_lineno:
                line_range = f"line {call.entry_lineno}"
            else:
                line_range = f"lines {call.entry_lineno}–{call.exit_lineno}"

            print(f"\n  Call #{i}  [pid {call.pid}]  {line_range}")
            print(f"  {'─' * 64}")
            print(f"  ENTRY [line {call.entry_lineno:>6}]:  {call.entry_text}")
            if call.exit_text:
                lineno_label = call.exit_lineno if call.exit_lineno else "?"
                print(f"  EXIT  [line {lineno_label:>6}]:  {call.exit_text}")

            if call.syscalls:
                print(f"\n  Syscalls during this call ({len(call.syscalls)}):")
                for sc_lineno, sc_text in call.syscalls:
                    print(f"    [line {sc_lineno:>6}]  {sc_text}")
            else:
                print("\n  Syscalls: (none)")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--func", "-f",
        action="append",
        required=True,
        metavar="FUNC_NAME",
        help="function to filter; can be repeated for multiple functions",
    )
    ap.add_argument("ltrace_file", help="ltrace -f -S output file")
    args = ap.parse_args()

    with open(args.ltrace_file) as fh:
        lines = fh.readlines()

    results = parse(lines, args.func)
    print_report(results)


if __name__ == "__main__":
    main()
