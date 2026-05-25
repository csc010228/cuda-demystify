# `cuda_api_syscalls`

Trace any CUDA runtime or driver API function and capture every system call
made during each of its invocations, using ltrace.

## Files

| File | Purpose |
|------|---------|
| `demystify.sh` | Orchestration script |
| `filter_ltrace_syscalls.py` | Post-process ltrace output → syscalls per call |

## Usage

```bash
# Raw ltrace only (all library calls + syscalls, no filtering)
bash cuda/cuda_api_syscalls/demystify.sh -- COMMAND [ARGS...]

# With function filtering (one or more --func flags)
bash cuda/cuda_api_syscalls/demystify.sh \
    --func cuInit \
    --func cudaMalloc \
    -- COMMAND [ARGS...]
```

`--func` accepts any CUDA runtime or driver API symbol:

```bash
-f __cudaRegisterFatBinary   # runtime internal
-f cudaMalloc                # runtime API
-f cuInit                    # driver API
-f cuLaunchKernel            # driver API
```

## Output

```
.cuda-demystify/<timestamp>/
  ltrace.txt             raw ltrace -f -S -ttt -C output (always written)
  syscalls_per_func.txt  syscalls scoped to each --func call (if --func given)
```

`ltrace.txt` contains all dynamic library calls and syscalls with Unix
timestamps and per-thread `[pid XXXX]` prefixes.

`syscalls_per_func.txt` example:

```
====================================================================
  cuInit()  —  1 call found
====================================================================

  Call #1  [pid 1234]  lines 42–89
  ────────────────────────────────────────────────────────────────
  ENTRY [line    42]:  [pid 1234] 1748000000.123 cuInit(0 <unfinished ...>
  EXIT  [line    89]:  [pid 1234] 1748000000.890 <... cuInit resumed>) = 0

  Syscalls during this call (3):
    [line    43]  [pid 1234] 1748000000.124 SYS_openat("/dev/nvidiactl", ...) = 3
    [line    44]  [pid 1234] 1748000000.125 SYS_ioctl(3, ...) = 0
    [line    45]  [pid 1234] 1748000000.126 SYS_mmap(0, 4096, ...) = 0x7f...
```

Line numbers reference `ltrace.txt` directly for cross-referencing.

## How it works

`ltrace -f -S -ttt -C` interleaves `SYS_xxx` syscall lines with library call
lines. When a function makes syscalls before returning, ltrace uses the
`<unfinished>`/`<resumed>` format. `filter_ltrace_syscalls.py` walks the file
once per thread (`[pid XXXX]`), tracking which function is currently open, and
collects every `SYS_xxx` line that falls inside that window.

> **Note:** ltrace intercepts calls through the PLT, so it only works for
> dynamically-linked functions. If a function is statically linked (e.g. via
> `-lcudart_static`), it will not appear in `ltrace.txt`.

## Prerequisites

```bash
sudo apt install ltrace
```
