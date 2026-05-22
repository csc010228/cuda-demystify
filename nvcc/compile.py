#!/usr/bin/env python3
"""
cuda-demystify · nvcc pipeline inspector

Strategy
--------
1. Run `nvcc --dryrun` to get the exact sub-commands nvcc would execute.
2. Redirect every temp-file path from /tmp → <outdir>/tmp/ so all
   intermediate products land in a controlled directory.
3. Replay each sub-command individually via bash.
4. After each step, copy the newly produced files into the step's own
   subfolder and write a standalone run.sh for that step.
5. Run cuobjdump on the resulting CUBIN / executable for SASS output.

Output layout
-------------
<outdir>/
  dryrun.sh                  runnable: shows the nvcc pipeline
  step_01_preprocess/
    run.sh                   runnable: replays this step
    *.cpp4.ii                files produced by this step
  step_02_cudafe/
    run.sh
    *.cudafe1.cpp  *.module_id  ...
  ...
  step_NN_<kind>/
    run.sh
    <output files>
  cuobjdump/
    <name>_sass.txt  <name>_ptx.txt  <name>_elf.txt
  executable
  tmp/                       all intermediate files
"""

from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── terminal colours ──────────────────────────────────────────────────────────
B, C, G, R, Y, D, X = ("\033[1m", "\033[36m", "\033[32m",
                         "\033[31m", "\033[33m", "\033[2m", "\033[0m")

def bold(s):   return f"{B}{s}{X}"
def cyan(s):   return f"{C}{s}{X}"
def green(s):  return f"{G}{s}{X}"
def red(s):    return f"{R}{s}{X}"
def yellow(s): return f"{Y}{s}{X}"
def dim(s):    return f"{D}{s}{X}"

def section(title: str) -> None:
    print(f"\n{cyan('─── ' + title + ' ───')}")

# ── pipeline step classifier ──────────────────────────────────────────────────
_HOST_CC = {
    "gcc", "g++", "cc", "c++", "clang", "clang++",
    "x86_64-linux-gnu-gcc", "x86_64-linux-gnu-g++",
    "aarch64-linux-gnu-gcc",
}

def classify(cmd: str) -> tuple[str, str]:
    """Return (key, human description) for a sub-command string."""
    tokens = cmd.split()
    if not tokens:
        return "other", "other"
    prog = os.path.basename(tokens[0].strip('"'))
    flags = set(tokens[1:])

    if prog in ("rm", "del"):
        return "rm", "rm  (remove temp file)"
    if "cudafe" in prog:
        return "cudafe", "cudafe++  ·  CUDA front-end: split host / device code"
    if prog == "cicc":
        return "cicc", "cicc  ·  CUDA C++ → PTX  (NVVM IR compiler)"
    if prog == "ptxas":
        return "ptxas", "ptxas  ·  PTX → CUBIN  (GPU machine code)"
    if "fatbinary" in prog:
        return "fatbinary", "fatbinary  ·  bundle PTX + CUBIN → fat binary"
    if prog == "nvlink":
        return "nvlink", "nvlink  ·  device-side linking"
    if prog in _HOST_CC or prog.endswith("-gcc") or prog.endswith("-g++"):
        if "-E" in flags:
            return "preprocess", "gcc -E  ·  preprocessing (expand macros, inline headers)"
        if "-S" in flags:
            return "host_asm", "gcc -S  ·  host assembly"
        if "-c" in flags:
            return "host_compile", "gcc -c  ·  host compilation → object file"
        return "host_link", "g++  ·  host link → executable"
    if prog in ("ld", "collect2") or prog.endswith("-ld"):
        return "host_link", "ld  ·  host link → executable"
    return "other", f"other  ({prog})"


# ── file-extension labels ─────────────────────────────────────────────────────
_EXT_LABELS: list[tuple[str, str]] = [
    ("cpp4.ii",        "Preprocessed CUDA source (device pass 1)"),
    ("cpp1.ii",        "Preprocessed CUDA source (device pass 2)"),
    ("cudafe1.cpp",    "Host C++ after cudafe (device stubs removed)"),
    ("cudafe1.stub.c", "Host stub for device functions"),
    ("cudafe1.gpu",    "Device code after cudafe (NVVM IR)"),
    ("cudafe1.c",      "Device C stub after cicc"),
    ("module_id",      "Module ID file"),
    ("ptx",            "PTX: GPU virtual ISA (human-readable)"),
    ("cubin",          "CUBIN: GPU machine code"),
    ("fatbin.c",       "Fat binary as C source array"),
    ("fatbin",         "Fat binary"),
    ("reg.c",          "Device-link registration C source"),
    ("dlink.o",        "Device-link object"),
    ("o",              "Object file"),
    ("s",              "Assembly"),
]

def label_file(p: Path) -> str:
    name = p.name
    for ext, lbl in _EXT_LABELS:
        if name.endswith("." + ext):
            return lbl
    return ""


# ── dryrun parsing ────────────────────────────────────────────────────────────
def parse_dryrun(text: str) -> tuple[list[str], list[dict]]:
    """
    Split nvcc --dryrun output (lines prefixed with `#$ `) into:
      var_lines  – safe shell variable assignments  (e.g. CICC_PATH=…)
      steps      – list of {"cmd": str, "skip": bool, "skip_reason": str}

    Multi-word pseudo-assignments like LIBRARIES= "-L/a" "-L/b" are not
    valid bash and would produce spurious errors; they're dropped because
    every command already embeds the full paths literally.

    rm/del commands are marked skip=True — we preserve all intermediates.
    """
    var_lines: list[str] = []
    steps: list[dict] = []
    for raw in text.splitlines():
        if not raw.startswith("#$ "):
            continue
        content = raw[3:]
        if re.match(r"^\w+=", content):
            eq = content.index("=")
            value = content[eq + 1:].strip()
            # Only include assignments with a single-token value; multi-word
            # values (e.g. LIBRARIES= "-L/a" "-L/b") would be misinterpreted
            # by bash as a command invocation.
            single_token = (
                not value
                or " " not in value
                or (value.startswith('"') and value.count('"') == 2)
            )
            if single_token:
                var_lines.append(content)
        else:
            prog = os.path.basename(content.split()[0].strip('"')) if content.split() else ""
            if prog in ("rm", "del"):
                steps.append({"cmd": content, "skip": True,
                               "skip_reason": "rm skipped — keeping all intermediates"})
            else:
                steps.append({"cmd": content, "skip": False, "skip_reason": ""})
    return var_lines, steps


# ── temp-path redirection ─────────────────────────────────────────────────────
_TMPXFT_DIR_RE = re.compile(
    r'(/[^\s"\']*?)/tmpxft_[0-9a-f]+_[0-9a-f]+-\d+'
)

def find_orig_tmp_dir(dryrun_text: str) -> str | None:
    """Return the directory that holds nvcc's tmpxft files (usually /tmp)."""
    m = _TMPXFT_DIR_RE.search(dryrun_text)
    return m.group(1) if m else None


def redirect_temps(cmd: str, orig_dir: str, new_dir: str) -> str:
    """
    Replace every occurrence of `orig_dir/tmpxft_` with `new_dir/tmpxft_`
    in cmd, leaving all other paths (e.g. the final exe path) untouched.
    """
    return cmd.replace(orig_dir + "/tmpxft_", new_dir + "/tmpxft_")


# ── write dryrun.sh ───────────────────────────────────────────────────────────
def write_dryrun_sh(outdir: Path, dryrun_cmd: list[str], dryrun_text: str) -> Path:
    path = outdir / "dryrun.sh"
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Run nvcc --dryrun to show the full compilation pipeline\n")
        f.write("# without executing any commands.\n\n")
        f.write(" ".join(dryrun_cmd) + "\n")
    path.chmod(0o755)
    return path


# ── write step run.sh ─────────────────────────────────────────────────────────
def write_run_sh(step_dir: Path, n: int, desc: str, cmd: str,
                 var_lines: list[str]) -> None:
    path = step_dir / "run.sh"
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f"# Step {n}: {desc}\n")
        f.write("# Re-run this step in isolation.\n")
        f.write("# Input files are read from the shared tmp/ directory.\n\n")
        f.write("set -euo pipefail\n\n")
        if var_lines:
            f.write("# nvcc environment variables\n")
            for v in var_lines:
                f.write(v + "\n")
            f.write("\n")
        f.write("# Command\n")
        f.write(cmd + "\n")
    path.chmod(0o755)


# ── run one step ──────────────────────────────────────────────────────────────
def run_step(
    n: int,
    total: int,
    cmd: str,
    var_lines: list[str],
    outdir: Path,
    temps_dir: Path,
) -> dict:
    kind, desc = classify(cmd)
    step_dir = outdir / f"step_{n:02d}_{kind}"
    step_dir.mkdir(exist_ok=True)

    print(f"\n  {cyan(f'[{n:02d}/{total}]')}  {bold(desc)}")
    print(f"  {dim('$')} {cmd}")

    # Write run.sh before execution so it exists even if the step fails
    write_run_sh(step_dir, n, desc, cmd, var_lines)

    # Build a mini bash script: set all nvcc variables, then run the command.
    # CWD is temps_dir so that relative-path outputs from cudafe++ land there.
    script = "\n".join(var_lines) + "\n" + cmd

    # Snapshot temps before
    before = {p for p in temps_dir.rglob("*") if p.is_file()}

    t0 = time.monotonic()
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(temps_dir),
    )
    elapsed = time.monotonic() - t0

    # Snapshot temps after — copy new files into the step subfolder
    after = {p for p in temps_dir.rglob("*") if p.is_file()}
    new_files = sorted(after - before)
    for src in new_files:
        shutil.copy2(src, step_dir / src.name)

    # Write step log inside the step folder
    log_path = step_dir / "output.log"
    with open(log_path, "w") as lf:
        lf.write(f"# Step {n}/{total}: {desc}\n")
        lf.write(f"# Command:\n#   {cmd}\n\n")
        if result.stdout:
            lf.write("## stdout:\n" + result.stdout + "\n")
        if result.stderr:
            lf.write("## stderr:\n" + result.stderr + "\n")

    ok = result.returncode == 0
    status = green("✓") if ok else red("✗")
    print(f"  {status}  exit {result.returncode}  ({elapsed:.2f}s)"
          f"  [{step_dir.name}/]")

    if not ok and result.stderr.strip():
        for line in result.stderr.strip().splitlines()[:12]:
            print(f"      {red(line)}")

    if new_files:
        print(f"  {dim('→ step dir:')} {step_dir.name}/")
        for f in new_files:
            lbl = label_file(f)
            print(f"    {green(f.name)}" + (f"  {dim(lbl)}" if lbl else ""))

    return {
        "n": n,
        "kind": kind,
        "desc": desc,
        "cmd": cmd,
        "ok": ok,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "new_files": new_files,
        "step_dir": step_dir,
    }



# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="compile.py",
        description="cuda-demystify: inspect the full nvcc compilation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 compile.py kernel.cu
  python3 compile.py kernel.cu -o ./my-run
  python3 compile.py kernel.cu -- -arch=sm_86 -O2
  python3 compile.py kernel.cu -o ./out -- -arch=sm_80 -lineinfo -G
""",
    )
    ap.add_argument("input_cu", help="CUDA source file (.cu)")
    ap.add_argument(
        "-o", "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory (default: .cuda-demystify/<timestamp>)",
    )
    # parse_known_args: our flags (-o) are recognized; everything else
    # (including flags starting with -) becomes the nvcc extra args list.
    # This avoids the REMAINDER greediness problem where argparse.REMAINDER
    # would swallow -o before it could be recognised as our own flag.
    args, nvcc_extra = ap.parse_known_args()
    args.nvcc_args = [a for a in nvcc_extra if a != "--"]
    return args


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    nvcc_extra = [a for a in args.nvcc_args if a != "--"]

    # Validate
    cu_file = Path(args.input_cu).resolve()
    if not cu_file.exists():
        sys.exit(f"Error: '{args.input_cu}' not found.")
    if cu_file.suffix != ".cu":
        sys.exit(f"Error: input file must have a .cu extension.")
    if not shutil.which("nvcc"):
        sys.exit("Error: nvcc not found in PATH.")

    # Resolve output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(f".cuda-demystify/{ts}").resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)
    temps_dir = outdir / "tmp"
    temps_dir.mkdir(exist_ok=True)
    exe_path = outdir / "executable"

    # Header
    print()
    print(bold("cuda-demystify  ·  nvcc pipeline inspector"))
    ver_raw = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    ver_line = next((l for l in ver_raw.splitlines() if "release" in l.lower()), "unknown")
    print(f"  input      : {cu_file}")
    print(f"  output dir : {outdir}")
    print(f"  nvcc       : {ver_line.strip()}")
    print(f"  extra args : {' '.join(nvcc_extra) or '(none)'}")

    # ── Stage 0: dryrun ──────────────────────────────────────────────────────
    section("Stage 0  ·  nvcc --dryrun  (discover pipeline)")
    dryrun_cmd = ["nvcc", "--dryrun"] + nvcc_extra + [str(cu_file), "-o", str(exe_path)]
    print(f"  {dim('$')} {' '.join(dryrun_cmd)}")

    dr = subprocess.run(dryrun_cmd, capture_output=True, text=True)
    dryrun_text = dr.stderr if dr.stderr.strip() else dr.stdout

    # Write dryrun.sh
    dryrun_sh = write_dryrun_sh(outdir, dryrun_cmd, dryrun_text)
    (outdir / "dryrun.log").write_text(dryrun_text)

    var_lines, steps = parse_dryrun(dryrun_text)
    exec_steps = [s for s in steps if not s["skip"]]
    skip_steps = [s for s in steps if s["skip"]]
    print(f"  {green('✓')}  dryrun.sh  "
          f"({len(exec_steps)} commands, {len(skip_steps)} skipped rm, "
          f"{len(var_lines)} variable assignments)")

    if not exec_steps:
        print(red("  No commands found in dryrun output — see dryrun.log"))
        print(dim("  raw output:"))
        print(dryrun_text[:2000])
        sys.exit(1)

    # Find and announce the temp-file redirection
    orig_tmp = find_orig_tmp_dir(dryrun_text)
    if orig_tmp:
        print(f"  Temp dir in dryrun : {orig_tmp}/")
        print(f"  Redirecting to     : {temps_dir}/")
    else:
        print(f"  {yellow('Warning: could not find tmpxft pattern — temp paths not redirected')}")

    # Apply path redirection to every command
    if orig_tmp:
        for s in steps:
            s["cmd"] = redirect_temps(s["cmd"], orig_tmp, str(temps_dir))

    # ── Execute each sub-command ──────────────────────────────────────────────
    section(f"Executing {len(exec_steps)} pipeline sub-commands")

    results: list[dict] = []
    for i, s in enumerate(exec_steps, 1):
        r = run_step(i, len(exec_steps), s["cmd"], var_lines, outdir, temps_dir)
        results.append(r)


    # ── final summary ─────────────────────────────────────────────────────────
    section("Summary")
    print(f"  Output : {outdir}\n")

    ok_n    = sum(1 for r in results if r["ok"])
    fail_n  = len(results) - ok_n
    total_t = sum(r["elapsed"] for r in results)

    for r in results:
        sym = green("✓") if r["ok"] else red("✗")
        t   = dim(f"({r['elapsed']:.2f}s)")
        print(f"  {sym}  {r['n']:2d}.  {r['desc']}  {t}")
        for f in r["new_files"]:
            lbl = label_file(f)
            print(f"           → {f.name}" + (f"  {dim(lbl)}" if lbl else ""))
    if skip_steps:
        print(f"  {dim('·')}  {len(skip_steps)} rm command(s) skipped  "
              f"{dim('(intermediates kept)')}")

    print()
    print(f"  {bold('Output structure:')}")
    print(f"  {outdir.name}/")
    print(f"    dryrun.sh             # nvcc --dryrun  (show pipeline)")
    for r in results:
        files_preview = "  ".join(f.name for f in r["new_files"][:2])
        ellipsis = " …" if len(r["new_files"]) > 2 else ""
        print(f"    {r['step_dir'].name}/")
        print(f"      run.sh            # replay step {r['n']}")
        if files_preview:
            print(f"      {files_preview}{ellipsis}")
    if exe_path.exists():
        sz = exe_path.stat().st_size
        print(f"    executable            # ({sz:,} bytes)")
    print()

    print(f"  total time : {total_t:.2f}s")
    if fail_n == 0:
        print(f"  {green(f'All {ok_n} steps succeeded.')}")
    else:
        print(f"  {red(f'{fail_n} step(s) failed.')}  {ok_n} succeeded.")
    print()


if __name__ == "__main__":
    main()
