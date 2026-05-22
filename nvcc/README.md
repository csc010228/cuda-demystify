# Demystify nvcc compile trajectory

Inspect every intermediate product that `nvcc` produces when compiling a `.cu` file into an executable.

Run `nvcc --dryrun` to discover the full sub-command pipeline, then replay each command one by one so every intermediate product is captured and visible.

## Usage

```
python3 compile.py <input.cu> [-o output_dir] [-- nvcc_args...]
```

| Argument | Description |
|---|---|
| `input.cu` | CUDA source file (required) |
| `-o DIR` / `--output-dir DIR` | Where to write all outputs (optional) |
| `nvcc_args` | Extra flags forwarded verbatim to `nvcc` (after `--`) |

If `-o` is omitted, outputs land in `.cuda-demystify/<YYYYMMDD_HHMMSS>/`.

### Examples

```bash
python3 compile.py vector_add.cu
python3 compile.py vector_add.cu -o ./run-01
python3 compile.py vector_add.cu -- -arch=sm_86 -O2
python3 compile.py vector_add.cu -o ./run-01 -- -arch=sm_80 -lineinfo -G
```

## Output layout

```
<outdir>/
  dryrun.sh                   # bash: run nvcc --dryrun with the same args
  dryrun.log                  # raw --dryrun output (all sub-commands listed)
  step_01_preprocess/
    run.sh                    # bash: replay this step in isolation
    output.log                # stdout + stderr from this step
    *.cpp4.ii                 # files produced by this step
  step_02_cudafe/
    run.sh  output.log
    *.cudafe1.cpp  *.module_id
  step_03_preprocess/
    run.sh  output.log  *.cpp1.ii
  step_04_cicc/
    run.sh  output.log
    *.ptx  *.cudafe1.gpu  *.cudafe1.stub.c  *.cudafe1.c
  step_05_ptxas/
    run.sh  output.log  *.sm_NN.cubin
  step_06_fatbinary/
    run.sh  output.log  *.fatbin.c
  step_07_host_compile/
    run.sh  output.log  *.o
  step_08_nvlink/
    run.sh  output.log  *dlink.sm_NN.cubin  *dlink.reg.c
  step_09_fatbinary/
    run.sh  output.log  *dlink.fatbin.c
  step_10_host_compile/
    run.sh  output.log  *dlink.o
  step_11_host_link/
    run.sh  output.log
  executable
  tmp/                        # all intermediate files (full set)
```

## Pipeline and intermediate files

```
nvcc --dryrun  →  dryrun.log  (the map: all sub-commands nvcc would run)

  Step 1   gcc -E       preprocessing pass 1    → *.cpp4.ii
  Step 2   cudafe++     CUDA front-end           → *.cudafe1.cpp  *.module_id
  Step 3   gcc -E       preprocessing pass 2    → *.cpp1.ii
  Step 4   cicc         CUDA C++ → PTX           → *.ptx  *.cudafe1.gpu
  Step 5   ptxas        PTX → CUBIN              → *.sm_NN.cubin
  Step 6   fatbinary    PTX+CUBIN → fat binary   → *.fatbin.c
  Step 7   gcc -c       compile host C++         → *.o
  Step 8   nvlink       device-side linking      → *dlink.sm_NN.cubin  *dlink.reg.c
  Step 9   fatbinary    device-link fat binary   → *dlink.fatbin.c
  Step 10  gcc -c       compile link stub        → *dlink.o
  Step 11  g++          host link                → executable
```

| Extension | Stage | What it is |
|---|---|---|
| `*.cpp4.ii` | gcc -E pass 1 | Preprocessed CUDA source (device) |
| `*.cpp1.ii` | gcc -E pass 2 | Preprocessed CUDA source (device, second pass) |
| `*.cudafe1.cpp` | cudafe++ | Host C++ with device code replaced by stubs |
| `*.cudafe1.stub.c` | cudafe++ | Host-side stubs for device functions |
| `*.cudafe1.gpu` | cicc | Device NVVM IR after cudafe |
| `*.cudafe1.c` | cicc | Device C stub after cicc |
| `*.ptx` | cicc | PTX: GPU virtual ISA (human-readable) |
| `*.sm_NN.cubin` | ptxas | CUBIN: GPU machine code for SM version NN |
| `*.fatbin.c` | fatbinary | Fat binary encoded as a C source array |
| `*dlink.reg.c` | nvlink | Device-link registration C source |
| `*dlink.o` | gcc -c | Device-link object compiled from the above |
| `*.o` | gcc -c | Host object file |
