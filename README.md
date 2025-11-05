# ModernBERT Inference Benchmarks

Performance benchmarking suite for ModernBERT with multiple attention backends and compilation strategies, including torch.compile (Inductor), TensorRT AOT, and TensorRT hybrid graph compilation.

## Summary

Best numbers so far are from FA2 + pytorch inductor backend  with static shapes and warmup to cache.

- Most dynamic-capture compiler frontends can’t keep the whole model in a single graph because FA2 kernels plus data-dependent (ragged) shapes trigger graph breaks. As a result, attention runs via eager fallbacks (FA2 stays as an external CUDA op), and only the surrounding blocks are compiled.
- TensorRT AOT would further optimize the model if FA2 were lowerable into the engine. TRT-LLM has an attention plugin, but ModernBERT can’t use that path today, so FA2 remains outside.
- Using the TensorRT backend via PyTorch (torch.compile) still relies on runtime capture/partitioning, so you don’t get a single static AOT engine without dropping FA2 support.
- In practice, TRT AOT can deliver strong speedups on the non-attention parts (fusing GEMMs, epilogues, norms). But without FA2, it falls back to quadratic attention, which scales poorly at longer sequence lengths—often making it slower overall than the FA2 path despite those layer-wise optimizations.

### trt vs inductor jit difference in perf

Why Inductor beats TRT JIT here:

Inductor emits a single compiled Torch graph where FA2 is one extern_call, but surrounding ops (permutes, bias/add, residuals, some LN epilogues) still get fused into larger Triton/CUDA kernels within the same module.
TensorRT JIT, by contrast, partitions by operator support. Because FA2 isn’t lowered, you get more TRT↔PyTorch handoffs (engine launches + casts/contigs at boundaries), which are expensive at long sequence lengths.

Fewer runtime boundaries: One compiled Torch graph with a single FA2 extern_call vs many TRT partitions around unsupported ops.

Better fusion adjacent to FA2: Inductor fuses prologue/epilogue ops into larger kernels; TRT JIT often leaves them as tiny islands or forces format casts.

Lower dtype/layout thrash: Inductor keeps BF16/FP32 accumulation consistent; TRT edges frequently introduce contiguous()/to() overhead.

Result: Lower launch count + less memory traffic → better E2E latency and higher tok/s at 2–4 batch, 2k–4k seq.


## Setup

### Prerequisites

```bash
# Create virtual environment
python3 -m venv flash-attn
source flash-attn/bin/activate

# Install PyTorch 2.8.0
pip install torch==2.8.0
```

### Install Flash Attention 2

```bash
pip install flash-attn --no-build-isolation
```

Or build from source:
```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install
```

### Install SageAttention

```bash
pip install -U sageattention
```

### Install TensorRT

```bash
pip install torch-tensorrt
```

### Install Dependencies

```bash
pip install transformers datasets numpy tqdm ninja packaging accelerate
```

### Model Checkpoint

Download or prepare ModernBERT checkpoint in `bf16_checkpoints/` directory. Ensure checkpoints are in bfloat16 precision for optimal TensorRT performance.

## Benchmarks

### Baseline (Eager Execution)

```bash
# Eager PyTorch baseline with Flash Attention 2
python benchmark_modernbert.py \
  --checkpoint ./bf16_checkpoints \
  --precision bf16 \
  --batch_sizes 2 4 \
  --max_lengths 2048 3072 4096 \
  --num_sample 100
```

### SageAttention vs Flash Attention 2

```bash
# SageAttention (Triton kernels)
python bench_sage.py \
  --checkpoint ./bf16_checkpoints \
  --precision bf16 \
  --batch_sizes 2 4 \
  --max_lengths 2048 3072 4096 \
  --num_sample 100

# Flash Attention 2 with torch.compile (Inductor backend)
python bench_inductor.py \
  --checkpoint ./bf16_checkpoints \
  --precision bf16 \
  --attn_impl flash_attention_2 \
  --mode max-autotune-no-cudagraphs \
  --num_sample 100
```

### TensorRT Compilation

```bash
# TensorRT AOT (torch.export path)
python build_trt_engine_eager.py

# TensorRT Hybrid Graph (torch.compile backend)
python tensorrt-hybrid_graph.py
```

## Scripts

- `benchmark_modernbert.py` - Baseline eager execution benchmarks with PyTorch
- `bench_sage.py` - SageAttention benchmarks
- `bench_inductor.py` - torch.compile + Flash Attention 2 benchmarks
- `bench_tensorrt.py` - TensorRT engine benchmarks
- `build_trt_engine_eager.py` - Build TensorRT engine via AOT export #test
- `tensorrt-hybrid_graph.py` - TensorRT hybrid graph compilation
- `profile_sage.py` - Detailed profiling with Chrome traces


## Attention Implementations

ModernBERT supports multiple attention backends, each with different performance characteristics and compilation compatibility:

### Flash Attention 2 (FA2)

CUDA-optimized attention kernel with O(N) memory complexity. Default choice for production inference.

**Characteristics:**
- Fused attention operations in a single CUDA kernel
- Supports variable sequence lengths via unpadding
- Works with torch.compile (Inductor) as an extern call
- Not lowerable to TensorRT engines (remains eager fallback)
- Best performance at long sequences (2k-4k tokens)

**Usage:**
```python
model = ModernBertForMaskedLM.from_pretrained(
    checkpoint,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

### SageAttention

Triton-based attention implementation with custom kernel optimizations.

**Characteristics:**
- Pure Triton kernels (no external CUDA dependencies)
- Can be excluded from torch.compile for hybrid execution
- Competitive performance on shorter sequences
- More flexible for experimental compilation paths

**Usage:**
```python
from sageattention import sageattn
# Applied via model config or direct kernel replacement
```

### SDPA (Scaled Dot-Product Attention)

PyTorch native attention with automatic backend selection (FlashAttention, memory-efficient, or math).

**Characteristics:**
- Fallback when FA2 unavailable
- Automatic kernel selection based on input shapes
- Full TensorRT compatibility (lowers to TRT attention ops)
- Slower than FA2 at long sequences due to quadratic memory

**Benchmarking Methodology:**

All attention benchmarks use:
- Batch sizes: 2, 4
- Sequence lengths: 2048, 3072, 4096 tokens
- 100 warmup iterations for stable timing
- BF16 precision throughout
- Static shapes where possible to maximize compilation benefits

Metrics reported:
- **Tok ms**: Tokenization latency
- **Infer ms**: Model forward pass latency
- **E2E ms**: End-to-end latency (tokenization + inference)
- **Throughput**: Tokens processed per second

## Compilation Strategies

### 2. Torch-TensorRT AOT (torch.export)

Ahead-of-time compilation via `torch.export` + `torch_tensorrt.dynamo.compile()`. Works best in strict mode with `require_full_compilation=True`.

**Key settings:**
- Set `reference_compile=False` to avoid double-tracing issues
- Use realistic dynamic ranges: `min_shape=(2, 512), opt_shape=(8, 1024), max_shape=(8, 2048)`
- Enable `enabled_precisions={torch.bfloat16}` for BF16 precision
- Narrower shape ranges yield faster engines

**TensorRT input specification:**
```python
torch_tensorrt.Input(
    min_shape=(2, 512),
    opt_shape=(8, 1024),
    max_shape=(8, 2048),
    dtype=torch.bfloat16
)
```

Never specify both `shape` and `min_shape/opt_shape/max_shape` — choose static or dynamic, not both.

## Aggressive TensorRT Optimization

Recommended settings for maximum performance:

```python
trt_model = torch_tensorrt.dynamo.compile(
    exported_program,
    inputs=[...],
    enabled_precisions={torch.bfloat16},
    require_full_compilation=True,
    optimization_level=5,
    assume_dynamic_shape_support=True,
    use_fast_partitioner=True,
    enable_experimental_decompositions=True,
    cache_built_engines=True,
    reuse_cached_engines=True,
    workspace_size=8 << 30,  # 8GB
    sparse_weights=True,
    hardware_compatible=True,
    debug=False
)
```

### 3. Torch-Compile Hybrid Graph (torch.compile + TensorRT Backend)

Just-in-time compilation with `torch.compile(model, backend="torch_tensorrt")`. More flexible than AOT, allows hybrid fallback where unsupported ops run in PyTorch.

**Pros:** Works with FlashAttention2, handles dynamic shapes
**Cons:** Partial TensorRT coverage, graph breaks at unsupported ops

**Configuration:**
```python
compiled = torch.compile(
    model, 
    backend="torch_tensorrt",
    dynamic=True
)
```

For experimentation, use `torch._dynamo.config.suppress_errors = True` to silently fall back to eager execution on tracing failures.

### 4. Torch.compile (Inductor Backend)

Pure PyTorch compilation without TensorRT. Best compatibility with FA2 kernels.

**Mode:** `max-autotune-no-cudagraphs` (CUDA Graphs disabled due to incompatibilities)

## CUDA Graphs Limitations

ModernBERT's FlashAttention2 path has two CUDA Graphs incompatibilities:

1. **Host read in unpadding logic:** `seqlens_in_batch.max().item()` performs a host read that prevents stable graph capture. Inductor skips CUDA Graphs for this region.

2. **RoPE cache mutation:** FA2's rotary embedding updates persistent buffers (`_cos_cached`, `_sin_cached`) across iterations, triggering "write-after-use" errors in CUDA Graphs.

**Workaround:** Use `max-autotune-no-cudagraphs` mode or call `torch.compiler.cudagraph_mark_step_begin()` to mark safe mutation points.

## BF16 Precision

ModernBERT checkpoints are stored in bfloat16. Use BF16 throughout the pipeline:
- Matches model weights
- Retains FP32 dynamic range (avoids FP16 overflows)
- Leverages Tensor Cores on Ampere+ GPUs (A10G, A100, L4, H100)


## Benchmark Results

### Baseline (Eager Execution)

| Batch | Seq  | Tokenization time ms | Infer ms | E2E ms | Throughput (tok/s) |
|-------|------|--------|----------|--------|--------------------|
| 2     | 2048 | 6.5    | 33.7     | 40.7   | 100,764            |
| 2     | 3072 | 9.7    | 44.7     | 54.5   | 112,688            |
| 2     | 4096 | 12.5   | 63.2     | 75.9   | 107,901            |
| 4     | 2048 | 9.0    | 56.4     | 65.6   | 124,824            |
| 4     | 3072 | 13.3   | 87.4     | 100.9  | 121,740            |
| 4     | 4096 | 17.2   | 121.8    | 139.3  | 117,605            |

### Torch.compile (Inductor + Flash Attention 2)

static shapes with warmup, FA2 kernels:

| Engine                  | Batch | Seq  | Tokenization time ms | Infer ms | E2E ms | Throughput (tok/s) |
|-------------------------|-------|------|--------|----------|--------|--------------------|
| torch.compile[inductor] | 2     | 2048 | 6.6    | 25.6     | 32.7   | 125,382            |
| torch.compile[inductor] | 2     | 3072 | 9.7    | 40.3     | 50.2   | 122,309            |
| torch.compile[inductor] | 2     | 4096 | 12.4   | 55.2     | 67.9   | 120,727            |
| torch.compile[inductor] | 4     | 2048 | 9.1    | 47.9     | 57.2   | 143,162            |
| torch.compile[inductor] | 4     | 3072 | 13.4   | 75.7     | 89.4   | 137,474            |
| torch.compile[inductor] | 4     | 4096 | 17.5   | 106.9    | 124.7  | 131,398            |

**Note:** Significantly improved inference latency with FA2 compared to eager attention baseline.

### Torch.compile (TensorRT Backend + Flash Attention 2)

JIT compilation with torch_tensorrt backend:

| Engine                        | Batch | Seq  | Tokenization time ms | Infer ms | E2E ms | Throughput (tok/s) |
|-------------------------------|-------|------|--------|----------|--------|--------------------|
| torch.compile[torch_tensorrt] | 2     | 2048 | 6.6    | 33.9     | 40.9   | 100,241            |
| torch.compile[torch_tensorrt] | 2     | 3072 | 9.7    | 46.8     | 56.7   | 108,345            |
| torch.compile[torch_tensorrt] | 2     | 4096 | 12.4   | 65.2     | 77.9   | 105,214            |
| torch.compile[torch_tensorrt] | 4     | 2048 | 9.7    | 58.2     | 68.1   | 120,215            |
| torch.compile[torch_tensorrt] | 4     | 3072 | 14.0   | 87.8     | 102.2  | 120,258            |
| torch.compile[torch_tensorrt] | 4     | 4096 | 18.4   | 122.2    | 140.9  | 116,245            |

**Note:** TensorRT backend shows similar performance to eager baseline, suggesting limited TensorRT optimization with FA2 kernels.

### SageAttention + Torch.compile (No Attention Compilation)

Using SageAttention with torch.compile but excluding attention from Triton compilation:

| Engine                | Batch | Seq  | Tokenization time ms | Infer ms | E2E ms | Throughput (tok/s) |
|-----------------------|-------|------|--------|----------|--------|--------------------|
| Sage+Compile(no-attn) | 2     | 2048 | 6.6    | 37.2     | 44.2   | 92,610             |
| Sage+Compile(no-attn) | 2     | 3072 | 9.8    | 62.7     | 72.7   | 84,456             |
| Sage+Compile(no-attn) | 2     | 4096 | 13.0   | 87.4     | 100.6  | 81,429             |
| Sage+Compile(no-attn) | 4     | 2048 | 9.5    | 73.0     | 82.8   | 98,905             |
| Sage+Compile(no-attn) | 4     | 3072 | 14.1   | 113.9    | 128.3  | 95,796             |
| Sage+Compile(no-attn) | 4     | 4096 | 18.4   | 164.3    | 183.1  | 89,483             |

## Results Files

Benchmark results are saved as JSON:
- `benchmark_results_sage.json`
- `bench_inductor_results.json`
- `bench_tensorrt_results.json`
- `benchmark_results.json`

Profiling traces:
- `trace_sage_compiled.json`
- `trace_fa2_compiled.json`
- `trace_inductor_compiled.json`
- `inductor_fa2.nsys-rep`
- `sage.nsys-rep`

View Chrome traces at `chrome://tracing` or https://ui.perfetto.dev/

## Key Takeaways

- **torch.compile (Inductor) + FA2** delivers the best inference performance with dynamic shape support and excellent warmup behavior (125k-143k tok/s throughput)
- **torch.compile (TensorRT backend)** performs similarly to eager baseline, indicating FA2 kernels aren't being optimized by TensorRT in JIT mode
- **SageAttention + Compile (no-attn)** provides competitive performance when attention is excluded from Triton compilation (81k-99k tok/s)
- **Eager baseline** with FA2 shows solid performance (100k-125k tok/s) without compilation overhead
- ModernBERT's unpadding logic (`.item()` calls) prevents full CUDA Graphs and torch.export compatibility
- Always use BF16 precision to match checkpoint format and leverage Tensor Cores
- Narrow dynamic shape ranges in TensorRT yield faster, lighter engines
- FA2 with torch.compile (Inductor) shows the best inference latency across all configurations
