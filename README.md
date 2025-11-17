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
# TensorRT AOT (torch.export path with static shapes)
python build_trt_engine_eager.py \
  --checkpoint ./bf16_checkpoints \
  --precision bf16 \
  --static_b 2 \
  --static_s 4096 \
  --num_sample 100 \
  --report_memory \
  --save_ts modernbert_trt_static.ts

# TensorRT JIT (torch.compile backend with dynamic shapes)
python bench_tensorrt.py \
  --checkpoint ./bf16_checkpoints \
  --precision bf16 \
  --attn_impl sdpa \
  --batch_sizes 2 4 \
  --max_lengths 2048 3072 4096 \
  --num_sample 100
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
- **Post ms**: Post-processing latency (argmax, softmax, label mapping)
- **E2E ms**: End-to-end latency (tokenization + inference + post-processing)
- **Throughput**: Tokens processed per second

## Compilation Strategies

### 1. Torch-TensorRT AOT (torch.export)

Ahead-of-time compilation via `torch.export` + `torch_tensorrt.dynamo.compile()`. Best for production deployment with fixed shapes.

**Key characteristics:**
- Requires `attn_implementation="eager"` (SDPA) - FA2 not supported
- Static shapes only - no dynamic batching
- Full graph compilation with `require_full_compilation=True`
- Produces serializable TorchScript artifact for deployment
- Set `reference_compile=False` to avoid double-tracing issues

**Static shape specification:**
```python
# For static shapes, use simple tuple
torch_tensorrt.Input(
    (batch_size, seq_len),
    dtype=torch.long
)
```

**For dynamic shapes (not recommended for AOT):**
```python
# Use min/opt/max ranges
torch_tensorrt.Input(
    min_shape=(2, 512),
    opt_shape=(8, 1024),
    max_shape=(8, 2048),
    dtype=torch.bfloat16
)
```

Never specify both `shape` and `min_shape/opt_shape/max_shape` — choose static or dynamic, not both.

**Compilation settings:**
```python
trt_gm = torch_tensorrt.dynamo.compile(
    exported_program,
    inputs=trt_inputs,
    device=torch_tensorrt.Device("cuda:0"),
    enabled_precisions={torch.bfloat16},
    require_full_compilation=True,
    optimization_level=5,
    workspace_size=(2 << 30),  # 2GB
    min_block_size=5,
    use_fast_partitioner=True,
    enable_experimental_decompositions=True,
    cache_built_engines=True,
    reuse_cached_engines=True,
)
```

### 2. Torch-Compile JIT (torch.compile + TensorRT Backend)

Just-in-time compilation with `torch.compile(model, backend="torch_tensorrt")`. More flexible than AOT, allows hybrid fallback where unsupported ops run in PyTorch.

**Pros:** Works with FlashAttention2, handles dynamic shapes, no pre-export needed
**Cons:** Partial TensorRT coverage, graph breaks at unsupported ops, runtime compilation overhead

**Configuration:**
```python
compiled = torch.compile(
    model, 
    backend="torch_tensorrt",
    dynamic=True,
    options={
        "device": torch_tensorrt.Device("cuda:0"),
        "enabled_precisions": {torch.bfloat16},
        "optimization_level": 5,
        "min_block_size": 16,
        "keep_fa2": True,  # Keep Flash Attention on PyTorch side
    }
)
```

For experimentation, use `torch._dynamo.config.suppress_errors = True` to silently fall back to eager execution on tracing failures.

### 3. Torch.compile (Inductor Backend)

Pure PyTorch compilation without TensorRT. Best compatibility with FA2 kernels and delivers the best performance.

**Mode:** `max-autotune-no-cudagraphs` (CUDA Graphs disabled due to incompatibilities)

**Configuration:**
```python
compiled = torch.compile(
    model,
    backend="inductor",
    mode="max-autotune-no-cudagraphs",
    dynamic=False  # Static shapes for best performance
)
```

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

| Batch Size | Seq Length | Tokenization Latency (ms) | Inference Latency (ms) | Post-processing Latency (ms) | End-to-End Latency (ms) | Throughput (tokens/sec) |
|------------|------------|---------------------------|------------------------|------------------------------|-------------------------|-------------------------|
| 2          | 2048       | 7.1                       | 34.1                   | 1.3                          | 42.5                    | 96,351                  |
| 2          | 3072       | 10.2                      | 45.0                   | 0.2                          | 55.4                    | 110,813                 |
| 2          | 4096       | 13.6                      | 63.5                   | 0.3                          | 77.4                    | 105,825                 |
| 2          | 8192       | 25.3                      | 148.8                  | 0.3                          | 174.4                   | 93,919                  |
| 4          | 2048       | 10.8                      | 56.6                   | 0.3                          | 67.7                    | 120,916                 |
| 4          | 3072       | 16.6                      | 87.6                   | 0.3                          | 104.5                   | 117,571                 |
| 4          | 4096       | 21.0                      | 121.9                  | 0.3                          | 143.3                   | 114,354                 |
| 4          | 8192       | 42.0                      | 290.5                  | 0.3                          | 332.9                   | 98,447                  |

### Torch.compile (Inductor + Flash Attention 2)

static shapes with warmup, FA2 kernels:

| Engine                  | Batch Size | Seq Length | Tokenization Latency (ms) | Inference Latency (ms) | Post-processing Latency (ms) | End-to-End Latency (ms) | Throughput (tokens/sec) |
|-------------------------|------------|------------|---------------------------|------------------------|------------------------------|-------------------------|-------------------------|
| torch.compile[inductor] | 2          | 2048       | 7.1                       | 26.1                   | 0.5                          | 33.8                    | 121,290                 |
| torch.compile[inductor] | 2          | 3072       | 10.4                      | 40.6                   | 0.2                          | 51.3                    | 119,722                 |
| torch.compile[inductor] | 2          | 4096       | 13.5                      | 55.6                   | 0.2                          | 69.4                    | 117,960                 |
| torch.compile[inductor] | 2          | 8192       | 25.4                      | 134.0                  | 0.3                          | 159.7                   | 102,574                 |
| torch.compile[inductor] | 4          | 2048       | 10.8                      | 48.3                   | 0.2                          | 59.4                    | 137,844                 |
| torch.compile[inductor] | 4          | 3072       | 16.2                      | 76.0                   | 0.2                          | 92.6                    | 132,644                 |
| torch.compile[inductor] | 4          | 4096       | 21.2                      | 106.6                  | 0.3                          | 128.1                   | 127,856                 |
| torch.compile[inductor] | 4          | 8192       | 41.9                      | 262.1                  | 0.3                          | 304.4                   | 107,638                 |

**Note:** Significantly improved inference latency with FA2 compared to eager attention baseline.

### Torch.compile (TensorRT JIT Backend + Flash Attention 2)

JIT compilation with torch_tensorrt backend:

| Engine                        | Batch Size | Seq Length | Tokenization Latency (ms) | Inference Latency (ms) | Post-processing Latency (ms) | End-to-End Latency (ms) | Throughput (tokens/sec) |
|-------------------------------|------------|------------|---------------------------|------------------------|------------------------------|-------------------------|-------------------------|
| torch.compile[torch_tensorrt] | 2          | 2048       | 7.1                       | 33.9                   | 0.3                          | 41.3                    | 99,118                  |
| torch.compile[torch_tensorrt] | 2          | 3072       | 10.4                      | 46.8                   | 0.2                          | 57.5                    | 106,895                 |
| torch.compile[torch_tensorrt] | 2          | 4096       | 13.2                      | 65.6                   | 0.2                          | 79.2                    | 103,479                 |
| torch.compile[torch_tensorrt] | 2          | 8192       | 26.2                      | 151.5                  | 0.2                          | 178.1                   | 91,971                  |
| torch.compile[torch_tensorrt] | 4          | 2048       | 11.2                      | 57.6                   | 0.2                          | 69.1                    | 118,617                 |
| torch.compile[torch_tensorrt] | 4          | 3072       | 16.9                      | 88.2                   | 0.2                          | 105.6                   | 116,412                 |
| torch.compile[torch_tensorrt] | 4          | 4096       | 21.9                      | 122.5                  | 0.2                          | 144.8                   | 113,182                 |
| torch.compile[torch_tensorrt] | 4          | 8192       | 43.9                      | 290.8                  | 0.3                          | 335.1                   | 97,774                  |

**Note:** TensorRT JIT backend shows similar performance to eager baseline, suggesting limited TensorRT optimization with FA2 kernels.

### TensorRT AOT (Static Graph via torch.export)

Ahead-of-time compilation with static shapes using torch.export + TensorRT dynamo backend:

| Engine                           | Batch Size | Seq Length | Tokenization Latency (ms) | Inference Latency (ms) | Post-processing Latency (ms) | End-to-End Latency (ms) | Throughput (tokens/sec) |
|----------------------------------|------------|------------|---------------------------|------------------------|------------------------------|-------------------------|-------------------------|
| torch_tensorrt[dynamo][static]   | 2          | 4096       | 14.4                      | 82.9                   | 0.3                          | 97.7                    | 83,768                  |

**Note:** TensorRT AOT with static shapes requires SDPA attention (no FA2 support). While it produces a fully compiled engine, the lack of Flash Attention optimization results in slower inference compared to Inductor + FA2, especially at longer sequence lengths where quadratic attention becomes a bottleneck.

### SageAttention + Torch.compile (No Attention Compilation)

Using SageAttention with torch.compile but excluding attention from Triton compilation:

| Engine                | Batch Size | Seq Length | Tokenization Latency (ms) | Inference Latency (ms) | Post-processing Latency (ms) | End-to-End Latency (ms) | Throughput (tokens/sec) |
|-----------------------|------------|------------|---------------------------|------------------------|------------------------------|-------------------------|-------------------------|
| Sage+Compile(no-attn) | 2          | 2048       | 6.7                       | 37.4                   | 0.4                          | 44.6                    | 91,799                  |
| Sage+Compile(no-attn) | 2          | 3072       | 10.8                      | 62.9                   | 0.2                          | 74.1                    | 82,931                  |
| Sage+Compile(no-attn) | 2          | 4096       | 13.5                      | 87.4                   | 0.2                          | 101.3                   | 80,904                  |
| Sage+Compile(no-attn) | 2          | 8192       | 28.0                      | 221.2                  | 0.3                          | 249.6                   | 65,636                  |
| Sage+Compile(no-attn) | 4          | 2048       | 11.3                      | 73.1                   | 0.3                          | 84.8                    | 96,593                  |
| Sage+Compile(no-attn) | 4          | 3072       | 17.3                      | 113.8                  | 0.3                          | 131.5                   | 93,437                  |
| Sage+Compile(no-attn) | 4          | 4096       | 22.5                      | 164.2                  | 0.3                          | 187.1                   | 87,577                  |
| Sage+Compile(no-attn) | 4          | 8192       | 46.2                      | 430.3                  | 0.3                          | 477.0                   | 68,693                  |

## Results Files

Benchmark results are saved as JSON:
- `benchmark_results_sage.json`
- `bench_inductor_results.json`
- `bench_tensorrt_results.json`
- `bench_trt_static_results.json`
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
