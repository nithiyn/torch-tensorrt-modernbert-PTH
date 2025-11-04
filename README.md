# ModernBERT Inference Benchmarks

Performance benchmarking suite for ModernBERT with multiple attention backends and compilation strategies, including torch.compile (Inductor), TensorRT AOT, and TensorRT hybrid graph compilation.

## Summary

Best numbers so far are from FA2 + pytorch inductor backend  with static shapes and warmup to cache.

- Most dynamic-capture compiler frontends can’t keep the whole model in a single graph because FA2 kernels plus data-dependent (ragged) shapes trigger graph breaks. As a result, attention runs via eager fallbacks (FA2 stays as an external CUDA op), and only the surrounding blocks are compiled.
- TensorRT AOT would further optimize the model if FA2 were lowerable into the engine. TRT-LLM has an attention plugin, but ModernBERT can’t use that path today, so FA2 remains outside.
- Using the TensorRT backend via PyTorch (torch.compile) still relies on runtime capture/partitioning, so you don’t get a single static AOT engine without dropping FA2 support.
- In practice, TRT AOT can deliver strong speedups on the non-attention parts (fusing GEMMs, epilogues, norms). But without FA2, it falls back to quadratic attention, which scales poorly at longer sequence lengths—often making it slower overall than the FA2 path despite those layer-wise optimizations.

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


## Compilation Strategies

### 2. Torch-TensorRT AOT (torch.export)

Ahead-of-time compilation via `torch.export` + `torch_tensorrt.dynamo.compile()`. Works best in strict mode with `require_full_compilation=True`.

**Key settings:**
- Set `reference_compile=False` to avoid double-tracing issues
- Use realistic dynamic ranges: `min_shape=(2, 512), opt_shape=(8, 1024), max_shape=(8, 2048)`
- Enable `enabled_precisions={torch.bfloat16}` for BF16 precision
- Narrower shape ranges yield faster engines

**Current limitation:** Fails on data-dependent shape expressions (`.item()` calls in `_unpad_modernbert_input`)

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

## Benchmark Results

### Baseline (Eager Execution)

| Batch | Seq  | Tok ms | Infer ms | E2E ms | Throughput (s/s) |
|-------|------|--------|----------|--------|------------------|
| 2     | 2048 | 7.5    | 32.8     | 40.9   | 48.92            |
| 2     | 3072 | 11.2   | 45.1     | 56.9   | 35.13            |
| 2     | 4096 | 14.2   | 63.6     | 78.4   | 25.51            |
| 4     | 2048 | 11.4   | 56.8     | 68.8   | 58.16            |
| 4     | 3072 | 17.1   | 87.7     | 105.4  | 37.95            |
| 4     | 4096 | 22.2   | 121.9    | 144.7  | 27.64            |

### Torch.compile (Inductor + Flash Attention 2)

Dynamic shapes with warmup, FA2 kernels:

| Engine                  | Batch | Seq  | Tok ms | Infer ms | E2E ms | Throughput (s/s) |
|-------------------------|-------|------|--------|----------|--------|------------------|
| torch.compile[inductor] | 2     | 2048 | 6.5    | 25.5     | 32.5   | 61.46            |
| torch.compile[inductor] | 2     | 3072 | 9.7    | 40.3     | 50.3   | 39.77            |
| torch.compile[inductor] | 2     | 4096 | 12.6   | 55.2     | 68.0   | 29.40            |
| torch.compile[inductor] | 4     | 2048 | 9.1    | 47.9     | 57.2   | 69.89            |
| torch.compile[inductor] | 4     | 3072 | 13.5   | 75.7     | 89.5   | 44.67            |
| torch.compile[inductor] | 4     | 4096 | 17.5   | 107.0    | 124.7  | 32.07            |

**Note:** Significantly improved inference latency with FA2 compared to eager attention baseline.

### SageAttention + Torch.compile (No Attention Compilation)

Using SageAttention with torch.compile but excluding attention from Triton compilation:

| Engine                    | Batch | Seq  | Tok ms | Infer ms | E2E ms | Throughput (s/s) |
|---------------------------|-------|------|--------|----------|--------|------------------|
| Sage+Compile(no-attn)     | 2     | 2048 | 6.5    | 37.2     | 44.1   | 45.32            |
| Sage+Compile(no-attn)     | 2     | 3072 | 10.0   | 62.6     | 72.9   | 27.43            |
| Sage+Compile(no-attn)     | 2     | 4096 | 13.1   | 87.3     | 100.7  | 19.85            |
| Sage+Compile(no-attn)     | 4     | 2048 | 9.8    | 72.9     | 83.0   | 48.18            |

### TensorRT AOT (Eager Attention Backend)

Batch sizes 2,4 with seq=2048, eager attention (no FA2):

| Engine   | Batch | Seq  | Tok ms | Infer ms | E2E ms | Throughput (s/s) |
|----------|-------|------|--------|----------|--------|------------------|
| TRT      | 2     | 2048 | 7.5    | 31.3     | 70.6   | 28.31            |
| TRT      | 4     | 2048 | 11.2   | 62.6     | 137.3  | 29.14            |
| Baseline | 2     | 2048 | 7.5    | 30.8     | 38.9   | 51.44            |
| Baseline | 4     | 2048 | 11.1   | 61.8     | 73.5   | 54.39            |

**Note:** TensorRT AOT shows lower throughput due to lack of FA2 kernel support and overhead from eager attention fallback.

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

- **torch.compile (Inductor) + FA2** delivers the best inference performance with dynamic shape support and excellent warmup behavior (61-70 s/s throughput for batch 2-4, seq 2048)
- **SageAttention + Compile (no-attn)** provides competitive performance when attention is excluded from Triton compilation
- **TensorRT AOT** is most stable but lacks FA2 support, resulting in lower throughput
- **TensorRT Hybrid Graph** offers flexibility with partial TensorRT coverage
- ModernBERT's unpadding logic (`.item()` calls) prevents full CUDA Graphs and torch.export compatibility
- Always use BF16 precision to match checkpoint format and leverage Tensor Cores
- Narrow dynamic shape ranges in TensorRT yield faster, lighter engines
- FA2 with torch.compile shows significantly better inference latency compared to eager attention baseline
