# Actor Debug Environment

This directory provides an isolated debugging environment for the OpenRLHF actor module, focused on replaying real training inputs to debug compilation and performance issues.

## Core Components

The primary debugging workflow centers around two key scripts:

### **`replay_inputs.py`** - Main debugging script
Replays captured training inputs through a standalone actor instance with comprehensive testing options including forward/backward passes, compilation modes, layer count modification, and multi-GPU simulation.

### **`init_actor_only.py`** - Actor initialization dependency  
Provides standalone actor initialization without Ray dependencies, supporting various configurations like custom layer counts, compilation modes, and DeepSpeed integration.

```bash
# Basic replay of first 5 inputs with backward pass
deepspeed --num_gpus=1 replay_inputs.py --max-inputs 5

# Test compilation modes (Flash Attention compatibility)
deepspeed --num_gpus=1 replay_inputs.py --compile --no-packing --max-inputs 5
deepspeed --num_gpus=1 replay_inputs.py --compile --no-flash-attn --max-inputs 5

# Test with reduced model size for memory optimization
deepspeed --num_gpus=1 replay_inputs.py --num-layers 2 --max-inputs 5

# Multi-GPU padding simulation with shuffled inputs
deepspeed --num_gpus=1 replay_inputs.py --shuffle-inputs --max-inputs 10
```

### Core Replay Options

```bash
# Compilation Testing (Flash Attention Compatibility)
deepspeed --num_gpus=1 replay_inputs.py --compile --no-packing --max-inputs 5    # Flash Attention with compilation
deepspeed --num_gpus=1 replay_inputs.py --compile --no-flash-attn --max-inputs 5 # SDPA fallback 
deepspeed --num_gpus=1 replay_inputs.py --deepcompile --no-flash-attn --max-inputs 5 # DeepCompile mode

# Memory Optimization with Layer Count Reduction  
deepspeed --num_gpus=1 replay_inputs.py --num-layers 2 --max-inputs 5   # ~8GB instead of ~45GB
deepspeed --num_gpus=1 replay_inputs.py --num-layers 4 --max-inputs 5   # Medium size model
deepspeed --num_gpus=1 replay_inputs.py --num-layers 8 --max-inputs 5   # Larger but still reduced

# DeepSpeed ZeRO Stage Testing
deepspeed --num_gpus=1 replay_inputs.py --zero-stage 0 --max-inputs 5   # No optimization (baseline)
deepspeed --num_gpus=1 replay_inputs.py --zero-stage 2 --max-inputs 5   # Optimizer + gradient partitioning
deepspeed --num_gpus=1 replay_inputs.py --zero-stage 3 --max-inputs 5   # Full partitioning

# Multi-GPU Simulation and Padding Issues
deepspeed --num_gpus=1 replay_inputs.py --shuffle-inputs --max-inputs 10   # Creates sequence length mismatches
deepspeed --num_gpus=1 replay_inputs.py --shuffle-inputs --random-seed 123 --max-inputs 10

# Forward-Only Testing (Skip Backward Pass)
deepspeed --num_gpus=1 replay_inputs.py --no-backward --max-inputs 5
deepspeed --num_gpus=1 replay_inputs.py --no-backward --compile --no-packing --max-inputs 5

# Process All Available Inputs (1000+ files)
deepspeed --num_gpus=1 replay_inputs.py --quiet   # All inputs, minimal output
```

## Key Features

### **Input Replay (`replay_inputs.py`)**

The main debugging script with comprehensive options:

**Core Capabilities:**
- **1000+ Real Training Inputs**: Pre-collected inputs from actual training runs ready for replay
- **Forward + Backward Pass**: Full gradient computation with configurable loss functions  
- **Compilation Testing**: PyTorch compile and DeepSpeed DeepCompile with Flash Attention compatibility
- **Memory Optimization**: Custom layer counts to reduce model size from 8B to ~1.5B parameters
- **Multi-GPU Simulation**: Shuffled inputs create sequence length mismatches to test padding behavior
- **Performance Measurement**: Detailed timing for forward/backward passes and memory usage

**Key Arguments:**
- `--max-inputs N`: Limit number of inputs to replay (useful for quick testing)
- `--num-layers N`: Custom layer count (2 layers = ~8GB instead of ~45GB memory)
- `--compile`: Enable PyTorch compilation (use with `--no-packing` for Flash Attention)
- `--no-flash-attn`: Use SDPA instead of Flash Attention (compilation-safe fallback)
- `--shuffle-inputs`: Create sequence length mismatches across ranks (tests padding)
- `--no-backward`: Skip backward pass for forward-only testing
- `--zero-stage [0-3]`: Test different DeepSpeed ZeRO optimization levels

### **Actor Initialization (`init_actor_only.py`)**

Dependency script that provides standalone actor initialization:

**Configuration:**
- Model: `meta-llama/Llama-3.1-8B-Instruct` 
- DeepSpeed ZeRO Stage 1 (configurable)
- Flash Attention enabled (CUDA only)
- BF16 precision enabled
- Custom layer count support via monkey patching

## Debugging Workflow

### **Primary Workflow (Recommended)**

**1000+ real training inputs are already available** - no setup needed:

```bash
# Start with basic replay to verify functionality 
deepspeed --num_gpus=1 replay_inputs.py --max-inputs 5

# Test compilation issues with Flash Attention
deepspeed --num_gpus=1 replay_inputs.py --compile --no-packing --max-inputs 5
deepspeed --num_gpus=1 replay_inputs.py --compile --no-flash-attn --max-inputs 5

# Test memory optimization with reduced model size
deepspeed --num_gpus=1 replay_inputs.py --num-layers 2 --max-inputs 5

# Analyze results in logs/ directory
```

### **Advanced Debugging Scenarios**

```bash
# Multi-GPU padding issues reproduction
deepspeed --num_gpus=1 replay_inputs.py --shuffle-inputs --max-inputs 10

# DeepSpeed ZeRO stage comparison
deepspeed --num_gpus=1 replay_inputs.py --zero-stage 0 --max-inputs 5  # Baseline
deepspeed --num_gpus=1 replay_inputs.py --zero-stage 3 --max-inputs 5  # Full optimization

# Forward-only performance testing
deepspeed --num_gpus=1 replay_inputs.py --no-backward --compile --max-inputs 10

# Process all available inputs for comprehensive testing
deepspeed --num_gpus=1 replay_inputs.py --quiet
```

## Troubleshooting

### Common Issues

**GPU Memory Errors**
- Use `--num-layers 2` to reduce memory from ~45GB to ~8GB
- Try `--zero-stage 3` for maximum memory optimization
- Start with `--max-inputs 1` for minimal memory testing

**Compilation Failures with Flash Attention**
- Use `--compile --no-packing` (enables compilation-compatible flash_attn_func)
- Use `--compile --no-flash-attn` (falls back to SDPA)
- Avoid `--compile` with default packing (known incompatibility)

**DeepSpeed Launcher Required**
- Always use: `deepspeed --num_gpus=1 replay_inputs.py`
- The `--local_rank=0` argument is automatically added by DeepSpeed
- Direct Python execution will fail with communication backend errors

## Flash Attention + Compilation Analysis

### **Root Cause: Function Path Incompatibility**

OpenRLHF's packing approach uses `flash_attn_varlen_func` which fails compilation due to FakeTensor handling in C++ bindings.

**Solutions tested in this environment:**
- `--compile --no-packing`: Uses compilation-compatible `flash_attn_func` 
- `--compile --no-flash-attn`: Falls back to SDPA (always works)
- `--deepcompile --no-flash-attn`: DeepSpeed DeepCompile with SDPA

### **Verified Working Configurations**
```bash
# Flash Attention + Compilation (no packing required)
deepspeed --num_gpus=1 replay_inputs.py --compile --no-packing --max-inputs 5

# SDPA Fallback (most reliable)  
deepspeed --num_gpus=1 replay_inputs.py --compile --no-flash-attn --max-inputs 5

# DeepCompile mode
deepspeed --num_gpus=1 replay_inputs.py --deepcompile --no-flash-attn --max-inputs 5
```

## Technical Details

**Memory Usage:**
- Full model (32 layers): ~45-75GB GPU memory
- Reduced model (2 layers): ~8GB GPU memory (0.19x reduction)
- Parameter reduction: 8B → 1.5B parameters

**Available Data:** 
- 1000+ real training input files from actual OpenRLHF runs
- Multi-rank inputs supporting distributed training simulation
- Comprehensive metadata including timestamps, batch sizes, sequence lengths

**DeepSpeed Integration:**
- All ZeRO stages (0-3) supported and tested
- Proper collective communication backend initialization
- Multi-GPU simulation via input shuffling

**Performance Measurement:**
- Separate forward/backward pass timing
- CUDA synchronization for accurate measurements  
- Detailed logging saved to `logs/replay_results_*.pkl`