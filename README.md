# Actor Debug Setup

This directory contains tools for debugging the OpenRLHF actor module in isolation.

## Overview

The setup allows you to:
1. **Initialize only the actor module** without the full RLHF training setup
2. **Dump actor inputs** during real training runs
3. **Replay dumped inputs** to the actor for debugging

## Directory Structure

```
actor_debug/
├── README.md                    # This file
├── init_actor_only.py          # Initialize actor standalone
├── enable_input_dumping.py     # Setup input dumping capability
├── control_dumping.py          # Enable/disable input dumping
├── replay_inputs.py            # Replay dumped inputs to actor
├── setup_debug.sh              # Quick setup script
├── create_dummy_input.py       # Create dummy inputs for testing
├── actor_original_backup.py    # Backup of original actor.py
├── dumping_enabled.flag        # Flag file indicating dumping is enabled
├── input_dumps/                # Directory where inputs are saved (1025+ files!)
├── logs/                       # Directory for debug logs
├── checkpoints/               # Directory for actor checkpoints
└── saved_model/               # Directory for saved models
```

## Quick Start

### 1. Setup the Debug Environment

```bash
cd /home/mtanaka/work/dc/actor_debug
./setup_debug.sh
```

This will:
- Backup the original `actor.py` file
- Modify `actor.py` to support input dumping
- Create control scripts
- Test actor initialization

### 2. Test Actor Initialization

```bash
# Basic initialization
deepspeed --num_gpus=1 init_actor_only.py

# With DeepCompile enabled (slow compilation)
deepspeed --num_gpus=1 init_actor_only.py --deepcompile

# With PyTorch compilation enabled (without DeepCompile)
deepspeed --num_gpus=1 init_actor_only.py --compile

# With Flash Attention disabled
deepspeed --num_gpus=1 init_actor_only.py --no-flash-attn

# DeepCompile without Flash Attention (avoids known error)
deepspeed --num_gpus=1 init_actor_only.py --deepcompile --no-flash-attn

# PyTorch compilation without Flash Attention
deepspeed --num_gpus=1 init_actor_only.py --compile --no-flash-attn

# PyTorch compilation with Flash Attention (no packing - uses flash_attn_func)
deepspeed --num_gpus=1 init_actor_only.py --compile --no-packing
```

This will:
- Initialize the actor model standalone with DeepSpeed
- Test a forward pass with CUDA and Flash Attention
- Save the initialized components
- Warn about problematic DeepCompile + Flash Attention + Packing Samples combination

### 3. Enable Input Dumping (Optional - Already Done)

```bash
python control_dumping.py enable
```

**Note**: Input dumping is already enabled and 1025+ real training inputs have been collected.

### 4. Run Training to Collect More Inputs (Optional)

```bash
cd /home/mtanaka/work/dc/run
./run.sh start
```

During training, actor inputs will be automatically saved to `actor_debug/input_dumps/`.

### 5. Replay Inputs for Debugging

```bash
# Replay first 5 collected inputs
deepspeed --num_gpus=1 replay_inputs.py --max-inputs 5

# Replay with PyTorch compilation (without DeepCompile)
deepspeed --num_gpus=1 replay_inputs.py --compile --max-inputs 5

# Replay with DeepSpeed DeepCompile
deepspeed --num_gpus=1 replay_inputs.py --deepcompile --max-inputs 5

# Replay all collected inputs (1025+ files!)
deepspeed --num_gpus=1 replay_inputs.py

# Replay specific input file
deepspeed --num_gpus=1 replay_inputs.py --input-file input_dumps/actor_inputs_rank0_20250619_052608_430740.pkl

# Quiet mode (less verbose output)
deepspeed --num_gpus=1 replay_inputs.py --max-inputs 10 --quiet

# PyTorch compilation without Flash Attention (safer)
deepspeed --num_gpus=1 replay_inputs.py --compile --no-flash-attn --max-inputs 5

# PyTorch compilation with Flash Attention (no packing - should work)
deepspeed --num_gpus=1 replay_inputs.py --compile --no-packing --max-inputs 5
```

## Detailed Usage

### Actor Initialization (`init_actor_only.py`)

Initializes the actor model with the same configuration as the full training setup but without Ray dependencies.

**Features:**
- Uses same model configuration as `run.sh`
- Supports DeepSpeed integration with CUDA
- Tests forward pass functionality with Flash Attention
- Saves initialized state for reuse
- **Requires DeepSpeed launcher**: `deepspeed --num_gpus=1 init_actor_only.py`

**Configuration:**
The script uses hardcoded configuration matching the `run.sh` parameters:
- Model: `meta-llama/Llama-3.1-8B-Instruct` (8B parameters)
- Zero Stage: 1
- Flash Attention: Enabled (CUDA only)
- BF16: Enabled
- Packing samples: Enabled

### Input Dumping Setup (`enable_input_dumping.py`)

Modifies the OpenRLHF actor to support input dumping during training.

**What it does:**
1. Creates backup of original `actor.py`
2. Patches `actor.py` to add dumping capability
3. Creates control script for enabling/disabling dumps

**Safety:**
- Always creates backup before modification
- Can restore original file if needed
- Dumping is disabled by default

### Dumping Control (`control_dumping.py`)

Controls whether input dumping is active during training.

```bash
# Enable dumping
python control_dumping.py enable

# Disable dumping  
python control_dumping.py disable
```

**How it works:**
- Creates/removes a flag file (`dumping_enabled.flag`)
- Actor checks for this flag during initialization
- No restart required when enabling/disabling

### Input Replay (`replay_inputs.py`)

Replays saved inputs through a standalone actor instance.

**Features:**
- Loads and replays all or specific input dumps
- Measures forward pass timing
- Handles multi-process dumps (different ranks)
- Saves detailed results
- Supports verbose and quiet modes
- **Requires DeepSpeed launcher**: `deepspeed --num_gpus=1 replay_inputs.py`

**Current Status**: 1025+ real training input files available for replay!

**Usage examples:**
```bash
# Basic replay (first few inputs)
deepspeed --num_gpus=1 replay_inputs.py --max-inputs 5

# With PyTorch compilation (without DeepCompile)
deepspeed --num_gpus=1 replay_inputs.py --compile --max-inputs 5

# With DeepSpeed DeepCompile
deepspeed --num_gpus=1 replay_inputs.py --deepcompile --max-inputs 5

# Quiet mode
deepspeed --num_gpus=1 replay_inputs.py --max-inputs 10 --quiet

# Replay all collected inputs (1025+ files!)
deepspeed --num_gpus=1 replay_inputs.py

# Replay specific file
deepspeed --num_gpus=1 replay_inputs.py --input-file input_dumps/actor_inputs_rank0_20250619_052608_430740.pkl

# PyTorch compilation without Flash Attention (safer)
deepspeed --num_gpus=1 replay_inputs.py --compile --no-flash-attn --max-inputs 5

# PyTorch compilation with Flash Attention (no packing - should work)
deepspeed --num_gpus=1 replay_inputs.py --compile --no-packing --max-inputs 5
```

## Input Dump Format

Each dumped input file contains:
```python
{
    'sequences': torch.Tensor,           # Input token sequences
    'action_mask': torch.Tensor,         # Action mask for RL
    'attention_mask': torch.Tensor,      # Attention mask
    'return_output': bool,               # Whether to return full output
    'allgather_logits': bool,           # Whether to allgather logits
    'return_logprobs': bool,            # Whether to return log probs
    'packed_seq_lens': list,            # Packed sequence lengths
    'return_entropy': bool,             # Whether to return entropy
    'timestamp': str,                   # When input was captured
    'rank': int,                       # Process rank
    'batch_size': int,                 # Batch size
    'seq_len': int,                    # Sequence length
}
```

## Debugging Workflow

1. **Setup**: Run `./setup_debug.sh` once to setup dumping capability

2. **Debug with Existing Data** (Recommended - 1025+ inputs available):
   - Initialize standalone actor: `deepspeed --num_gpus=1 init_actor_only.py`
   - Replay inputs: `deepspeed --num_gpus=1 replay_inputs.py --max-inputs 5`
   - Analyze results in `logs/` directory

3. **Collect More Data** (Optional): 
   - Enable dumping: `python control_dumping.py enable`
   - Run training: `cd /home/mtanaka/work/dc/run && ./run.sh start`
   - Let it run for a few iterations to collect inputs
   - Stop training and disable dumping: `python control_dumping.py disable`

4. **Iterate**:
   - Modify actor code as needed
   - Re-run replay to test changes with: `deepspeed --num_gpus=1 replay_inputs.py --max-inputs 10`
   - Collect more data if needed

## Troubleshooting

### Actor initialization fails
- **Use DeepSpeed launcher**: `deepspeed --num_gpus=1 init_actor_only.py`
- Check CUDA availability and memory (needs ~45GB GPU memory)
- Verify model path is accessible
- Ensure DeepSpeed is properly installed

### Replay fails with arguments error
- **Use DeepSpeed launcher**: `deepspeed --num_gpus=1 replay_inputs.py`
- The `--local_rank=0` argument is automatically added by DeepSpeed

### No input dumps collected
- Verify dumping is enabled: `ls dumping_enabled.flag`
- Check training actually started actor forward passes
- Look for error messages in training logs
- **Note**: 1025+ input files are already available!

### Replay fails with CUDA errors
- Check GPU memory availability (needs ~45-75GB)
- Try reducing number of inputs: `--max-inputs 1`
- Ensure consistent CUDA setup between dump and replay

### Input dumps too large
- Dumps are saved to disk, ensure sufficient space
- Consider limiting number of dumps collected
- Clean up old dumps regularly: `rm input_dumps/older_files*.pkl`

## Flash Attention + Compilation Investigation

### **Root Cause Analysis**

The compilation error with Flash Attention occurs due to **different flash attention functions being used**:

#### **Standard HuggingFace Models (✅ Compilation Compatible)**
- Uses: `flash_attn_func` for regular, unpacked sequences
- Compatible with PyTorch compilation and torch.dynamo tracing

#### **OpenRLHF with Packing (❌ Compilation Incompatible)**  
- Uses: `flash_attn_varlen_func` for variable-length, packed sequences
- **NOT** compatible with compilation due to FakeTensor handling issues

### **The Specific Error**
```
TorchRuntimeError: flash_attn::_flash_attn_varlen_forward() Expected a value of type 'int' 
for argument 'max_seqlen_q' but instead found type 'FakeTensor'
```

This occurs because:
1. OpenRLHF uses `packing_samples=True` by default for efficiency
2. Packed sequences require attention masks and variable lengths  
3. This forces the use of `flash_attn_varlen_func`
4. The varlen function expects concrete `int` values for `max_seqlen_q/k`
5. During compilation, torch.dynamo passes `FakeTensor` objects instead
6. The C++ binding fails to convert `FakeTensor` to `int`

### **Solutions**

1. **Disable packing** (enables `flash_attn_func`):
   ```bash
   deepspeed --num_gpus=1 replay_inputs.py --compile --no-packing --max-inputs 5
   ```

2. **Disable Flash Attention** (uses SDPA instead):
   ```bash
   deepspeed --num_gpus=1 replay_inputs.py --compile --no-flash-attn --max-inputs 5
   ```

3. **Use separate compilation mode** (PyTorch only, not DeepCompile):
   ```bash
   deepspeed --num_gpus=1 replay_inputs.py --compile --no-packing --max-inputs 5
   ```

### **Function Path Analysis**

**In `transformers/integrations/flash_attention.py`:**

```python
# OpenRLHF path (packing_samples=True) - FAILS with compilation
if attention_mask is not None:
    attn_output = flash_attn_varlen_func(...)  # Uses varlen version
    
# Standard HuggingFace path (packing_samples=False) - WORKS with compilation  
else:
    attn_output = flash_attn_func(...)  # Uses regular version
```

**Key insight**: Disabling packing (`--no-packing`) forces the `else` branch, using the compilation-compatible `flash_attn_func`.

## Restoration

To restore original actor.py:
```python
python -c "
import sys
sys.path.append('/home/mtanaka/work/dc/actor_debug')
from enable_input_dumping import restore_original_actor
restore_original_actor()
"
```

## Files Generated

- `actor_original_backup.py`: Backup of original actor.py
- `dumping_enabled.flag`: Flag file to control dumping (✅ enabled)
- `input_dumps/actor_inputs_*.pkl`: Dumped input files (✅ 1025+ files available)
- `logs/replay_results_*.pkl`: Replay session results
- `initialized_actor.pt`: Saved actor initialization state
- `create_dummy_input.py`: Script to create test inputs

## Current Status

✅ **Setup Complete**: All debugging infrastructure ready  
✅ **Input Dumping**: Enabled and 1025+ real training inputs collected  
✅ **Actor Initialization**: Working with DeepSpeed launcher  
✅ **Input Replay**: Working with DeepSpeed launcher  
✅ **CUDA Support**: Flash Attention and BF16 enabled  

## Notes

- **All scripts require DeepSpeed launcher**: Use `deepspeed --num_gpus=1 <script>`
- Input dumping adds minimal overhead during training
- Dumps are automatically saved with rank and timestamp info
- Multi-GPU training will generate dumps from all ranks
- Replay can handle inputs from any rank/configuration
- All modifications are reversible
- **Memory Requirements**: ~45-75GB GPU memory for full model