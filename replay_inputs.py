#!/usr/bin/env python3
"""
Script to replay saved actor inputs for debugging.
This loads dumped inputs and feeds them to a standalone actor instance.
"""

import os
import sys
import pickle
import glob
import torch
import argparse
import random
import numpy as np
from datetime import datetime

# Add OpenRLHF to Python path
sys.path.insert(0, '/home/mtanaka/work/dc/OpenRLHF')


def load_input_dumps(dump_dir="/home/mtanaka/work/dc/actor_debug/input_dumps", shuffle_inputs=False, rank=None, world_size=None, random_seed=None):
    """Load all input dump files from the dump directory
    
    Args:
        dump_dir: Directory containing input dumps
        shuffle_inputs: Whether to randomly shuffle inputs (different order per rank)
        rank: Current rank (0-indexed) for multi-GPU setup
        world_size: Total number of ranks for multi-GPU setup
        random_seed: Base random seed (each rank gets different seed)
    """
    dump_files = glob.glob(os.path.join(dump_dir, "actor_inputs_*.pkl"))
    
    if not dump_files:
        print(f"No input dump files found in {dump_dir}")
        return []
    
    print(f"Found {len(dump_files)} input dump files")
    
    dumps = []
    for dump_file in sorted(dump_files):
        try:
            with open(dump_file, 'rb') as f:
                data = pickle.load(f)
                data['source_file'] = dump_file
                dumps.append(data)
                # if rank is None or rank == 0:  # Only print on rank 0 for multi-GPU
                #     print(f"Loaded: {os.path.basename(dump_file)} (rank {data.get('rank', 'unknown')}, batch_size: {data.get('batch_size', 'unknown')}, seq_len: {data.get('seq_len', 'unknown')})")
        except Exception as e:
            print(f"Error loading {dump_file}: {e}")
    
    # Apply rank-specific shuffling if requested
    if shuffle_inputs and rank is not None and world_size is not None:
        # Use rank-specific random seed to ensure different orders across ranks
        if random_seed is None:
            random_seed = 42
        
        rank_seed = random_seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        
        # Shuffle the dumps differently for each rank
        random.shuffle(dumps)
        
        print(f"[Rank {rank}] Shuffled inputs with seed {rank_seed} to create sequence length mismatches across ranks")
        
        # Print first few sequence lengths for this rank
        if len(dumps) > 0:
            seq_lens = [d.get('seq_len', 'unknown') for d in dumps[:5]]
            print(f"[Rank {rank}] First 5 sequence lengths after shuffle: {seq_lens}")
    
    return dumps

def replay_single_input(actor, dump_data, device, verbose=True, enable_backward=True):
    """Replay a single input dump through the actor with backward pass"""
    
    if verbose:
        print(f"\n=== Replaying Input ===")
        print(f"Source: {os.path.basename(dump_data['source_file'])}")
        print(f"Timestamp: {dump_data.get('timestamp', 'unknown')}")
        print(f"Rank: {dump_data.get('rank', 'unknown')}")
        print(f"Batch size: {dump_data.get('batch_size', 'unknown')}")
        print(f"Sequence length: {dump_data.get('seq_len', 'unknown')}")
    
    # Extract input data
    sequences = dump_data['sequences']
    action_mask = dump_data['action_mask']
    attention_mask = dump_data['attention_mask']
    return_output = dump_data.get('return_output', False)
    allgather_logits = dump_data.get('allgather_logits', False)
    return_logprobs = dump_data.get('return_logprobs', False)
    packed_seq_lens = dump_data.get('packed_seq_lens', None)
    return_entropy = dump_data.get('return_entropy', False)

    # Move data to device
    if sequences is not None:
        sequences = sequences.to(device)
    if action_mask is not None:
        action_mask = action_mask.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Run forward and backward pass
    try:
        start_time = torch.cuda.synchronize() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        import time
        cpu_start_time = time.time()
        
        # Forward pass without no_grad() to enable gradients
        result = actor(
            sequences=sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            return_output=return_output,
            allgather_logits=allgather_logits,
            return_logprobs=return_logprobs,
            packed_seq_lens=packed_seq_lens,
            return_entropy=return_entropy,
        )
        
        # Extract action_log_probs for loss computation (actor returns log probs, not logits)
        if isinstance(result, tuple):
            action_log_probs = result[0]  # First element is action_log_probs
            output = result[1] if len(result) > 1 else None
        else:
            action_log_probs = result
            output = None
            
        # Conditional backward pass
        if enable_backward:
            # Simple loss for backward pass - just use the negative log probability
            # In real PPO training, this would be computed using PolicyLoss with advantages
            # Here we just create a dummy loss that's differentiable
            if action_mask is not None:
                # Mask the log probs to focus on action tokens only
                masked_log_probs = action_log_probs * action_mask.float()
                # Simple loss: negative mean of masked log probabilities
                loss = -masked_log_probs.sum() / action_mask.sum().clamp(min=1)
            else:
                # Fallback: mean of all log probabilities
                loss = -action_log_probs.mean()
            
            # Backward pass
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            backward_time = backward_end_time - backward_start_time
        else:
            # Skip backward pass
            loss = None
            backward_time = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        cpu_end_time = time.time()
        total_time = cpu_end_time - cpu_start_time
        forward_time = total_time - backward_time
        
        if verbose:
            print(f"Forward pass completed successfully!")
            print(f"Forward time: {forward_time:.4f} seconds")
            if enable_backward:
                print(f"Backward time: {backward_time:.4f} seconds")
                print(f"Total time: {total_time:.4f} seconds")
                print(f"Loss: {loss.item():.6f}")
            else:
                print(f"Backward pass: SKIPPED")
                print(f"Total time: {forward_time:.4f} seconds")
            
            if isinstance(result, tuple):
                print(f"Result type: tuple with {len(result)} elements")
                for i, item in enumerate(result):
                    if hasattr(item, 'shape'):
                        print(f"  Element {i}: shape {item.shape}, dtype {item.dtype}")
                    else:
                        print(f"  Element {i}: {type(item)}")
            elif hasattr(result, 'shape'):
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            else:
                print(f"Result type: {type(result)}")
        
        return {
            'success': True,
            'result': result,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': total_time,
            'loss': loss.item() if loss is not None else None,
            'enable_backward': enable_backward,
            'input_info': {
                'batch_size': dump_data.get('batch_size', 'unknown'),
                'seq_len': dump_data.get('seq_len', 'unknown'),
                'rank': dump_data.get('rank', 'unknown'),
                'timestamp': dump_data.get('timestamp', 'unknown'),
            }
        }
        
    except Exception as e:
        if verbose:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'input_info': {
                'batch_size': dump_data.get('batch_size', 'unknown'),
                'seq_len': dump_data.get('seq_len', 'unknown'),
                'rank': dump_data.get('rank', 'unknown'),
                'timestamp': dump_data.get('timestamp', 'unknown'),
            }
        }

def get_rank_info():
    """Get current rank and world size for multi-GPU setup"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            return rank, world_size
        else:
            # Check environment variables as fallback
            rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            return rank, world_size
    except Exception:
        return 0, 1

def replay_all_inputs(max_inputs=None, verbose=True, no_flash_attn=False, deepcompile=False, compile=False, no_packing=False, zero_stage=1, shuffle_inputs=False, random_seed=None, enable_backward=True, num_layers=0):
    """Replay all available input dumps"""
    
    # Get rank information for multi-GPU setup
    rank, world_size = get_rank_info()
    
    print(f"=== Actor Input Replay Debug Session (Rank {rank}/{world_size}) ===")
    print(f"Starting at: {datetime.now()}")
    if shuffle_inputs:
        print(f"🎲 Random shuffling ENABLED - Each rank will process inputs in different order")
        print(f"   This creates sequence length mismatches that trigger padding!")
    
    # Initialize actor
    print(f"\n1. Initializing actor on rank {rank}...")
    try:
        # Import init_actor_only temporarily to create args with options
        from init_actor_only import init_actor_standalone
        components = init_actor_standalone(
            flash_attn=not no_flash_attn, 
            deepcompile=deepcompile, 
            compile=compile, 
            packing=not no_packing, 
            zero_stage=zero_stage,
            num_layers=num_layers
        )
            
        actor = components['actor']
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        print(f"[Rank {rank}] Actor initialized successfully on device: {device}")
    except Exception as e:
        print(f"[Rank {rank}] Failed to initialize actor: {e}")
        return False
    
    # Load input dumps with rank-specific shuffling
    print(f"\n2. Loading input dumps on rank {rank}...")
    dumps = load_input_dumps(
        shuffle_inputs=shuffle_inputs, 
        rank=rank, 
        world_size=world_size, 
        random_seed=random_seed
    )
    
    if not dumps:
        print(f"[Rank {rank}] No input dumps available for replay")
        return False
    
    # Limit number of inputs if specified
    if max_inputs and len(dumps) > max_inputs:
        dumps = dumps[:max_inputs]
        print(f"[Rank {rank}] Limited to first {max_inputs} inputs")
    
    # Replay each input
    print(f"\n3. [Rank {rank}] Replaying {len(dumps)} inputs...")
    
    results = []
    successful_replays = 0
    failed_replays = 0
    total_forward_time = 0
    total_backward_time = 0
    total_loss = 0
    
    for i, dump_data in enumerate(dumps):
        if verbose or rank == 0:  # Reduce output spam on non-zero ranks
            print(f"\n--- [Rank {rank}] Input {i+1}/{len(dumps)} ---")
        
        result = replay_single_input(actor, dump_data, device, verbose=(verbose and rank == 0), enable_backward=enable_backward)
        results.append(result)
        
        if result['success']:
            successful_replays += 1
            total_forward_time += result['forward_time']
            total_backward_time += result.get('backward_time', 0)
            loss_value = result.get('loss', 0)
            if loss_value is not None:
                total_loss += loss_value
        else:
            failed_replays += 1
            # Always print errors regardless of rank
            print(f"[Rank {rank}] ERROR in input {i+1}: {result.get('error', 'Unknown error')}")
    
    # Summary
    print(f"\n=== [Rank {rank}] Replay Summary ===")
    print(f"Total inputs: {len(dumps)}")
    print(f"Successful: {successful_replays}")
    print(f"Failed: {failed_replays}")
    
    if successful_replays > 0:
        avg_forward_time = total_forward_time / successful_replays
        print(f"Average forward time: {avg_forward_time:.4f} seconds")
        print(f"Total forward time: {total_forward_time:.4f} seconds")
        
        if enable_backward:
            avg_backward_time = total_backward_time / successful_replays
            avg_loss = total_loss / successful_replays
            print(f"Average backward time: {avg_backward_time:.4f} seconds")
            print(f"Average loss: {avg_loss:.6f}")
            print(f"Total backward time: {total_backward_time:.4f} seconds")
        else:
            print("Backward pass: DISABLED")
            avg_backward_time = 0
            avg_loss = 0
    
    # Save results (only rank 0 saves to avoid conflicts)
    if rank == 0:
        results_file = f"/home/mtanaka/work/dc/actor_debug/logs/replay_results_rank{rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        try:
            with open(results_file, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'summary': {
                        'rank': rank,
                        'world_size': world_size,
                        'shuffle_inputs': shuffle_inputs,
                        'total_inputs': len(dumps),
                        'successful': successful_replays,
                        'failed': failed_replays,
                        'avg_forward_time': avg_forward_time if successful_replays > 0 else 0,
                        'avg_backward_time': avg_backward_time if successful_replays > 0 else 0,
                        'avg_loss': avg_loss if successful_replays > 0 else 0,
                        'total_forward_time': total_forward_time,
                        'total_backward_time': total_backward_time,
                    },
                    'timestamp': datetime.now().isoformat()
                }, f)
            print(f"[Rank {rank}] Results saved to: {results_file}")
        except Exception as e:
            print(f"[Rank {rank}] Could not save results: {e}")
    
    return successful_replays > 0


def main():
    parser = argparse.ArgumentParser(description="Replay actor inputs for debugging")
    parser.add_argument('--max-inputs', type=int, help="Maximum number of inputs to replay")
    parser.add_argument('--quiet', action='store_true', help="Reduce verbose output")
    parser.add_argument('--local_rank', type=int, default=-1, help="Local rank for DeepSpeed launcher")
    parser.add_argument('--no-flash-attn', action='store_true', help="Disable Flash Attention")
    parser.add_argument('--deepcompile', action='store_true', help="Enable DeepSpeed DeepCompile")
    parser.add_argument('--compile', action='store_true', help="Enable PyTorch compilation (without DeepCompile)")
    parser.add_argument('--no-packing', action='store_true', help="Disable packing samples (may fix compilation with flash attention)")
    parser.add_argument('--zero-stage', type=int, default=1, choices=[0, 1, 2, 3], help="DeepSpeed ZeRO stage (default: 1)")
    
    # New arguments for multi-rank padding reproduction
    parser.add_argument('--shuffle-inputs', action='store_true', 
                       help="Randomly shuffle inputs differently per rank to create sequence length mismatches (triggers padding)")
    parser.add_argument('--random-seed', type=int, default=42,
                       help="Base random seed for shuffling (each rank gets seed + rank)")
    parser.add_argument('--no-backward', action='store_true',
                       help="Skip backward pass (only run forward pass)")
    parser.add_argument('--num-layers', type=int, default=0,
                       help="Number of transformer layers (0 = use default from model config)")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    enable_backward = not args.no_backward
    
    success = replay_all_inputs(
        max_inputs=args.max_inputs, 
        verbose=verbose, 
        no_flash_attn=args.no_flash_attn, 
        deepcompile=args.deepcompile, 
        compile=args.compile, 
        no_packing=args.no_packing,
        zero_stage=args.zero_stage,
        shuffle_inputs=args.shuffle_inputs,
        random_seed=args.random_seed,
        enable_backward=enable_backward,
        num_layers=args.num_layers
    )
    
    if success:
        print("\nReplay session completed successfully!")
        sys.exit(0)
    else:
        print("\nReplay session failed!")
        sys.exit(1)

if __name__ == "__main__":
    import torch
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.verbose = True
    torch._dynamo.config.enable_compiler_collectives = True
    torch._dynamo.config.capture_scalar_outputs = True
    main()