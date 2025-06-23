#!/usr/bin/env python3
"""
Standalone script to initialize only the actor module for debugging.
This replicates the actor initialization from OpenRLHF without Ray dependencies.
"""

import os
import sys
import math
import torch
import argparse
from transformers.trainer import get_scheduler

# Add OpenRLHF to Python path
sys.path.insert(0, '/home/mtanaka/work/dc/OpenRLHF')

from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy

def create_args():
    """Create args similar to the full training script"""
    # Parse command line arguments for configuration options
    args = argparse.Namespace()
    
    # Model configuration (matching run.sh exactly)
    args.pretrain = "meta-llama/Llama-3.1-8B-Instruct"
    args.bf16 = True
    args.load_in_4bit = False
    args.lora_rank = 0
    args.lora_alpha = 16
    args.target_modules = None
    args.lora_dropout = 0
    args.temperature = 1.0
    args.use_liger_kernel = False
    
    # Training configuration
    args.actor_learning_rate = 5e-7
    args.adam_betas = (0.9, 0.95)
    args.l2 = 0.0
    args.lr_warmup_ratio = 0.03
    args.gradient_checkpointing = False
    args.gradient_checkpointing_use_reentrant = False
    
    # DeepSpeed configuration
    args.deepspeed_enable_sleep = False
    
    # Other configurations
    args.enable_ema = False
    args.micro_train_batch_size = 2
    args.eps_clip = 0.2
    args.ema_beta = 0.992
    args.disable_fast_tokenizer = False
    args.seed = 42
    
    # Checkpoint paths
    args.ckpt_path = "/home/mtanaka/work/dc/actor_debug/checkpoints"
    args.save_path = "/home/mtanaka/work/dc/actor_debug/saved_model"
    args.load_checkpoint = False
    
    return args

def init_actor_standalone(zero_stage=1, flash_attn=False, compile=False, deepcompile=False, packing=False, local_rank=-1):
    """Initialize the actor model standalone without Ray"""
    
    print("Initializing actor model for debugging...")
    
    # Create args
    args = create_args()
    args.flash_attn = flash_attn
    args.compile = compile
    args.deepcompile = deepcompile
    args.packing_samples = packing
    args.local_rank = local_rank  # Standalone mode, no distributed training


    # Create DeepSpeed strategy
    strategy = DeepspeedStrategy(
        seed=args.seed,
        micro_train_batch_size=args.micro_train_batch_size,
        train_batch_size=16,  # From run.sh
        zero_stage=zero_stage,
        bf16=args.bf16,
        args=args
    )
    
    # Setup ring attention manually for standalone mode
    strategy.ring_attn_size = 1
    strategy.ring_attn_rank = 0
    strategy.world_size = 1
    strategy.accumulated_gradient = 16 // args.micro_train_batch_size
    
    print("Note: Running in standalone mode (no distributed training)")
    
    print(f"Using model: {args.pretrain}")
    print(f"DeepSpeed Zero Stage: {zero_stage}")
    print(f"Flash Attention: {flash_attn}")
    print(f"BF16: {args.bf16}")
    print(f"DeepCompile: {deepcompile}")
    print(f"PyTorch Compile: {compile}")
    print(f"Packing Samples: {args.packing_samples}")
    
    # Initialize actor
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=not flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        temperature=args.temperature,
        use_liger_kernel=args.use_liger_kernel,
    )
    
    print("Actor model initialized successfully!")
    print(f"Actor model parameters: {sum(p.numel() for p in actor.parameters())}")
    
    # Configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, 
        actor.model, 
        "left", 
        strategy, 
        use_fast=not args.disable_fast_tokenizer
    )
    
    print(f"Tokenizer configured. Vocab size: {len(tokenizer)}")
    
    # Configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, 
        lr=args.actor_learning_rate, 
        betas=args.adam_betas, 
        weight_decay=args.l2
    )
    
    # Configure scheduler
    max_steps = 1000  # Dummy value for initialization
    actor_scheduler = get_scheduler(
        "cosine_with_min_lr",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )
    
    print("Optimizer and scheduler configured successfully!")
    
    # Prepare models/optimizers with DeepSpeed
    actor, actor_optim, actor_scheduler = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        is_rlhf=True,
    )
    
    print("DeepSpeed preparation completed!")

    return {
        'actor': actor,
        'actor_optim': actor_optim, 
        'actor_scheduler': actor_scheduler,
        'tokenizer': tokenizer,
        'strategy': strategy,
        'args': args
    }

def main():
    """Main function to test actor initialization"""
    try:
        components = init_actor_standalone()
        
        print("\n=== Actor Initialization Complete ===")
        print("Available components:")
        for key, value in components.items():
            if key != 'strategy':  # Skip strategy for cleaner output
                print(f"  - {key}: {type(value)}")
        
        # Test forward pass to potentially reproduce the DeepCompile + Flash Attention error
        print("\n=== Testing Actor Forward Pass ===")
        actor = components['actor']
        tokenizer = components['tokenizer']
        
        # Create dummy input
        test_text = "Hello, this is a test input for the actor model."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        # Move to GPU if available
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Test forward pass - this might trigger the error if DeepCompile + Flash Attention + Packing
        try:
            with torch.no_grad():
                output = actor(
                    input_ids,
                    attention_mask=attention_mask,
                    return_output=True
                )
                
            print(f"Forward pass successful!")
            print(f"Output logits shape: {output['logits'].shape}")
            
        except Exception as e:
            if "flash_attn" in str(e) and "FakeTensor" in str(e):
                print(f"🎯 REPRODUCED the DeepCompile + Flash Attention error!")
                print(f"Error: {e}")
                return False
            else:
                print(f"Forward pass failed with different error: {e}")
                raise
        
        # Save the initialized components for later use
        save_path = "/home/mtanaka/work/dc/actor_debug/initialized_actor.pt"
        torch.save({
            'actor_state_dict': actor.state_dict(),
            'args': components['args'],
            'tokenizer': tokenizer
        }, save_path)
        
        print(f"Actor components saved to: {save_path}")
        print("\nActor initialization and testing completed successfully!")
        
    except Exception as e:
        print(f"Error during actor initialization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()