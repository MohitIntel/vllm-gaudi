import torch
import time

def test_masked_scatter_performance():
    """
    Reproducer for masked_scatter_ performance issues on HPU with non_zero_i8 ops
    """
    # Setup device (use HPU if available, otherwise CPU for testing)
    try:
        import habana_frameworks.torch.core as htcore
        device = "hpu"
        print("Using HPU device")
    except ImportError:
        device = "cpu" 
        print("HPU not available, using CPU")
    
    # Test parameters - adjust these to match your use case
    batch_size = 1
    seq_len = 2048
    embed_dim = 4096
    num_multimodal_tokens = 576  # Typical for vision tokens
    
    # Create test tensors
    inputs_embeds = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float16)
    multimodal_embeds = torch.randn(num_multimodal_tokens, embed_dim, device=device, dtype=torch.float16)
    
    # Create boolean mask with multimodal positions
    is_multimodal = torch.zeros(seq_len, dtype=torch.bool, device=device)
    # Set first num_multimodal_tokens positions to True
    is_multimodal[:num_multimodal_tokens] = True
    # Shuffle to make it more realistic
    indices = torch.randperm(seq_len, device=device)
    is_multimodal = is_multimodal[indices]
    
    print(f"Input shape: {inputs_embeds.shape}")
    print(f"Multimodal embeddings shape: {multimodal_embeds.shape}")
    print(f"Mask shape: {is_multimodal.shape}")
    print(f"Number of True positions in mask: {is_multimodal.sum().item()}")
    
    if device == "hpu":
        htcore.mark_step()
    
    # Test the problematic masked_scatter_ operation
    print("\n=== Testing masked_scatter_ (PROBLEMATIC) ===")
    
    inputs_embeds_copy = inputs_embeds.clone()
    
    start_time = time.time()
    
    try:
        # This is the problematic operation that causes non_zero_i8 ops in TPC kernel
        inputs_embeds_copy.masked_scatter_(
            is_multimodal.unsqueeze(-1),  # Shape: [seq_len, 1] 
            multimodal_embeds
        )
        
        if device == "hpu":
            htcore.mark_step()  # Force synchronization
            
        end_time = time.time()
        print(f"masked_scatter_ completed in {end_time - start_time:.4f} seconds")
        
    except Exception as e:
        print(f"masked_scatter_ failed: {e}")
    
    # Test alternative approaches
    print("\n=== Testing torch.where alternative (DYNAMIC) ===")
    
    inputs_embeds_copy2 = inputs_embeds.clone()
    
    start_time = time.time()
    
    try:
        # This causes dynamic operations and recompilation
        multimodal_positions = torch.where(is_multimodal)[0][:multimodal_embeds.shape[0]]
        inputs_embeds_copy2[0, multimodal_positions] = multimodal_embeds
        
        if device == "hpu":
            htcore.mark_step()
            
        end_time = time.time()
        print(f"torch.where alternative completed in {end_time - start_time:.4f} seconds")
        
    except Exception as e:
        print(f"torch.where alternative failed: {e}")
    
    # Test static indexing approach
    print("\n=== Testing static indexing alternative (RECOMMENDED) ===")
    
    inputs_embeds_copy3 = inputs_embeds.clone()
    
    start_time = time.time()
    
    try:
        # Static approach using boolean indexing
        static_indices = torch.arange(seq_len, device=device)
        valid_positions = static_indices[is_multimodal][:multimodal_embeds.shape[0]]
        inputs_embeds_copy3[0, valid_positions] = multimodal_embeds
        
        if device == "hpu":
            htcore.mark_step()
            
        end_time = time.time()
        print(f"Static indexing completed in {end_time - start_time:.4f} seconds")
        
    except Exception as e:
        print(f"Static indexing failed: {e}")
    
    # Verify all methods produce the same result
    if device == "cpu":  # Only check on CPU to avoid HPU sync issues
        try:
            print("\n=== Verifying correctness ===")
            print(f"masked_scatter_ vs torch.where: {torch.allclose(inputs_embeds_copy, inputs_embeds_copy2, atol=1e-6)}")
            print(f"masked_scatter_ vs static indexing: {torch.allclose(inputs_embeds_copy, inputs_embeds_copy3, atol=1e-6)}")
        except:
            print("Could not verify correctness")

if __name__ == "__main__":
    # Run multiple iterations to see compilation effects
    print("Running masked_scatter_ performance reproducer...")
    
    for i in range(3):
        print(f"\n{'='*50}")
        print(f"ITERATION {i+1}")
        print(f"{'='*50}")
        test_masked_scatter_performance()