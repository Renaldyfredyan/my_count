def print_tensor_info(name, tensor):
    """Utility function untuk print tensor info"""
    if isinstance(tensor, (list, tuple)):
        print(f"\n{name} (list/tuple):")
        for i, t in enumerate(tensor):
            print(f"  [{i}] shape: {t.shape}, device: {t.device}, dtype: {t.dtype}")
            print(f"      max: {t.max().item():.4f}, min: {t.min().item():.4f}")
            print(f"      memory: {t.element_size() * t.nelement() / 1024 / 1024:.2f}MB")
    else:
        print(f"\n{name}:")
        print(f"  shape: {tensor.shape}, device: {tensor.device}, dtype: {tensor.dtype}")
        print(f"  max: {tensor.max().item():.4f}, min: {tensor.min().item():.4f}")
        print(f"  memory: {tensor.element_size() * tensor.nelement() / 1024 / 1024:.2f}MB")

def print_gpu_usage(location):
    """Print current GPU memory usage"""
    print(f"\nGPU Memory at {location}:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    print(f"  Cached: {torch.cuda.memory_cached() / 1024**2:.2f}MB")
    print(f"  Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB")