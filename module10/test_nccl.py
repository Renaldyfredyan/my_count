import torch
import torch.distributed as dist
import os

def run_test():
    # Setup
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Inisialisasi proses grup
    dist.init_process_group("nccl")
    
    # Siapkan device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Buat tensor pada setiap proses
    tensor = torch.ones(1) * rank
    tensor = tensor.to(device)
    
    print(f"Rank {rank}: Tensor awal = {tensor}")
    
    # Lakukan all_reduce (penjumlahan dari semua proses)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Hasil yang benar adalah sum(0, 1, ..., world_size-1)
    expected = sum(range(world_size))
    print(f"Rank {rank}: Tensor setelah all_reduce = {tensor}, Seharusnya = {expected}")
    
    # Selesai
    dist.destroy_process_group()
    
if __name__ == "__main__":
    run_test()