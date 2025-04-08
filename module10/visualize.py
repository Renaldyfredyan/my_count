from engine import build_model
from data import FSC147Dataset
from arg_parser import get_argparser

import argparse
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import matplotlib.pyplot as plt
import numpy as np

@torch.no_grad()
def evaluate_single_image(args, image_idx=0, image_name=None):

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # multi gpu communication
    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    # memuat state dictionary yang berisi parameter (weights dan bias)
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'), weights_only=True)['model']
    # Penyesuaian Nama Parameter (module.) karena menggunakan DistributedDataParallel (DDP) pelatihan multi-GPU
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    # Memuat state_dict ke Model
    model.load_state_dict(state_dict)

    # Pilih dataset (misalnya, 'val' atau 'test')
    # split = 'test'  
    split = args.split if hasattr(args, 'split') else 'test'  # Default tetap 'test'
    
    # Membuat dataset
    test = FSC147Dataset(
        args.data_path,
        args.image_size,
        split=split,
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        return_image_name=True
    )
    
    
    # Jika rank == 0, hanya proses utama yang akan memvisualisasikan gambar

        # Jika rank == 0, hanya proses utama yang akan memvisualisasikan gambar
    if rank == 0:
        # Jika nama file diberikan, cari indeks yang sesuai
        selected_idx = image_idx  # Default menggunakan indeks
        
        if image_name is not None:
            # Mencari indeks berdasarkan nama file
            for idx in range(len(test)):
                _, _, _, img_name = test[idx]
                
                # Ekstrak nama file dasar untuk perbandingan
                if isinstance(img_name, tuple):
                    img_name = img_name[0]
                
                base_img_name = os.path.basename(img_name) if isinstance(img_name, str) else ""
                
                # Jika nama file cocok, gunakan indeks ini
                if base_img_name == image_name or img_name == image_name:
                    selected_idx = idx
                    print(f"Found image '{image_name}' at index {selected_idx}")
                    break
            else:
                print(f"Warning: Image '{image_name}' not found, using index {selected_idx} instead")
        
        # Mengambil gambar dari dataset berdasarkan indeks yang dipilih
        img, bboxes, density_map, image_name = test[selected_idx]


    # if rank == 0:
    #     # Mengambil satu gambar dari dataset berdasarkan indeks
    #     img, bboxes, density_map, image_name = test[image_idx]
        
        # Menambahkan dimensi batch
        img = img.unsqueeze(0).to(device)
        bboxes = bboxes.unsqueeze(0).to(device)
        density_map = density_map.unsqueeze(0).to(device)
        
        # Model ke mode evaluasi
        model.eval()
        
        # Mendapatkan density map prediksi dari model
        out, _ = model(img, bboxes)
        
        # Pastikan image_name adalah string (bukan tuple)
        if isinstance(image_name, tuple):
            image_name = image_name[0]
            
        # Ekstrak nama file dasar untuk penyimpanan yang unik
        file_base_name = os.path.splitext(image_name)[0] if isinstance(image_name, str) else f"image_{image_idx}"
        
        # Visualisasi dan simpan density map prediksi
        predicted_density = out[0, 0].cpu().numpy()  # Mengambil gambar pertama, channel pertama saja
        
        # plt.figure(figsize=(10, 8))
        
        # Plot gambar asli
        # plt.subplot(2, 1, 1)
        # Convert tensor ke numpy dan transpose untuk plotting
        # original_img = img[0].cpu().permute(1, 2, 0).numpy()
        # # Normalize jika perlu
        # if original_img.max() > 1.0:
        #     original_img = original_img / 255.0

        # Normalize image properly
        # original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())

        # # plt.imshow(original_img)
        # # plt.title('Original Image')
        
        # # Plot density map prediksi
        # # plt.subplot(2, 1, 2)
        # # plt.subplot(2, 1, 1)
        # plt.imshow(predicted_density, cmap='hot')
        # plt.colorbar()
        # plt.title('Predicted Density Map')
        
        # # Simpan gambar
        # output_folder = "density_maps_prediction"
        # os.makedirs(output_folder, exist_ok=True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(output_folder, f'{file_base_name}_prediction.png'))
        # plt.close()
        
        
        # Hitung dan tampilkan jumlah objek yang diprediksi
        predicted_count = predicted_density.sum()
        actual_count = density_map[0, 0].cpu().numpy().sum()

       
        print(f'Gambar: {file_base_name}')
        print(f'Jumlah objek yang diprediksi: {predicted_count:.2f}')
        print(f'Jumlah objek sebenarnya: {actual_count:.2f}')
        print(f'Selisih: {abs(predicted_count - actual_count):.2f}')

        # Plot hanya density map prediksi dengan ukuran yang lebih besar
        plt.figure(figsize=(10, 8))  # Ukuran figure yang lebih besar

        # Visualisasi density map prediksi
        predicted_density = out[0, 0].cpu().numpy()  # Mengambil gambar pertama, channel pertama saja
        plt.imshow(predicted_density, cmap='hot')
        plt.colorbar()
        # plt.title(f'Predicted Density Map\nCount: {predicted_count:.2f}', fontsize=14)  # Tambahkan count di judul

        # Simpan gambar
        output_folder = "density_maps_prediction"
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, f'{file_base_name}_prediction.png'), dpi=100, bbox_inches='tight')  # Tingkatkan DPI
        plt.close()  # Tutup plot untuk menghindari masalah memori

        np.save(os.path.join(output_folder, f'{file_base_name}_predicted_density.npy'), predicted_density)
        

    # Hapus grup proses
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Single Image Evaluation', parents=[get_argparser()])
    parser.add_argument('--image_idx', type=int, default=0, help='Index of the image to evaluate')
    # parser.add_argument('--image_name', type=str, default=None, help='Name of the image file to evaluate')
    args = parser.parse_args()
    
    # Panggil fungsi untuk mengevaluasi satu gambar
    evaluate_single_image(args, image_idx=args.image_idx, image_name=args.image_name)