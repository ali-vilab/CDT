import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
import sys
from torch.utils.data import DataLoader, Subset
import os
sys.path.append(".")
from utils import AverageMeter, custom_to_images, SimpleImageDataset
from opensora_evaluate.cal_lpips import calculate_lpips
from opensora_evaluate.cal_psnr import calculate_psnr
from opensora_evaluate.cal_ssim import calculate_ssim
import time
from model.cdt import load_cdt


@torch.no_grad()
def main(args: argparse.Namespace):
    real_data_dir = args.real_data_dir 
    dataset = args.dataset
    device = args.device
    batch_size = args.batch_size
    num_workers = 4
    subset_size = args.subset_size
    
    if args.data_type == "bfloat16":
        data_type = torch.bfloat16
    elif args.data_type == "float32":
        data_type = torch.float32
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    folder_name = f"{args.method}_{args.data_type}"
    
    
    generated_images_dir = os.path.join('./reconstructed_results/image_results/', dataset, folder_name)
    metrics_results = os.path.join('./reconstructed_results/image_results/', dataset, 'results.txt')


    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)

    # ---- Load Model ----
    device = args.device
    assert 'CDT' in args.method, f"method must be CDT, but got {args.method}"
    if 'base' in args.method:
        print(f"Loading CDT-base")
        vae = load_cdt('base')
        print(f"CDT-base Loaded")
    elif 'small' in args.method:
        print(f"Loading CDT-small")
        vae = load_cdt('small')
        print(f"CDT-small Loaded")
    vae = vae.to(device).to(data_type).eval()
    model_size = sum([p.numel() for p in vae.parameters()]) / 1e6 
    print(f'Successfully loaded {args.method} model with {model_size:.3f} million parameters') 
    # ---- Load Model ----

    # ---- Prepare Dataset ----
    dataset = SimpleImageDataset(image_dir=real_data_dir)
    print(f"Total images found: {len(dataset)}")
    if subset_size:
        indices = range(subset_size)
        dataset = Subset(dataset, indices=indices)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers
    )
    # ---- Prepare Dataset

    # ---- Inference ----
    avg_ssim = AverageMeter()
    avg_psnr = AverageMeter()
    avg_lpips = AverageMeter()
    
    log_txt = os.path.join(generated_images_dir, 'results.txt')

    step = 0
    total_time = 0
    total_images = 0
    
    with open(log_txt, 'a+') as f:
        for batch in tqdm(dataloader):
            step += 1
            x, file_names = batch['image'], batch['file_name']
            original_width = batch['original_width'][0]
            original_height = batch['original_height'][0]
            torch.cuda.empty_cache()
            
            x = x.to(device=device, dtype=data_type)
            x=x.unsqueeze(2)
            start_time = time.time()
            video_recon = vae(x)
            torch.cuda.synchronize()
            end_time = time.time()
            total_time += end_time - start_time
            total_images += 1        
            
            x, video_recon = x.data.cpu().float(), video_recon.data.cpu().float()

        
            if not os.path.exists(generated_images_dir):
                os.makedirs(generated_images_dir, exist_ok=True)

            video_recon = video_recon.squeeze(2)
            for idx, image_recon in enumerate(video_recon):
                output_file = os.path.join(generated_images_dir, file_names[idx])
                custom_to_images(image_recon,output_file,original_height,original_width)


            video_recon = video_recon.unsqueeze(2)
            x = torch.clamp(x, -1, 1)
            x = (x + 1) / 2
            video_recon = torch.clamp(video_recon, -1, 1)
            video_recon = (video_recon + 1) / 2
            x = x.permute(0,2,1,3,4).float() 
            video_recon = video_recon.permute(0,2,1,3,4).float() 
            
            # SSIM
            tmp_list = list(calculate_ssim(x, video_recon)['value'].values())
            avg_ssim.updata(np.mean(tmp_list))
            
            # PSNR
            tmp_list = list(calculate_psnr(x, video_recon)['value'].values())
            avg_psnr.updata(np.mean(tmp_list))
            
            # LPIPS
            tmp_list = list(calculate_lpips(x, video_recon, args.device)['value'].values())    
            avg_lpips.updata(np.mean(tmp_list))
            
            if step % args.log_every_steps ==0:
                result = (
                        f'Step: {step}, PSNR: {avg_psnr.avg}\n'
                        f'Step: {step}, SSIM: {avg_ssim.avg}\n'
                        f'Step: {step}, LPIPS: {avg_lpips.avg}\n')
                print(result, flush=True)
                f.write("="*20+'\n')
                f.write(result)
    
    final_result = (f'psnr: {avg_psnr.avg}\n'
                    f'ssim: {avg_ssim.avg}\n'
                    f'lpips: {avg_lpips.avg}')
    
    print("="*20)
    print("Final Results:")
    print(final_result)
    print("="*20)
    print(f'Eval Info:\nmethod: {args.method}\nreal_data_dir: {args.real_data_dir}')
    print("="*20)


    
    with open(metrics_results, 'a') as f:
        f.write("="*20+'\n')
        f.write(f'PSNR: {avg_psnr.avg}\n')
        f.write(f'SSIM: {avg_ssim.avg}\n')
        f.write(f'LPIPS: {avg_lpips.avg}\n')
        f.write(f'Time: {total_time}\n')
        f.write(f'Images Number: {total_images}\n')
        f.write(f'Avg Time: {total_time/total_images:.4f}\n')
        f.write(f'Method: {args.method}\n')
        f.write(f'Real Data Dir: {args.real_data_dir}\n')
        f.write(f'Data Type: {data_type}\n')
        f.write("="*20+'\n\n')
    # ---- Inference ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_dir", type=str)
    parser.add_argument("--dataset", type=str, default='coco17')
    parser.add_argument("--method", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_type", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--log_every_steps", type=int, default=50)
    args = parser.parse_args()
    main(args)
    
