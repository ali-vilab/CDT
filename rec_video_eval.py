import os
import sys
import random
import argparse
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import torch 
from torch.utils.data import DataLoader, Subset
sys.path.append(".")
from utils import *
from opensora_evaluate.cal_lpips import calculate_lpips
from opensora_evaluate.cal_psnr import calculate_psnr
from opensora_evaluate.cal_ssim import calculate_ssim
import time
from model.cdt import load_cdt


@torch.no_grad()
def main(args: argparse.Namespace):
    real_data_dir = args.real_data_dir 
    dataset = args.dataset
    sample_rate = args.sample_rate
    resolution = args.resolution
    crop_size = args.crop_size
    num_frames = args.num_frames
    sample_rate = args.sample_rate
    device = args.device
    sample_fps = args.sample_fps
    batch_size = args.batch_size
    num_workers = args.num_workers
    subset_size = args.subset_size
     
    if args.data_type == "bfloat16":
        data_type = torch.bfloat16
    elif args.data_type == "float32":
        data_type = torch.float32
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")


    folder_name = f"{args.method}_{args.resolution}_{args.data_type}"
    
    
    generated_video_dir = os.path.join('./reconstructed_results/video_results/', dataset, folder_name)
    metrics_results = os.path.join('./reconstructed_results/video_results/', dataset, 'results.txt')


    if not os.path.exists(generated_video_dir):
        os.makedirs(generated_video_dir)


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
    dataset = RealVideoDataset(
        real_data_dir=real_data_dir,
        num_frames=num_frames,
        sample_rate=sample_rate,
        crop_size=crop_size,
        resolution=resolution,
    )
    if subset_size:
        indices = range(subset_size)
        dataset = Subset(dataset, indices=indices)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    # ---- Prepare Dataset



    # ---- Inference ----

    avg_ssim = AverageMeter()
    avg_psnr = AverageMeter()
    avg_lpips = AverageMeter()

    log_txt = os.path.join(generated_video_dir, 'results.txt')

    total_time = 0
    total_videos = 0
    step = 0
    
    with open(log_txt, 'a+') as f:
        for batch in tqdm(dataloader):
            step += 1
            x, file_names = batch['video'], batch['file_name']   
            if x.size(2) < args.num_frames:
                print(file_names)
                continue
            torch.cuda.empty_cache()
            x = x.to(device=device, dtype=data_type)

            start_time = time.time()
            video_recon = vae(x)  
            torch.cuda.synchronize()
            end_time = time.time()
            total_time += end_time - start_time
            total_videos += 1


            x, video_recon = x.data.cpu().float(), video_recon.data.cpu().float()
            
            # save reconstructed video            
            if not os.path.exists(generated_video_dir):
                os.makedirs(generated_video_dir, exist_ok=True)
            for idx, video in enumerate(video_recon):
                output_path = os.path.join(generated_video_dir, file_names[idx])
                custom_to_video(
                    video, fps=sample_fps / sample_rate, output_file=output_path
                )

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
    print(f'Eval Info:\nmethod: {args.method}\nresolution: {args.resolution}\nnum_frames: {args.num_frames}\nreal_data_dir: {args.real_data_dir}')
    print("="*20)

    with open(metrics_results, 'a') as f:
        f.write("="*20+'\n')
        f.write(f'PSNR: {avg_psnr.avg}\n')
        f.write(f'SSIM: {avg_ssim.avg}\n')
        f.write(f'LPIPS: {avg_lpips.avg}\n')
        f.write(f'Time: {total_time}\n')
        f.write(f'Images Number: {total_videos}\n')
        f.write(f'Avg Time: {total_time/total_videos:.4f}\n')
        f.write(f'Method: {args.method}\n')
        f.write(f'Resolution: {args.resolution}\n')
        f.write(f'Num Frames: {args.num_frames}\n')
        f.write(f'Real Data Dir: {args.real_data_dir}\n')
        f.write(f'Data Type: {data_type}\n')
        f.write("="*20+'\n\n')
    # ---- Inference ----


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_dir", type=str)
    parser.add_argument("--dataset", type=str, default='webvid')
    parser.add_argument("--method", type=str)
    parser.add_argument("--sample_fps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=17) # number of frames for video evaluation
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_type", type=str, default="float32", choices=["float32", "bfloat16"])
    
    args = parser.parse_args()
    main(args)
    
