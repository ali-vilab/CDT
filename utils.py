import os
import cv2
import torch
import numpy as np
import numpy.typing as npt
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0
    
    def updata(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.cnt += n
        if self.cnt == 0:
            self.avg = 1
        else:
            self.avg = self.sum / self.cnt


# =============================== Image ===============================

class Image_Crop(object):
    def __init__(self):
        pass
    def __call__(self, img):
        iw, ih = img.size
        ow = (iw // 8) * 8
        oh = (ih // 8) * 8 
        return img.crop((0, 0, ow, oh))

def custom_to_images(x, output_dir, h_i, h_w):
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 2, 0).float()
    x =  (x.numpy() * 255).round().astype(np.uint8)
    
    img = Image.fromarray(x, mode='RGB')
    img = T.Resize((h_i, h_w))(img)
    img.save(output_dir)

class SimpleImageDataset(Dataset):
    def __init__(self, image_dir):
    
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = T.Compose([
            Image_Crop(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size
        image = self.transform(image)
        
        return {
            'image': image,
            'file_name': img_name,
            'original_width': original_width,
            'original_height': original_height
        }

# =============================== Video ===============================

class CenterCrop(object):
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        # resize  
        iw, ih, ow, oh = *img.size, *self.size  
        scale = max(ow / iw, oh / ih)
        img = img.resize((round(scale * iw), round(scale * ih)), Image.LANCZOS)

        # center crop
        w, h = img.size
        if w > ow:
            x1 = (w - ow) // 2
            img = img.crop((x1, 0, x1 + ow, oh))
        elif h > oh:
            y1 = (h - oh) // 2
            img = img.crop((0, y1, ow, y1 + oh))
        return img


def array_to_video(
    image_array: npt.NDArray, fps: float = 30.0, path: str = "output_video.mp4"
) -> None:
    frame_dir = path.replace('&', '_').replace('.mp4', '_temp')
    os.makedirs(frame_dir, exist_ok=True)
    for fid, frame in enumerate(image_array):
        tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
        cv2.imwrite(tpth, frame[:,:,::-1])
    

    cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {path}'
    os.system(cmd)
    os.system(f'rm -rf {frame_dir}')


def custom_to_video(
    x: torch.Tensor, fps: float = 2.0, output_file: str = "output_video.mp4"
) -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 2, 3, 0).float().numpy()
    x = (255 * x).astype(np.uint8) 
    array_to_video(x, fps=fps, path=output_file)
    # breakpoint()
    return
 

class RealVideoDataset(Dataset):
    def __init__(
        self,
        real_data_dir,
        num_frames,
        sample_rate=1,
        crop_size=None,
        resolution=128,
    ) -> None:
        super().__init__()
        self.real_video_files = self._combine_without_prefix(real_data_dir)
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size

    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_video_file = self.real_video_files[index]
        real_video_tensor = self._load_video(real_video_file)
        video_name = os.path.basename(real_video_file)
        return {'video': real_video_tensor, 'file_name': video_name }

    def _load_video(self, video_path):
        num_frames = self.num_frames
        sample_rate = self.sample_rate
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames > sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            s = 0
            e = total_frames
            num_frames = int(total_frames / sample_frames_len * num_frames)
            print(
                f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
                video_path,
                total_frames,
            )

        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy() 
        frames = []  
        for frame_ in range(video_data.shape[0]): 
            frame_i = video_data[frame_,:,:,:]
            frame_i = Image.fromarray(frame_i)
            if frame_i.mode != 'RGB':
                frame_i = frame_i.convert('RGB')
            frame_i = _preprocess(frame_i, crop_size=self.crop_size)
            frames.append(frame_i)
        frames = torch.stack(frames, dim=1)
        return frames

    def _combine_without_prefix(self, folder_path, prefix="."):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(os.path.join(folder_path, name))
        folder.sort()
        return folder


def _preprocess(video_data, crop_size=None):
    transform=T.Compose([
            CenterCrop([crop_size,crop_size]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]) 
    video_outputs = transform(video_data)
    return video_outputs 