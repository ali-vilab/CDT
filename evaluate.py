import subprocess 
import argparse

# Define constants for dataset paths
WEBVID_PATH = '/mnt/wulanchabu/lpd_wlcb/dataset/webvid/val'
COCO17_PATH = '/cpfs01/Group-rep-learning/multimodal/datasets/coco/val2017'

def main(args):   
    # method name
    method = args.method
    # assert method name is CDT
    assert "CDT" in method, "method must be CDT"
    # dataset name
    dataset = args.dataset
    # mode: video or image evaluation
    mode = args.mode
    # crop size
    size = args.size
    # number of frames for video evaluation
    num_frames = 17  
    # subset size: 0 means all videos; otherwise, only use subset_size videos for evaluation, you can set the subset_size to 1 for debug
    subset_size = args.subset_size

    print(f"Mode: {mode} evaluation")
    print(f"Dataset: {dataset}")

    if dataset == 'webvid':
        assert mode == 'video', "webvid dataset only supports video evaluation"
        # use constant path for webvid dataset
        real_data_dir = WEBVID_PATH
    elif dataset == 'coco17':
        assert mode == 'image', "coco17 dataset only supports image evaluation"
        # use constant path for coco17 dataset
        real_data_dir = COCO17_PATH
    else:
        raise ValueError(f"the path of dataset {dataset} is not specified")


    if "image" in mode:
        # other config
        base_set_rec = (
            '--real_data_dir {real_data_dir} '
            '--dataset {dataset} '
            '--subset_size {subset_size} '
        ).format(real_data_dir=real_data_dir, 
                dataset=dataset, subset_size=subset_size)
        command = 'python rec_image_eval.py --method {method} '.format(method=method) + base_set_rec
    else:
        # other config
        base_set_rec = (
            '--crop_size {size} '
            '--resolution {size} '
            '--num_frames {num_frames} '
            '--real_data_dir {real_data_dir} '
            '--dataset {dataset} '
            '--subset_size {subset_size} '
        ).format(size=size, num_frames=num_frames, real_data_dir=real_data_dir, 
                dataset=dataset, subset_size=subset_size)        

        command = 'python rec_video_eval.py --method {method} '.format(method=method) + base_set_rec

    print(f"Command: {command}")
    
    subprocess.call(command, shell=True) 
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='CDT_base')
    parser.add_argument('--dataset', type=str, default='webvid', choices=['webvid', 'coco17'])
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'image'])
    parser.add_argument('--size', type=str, default='256')
    parser.add_argument('--subset_size', type=int, default=0)
    args = parser.parse_args()
    
    main(args)