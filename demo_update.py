import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as Tr
import torchvision.transforms.functional as TF


from glob import glob
from PIL import Image
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio.v2 as imageio

from utils.train_utils import restart_from_checkpoint_not_dist
from trackon.track_on_ff import TrackOnFF    # Frame Inputs
from trackon.track_on_cfg import TrackOnCfg

checkpoint_path = "./checkpoints/track_on_checkpoint.pt"
N=3

# === Helper functions ===
def read_video(video_path):
    reader = imageio.get_reader(video_path)
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    video = np.stack(frames)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # (T, 3, 720, 1920)
    
    print(f"{video.shape[0]} frames in video")
    
    return video
    
def write_gif(png_dir, out_dir):
    images = []
    
    sorted_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    
    for z, file_name in enumerate(sorted_files):    
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

    imageio.mimsave(out_dir, images, fps=30)

def get_updated_queries(t):
    if t < 11:
        return torch.tensor([[0, 587, 333], [0, 640, 345], [0, 345, 230]]), []
    elif t == 11:
        return torch.tensor([[0, 587, 333], [0, 640, 345], [10, 396, 208]]), [2]  # 396, 208
    else:
        return torch.tensor([[0, 587, 333], [0, 640, 345], [10, 396, 208]]), []

args = TrackOnCfg(checkpoint_path=checkpoint_path)
model = TrackOnFF(args)
restart_from_checkpoint_not_dist(args, run_variables={}, model=model)
distinct_colors = plt.cm.tab20(np.linspace(0, 1, N))
hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in distinct_colors]

model.cuda().eval()
model.set_memory_size(args.val_memory_size, args.val_memory_size)
model.visibility_treshold = args.val_vis_delta

# === Read video ====
video_path = "../MH_01_1.mp4"
video = read_video(video_path)  # (T, 3, H, W)
# === === ===
T = video.shape[0]

png_folder = "./out/MH_01_1_updated"
Path(png_folder).mkdir(parents=True, exist_ok=True)

vis_all = []
point_all = []

with torch.no_grad():
    for t in range(T):
    
        # === For the first frame, initialize the queries and memories ===
        if t == 0:
            queries, _ = get_updated_queries(t)
            model.init_queries_and_memory(queries.cuda(), video[t].unsqueeze(0).cuda())
        else:
            queries, removed_indices = get_updated_queries(t)
            model.update_queries_and_memory(queries.cuda(), video[t].unsqueeze(0).cuda(), removed_indices)
        N = queries.shape[0]
        # === === ===
    
        # === Model forward, for each frame ===
        
        point, vis = model.ff_forward(video[t].unsqueeze(0).cuda())
        # === === ===
    
        # === Save the predictions frame-by-frame ===
        vis_all.append(vis)
        point_all.append(point)
        
        plt.imshow(video[t].permute(1, 2, 0).long())
        for n in range(N):
            if vis[n]:
                plt.scatter(point[n, 0].cpu(), point[n, 1].cpu(), c=hex_colors[n], s=20)
    
        plt.axis("off")
        plt.savefig(os.path.join(png_folder, f"{t}.png"), bbox_inches='tight')
        plt.clf()
        # === === ===

write_gif(png_folder, os.path.join(png_folder, "out.gif"))
