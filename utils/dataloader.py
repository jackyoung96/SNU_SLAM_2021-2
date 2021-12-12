import os
import torch
from torch.utils.data import Dataset 
import numpy as np
from PIL import Image
import pickle
import gzip

def pkl_load(dir):
    with gzip.open(dir, 'rb') as f:
        return pickle.load(f)

def scaling(arr):
    arr2 = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        arr2[i] = arr[i] / (np.max(arr[i]) + 0.0001)
    return arr2

class KITTIDataset(Dataset):
    def __init__(self, lidar_dir, camera_dir, label_dir, set_idx, base_dir="/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI"):
        self.set_idx = set_idx
        self.lidar_dir = lidar_dir
        self.camera_dir = camera_dir
        # self.label_file = np.loadtxt(label_dir,delimiter=" ", dtype=float)
        self.label_dir = label_dir
        self.base_dir = base_dir

        self.set_count = [len(os.listdir(os.path.join(self.base_dir,idx,self.lidar_dir)))-1 for idx in self.set_idx]

        for idx in self.set_idx:
            if len(os.listdir(os.path.join(self.base_dir,idx,self.lidar_dir))) != len(os.listdir(os.path.join(self.base_dir,idx,self.camera_dir))):
                assert "The number of lidar and camera files are not matched"
            if len(os.listdir(os.path.join(self.base_dir,idx,self.lidar_dir))) != np.loadtxt(os.path.join(self.base_dir,idx,self.label_dir)).shape[0]+1:
                assert "The number of files and labels are not matched"
        

    def __len__(self):
        total_len = 0
        for set_idx in self.set_idx:
            total_len += len(os.listdir(os.path.join(self.base_dir,set_idx,self.lidar_dir)))-1
        return total_len

    def __getitem__(self, idx):
        set_idx = 0
        for count in self.set_count:
            if idx < count:
                break
            idx -= count
            set_idx += 1

        x_lidar1_name = os.path.join(self.base_dir, self.set_idx[set_idx], self.lidar_dir,"%06d.pickle"%idx)
        x_lidar1 = pkl_load(x_lidar1_name).transpose((2,0,1))
        x_lidar1 = scaling(x_lidar1)
        x_lidar1 = torch.Tensor(x_lidar1.copy())
        x_lidar2_name = os.path.join(self.base_dir, self.set_idx[set_idx], self.lidar_dir,"%06d.pickle"%(idx+1))
        x_lidar2 = pkl_load(x_lidar2_name).transpose((2,0,1))
        x_lidar2 = scaling(x_lidar2)
        x_lidar2 = torch.Tensor(x_lidar2.copy())

        x_camera1_name = os.path.join(self.base_dir, self.set_idx[set_idx], self.camera_dir,"%06d.png"%idx)
        x_camera1 = np.asarray(Image.open(x_camera1_name).resize((416,128))).transpose((2,1,0))
        x_camera1 = scaling(x_camera1)
        x_camera1 = torch.Tensor(x_camera1.copy())
        x_camera2_name = os.path.join(self.base_dir, self.set_idx[set_idx], self.camera_dir,"%06d.png"%(idx+1))
        x_camera2 = np.asarray(Image.open(x_camera2_name).resize((416,128))).transpose((2,1,0))
        x_camera2 = scaling(x_camera2)
        x_camera2 = torch.Tensor(x_camera2.copy())

        label = np.loadtxt(os.path.join(self.base_dir, self.set_idx[set_idx], self.label_dir),delimiter=" ", dtype=float)[idx]
        label = torch.Tensor(label.copy())

        return (x_lidar1, x_lidar2), (x_camera1, x_camera2), label

class KITTIDatasetRNN(Dataset):
    def __init__(self, lidar_dir, camera_dir, label_dir, set_idx, seq_length=10, base_dir="/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI"):

        self.set_idx = set_idx
        self.lidar_dir = lidar_dir
        self.camera_dir = camera_dir
        # self.label_file = np.loadtxt(label_dir,delimiter=" ", dtype=float)
        self.label_dir = label_dir
        self.base_dir = base_dir
        self.seq_length = seq_length

        self.set_count = [len(os.listdir(os.path.join(self.base_dir,idx,self.lidar_dir)))-(seq_length-1) for idx in self.set_idx]

        for idx in self.set_idx:
            if len(os.listdir(os.path.join(self.base_dir,idx,self.lidar_dir))) != len(os.listdir(os.path.join(self.base_dir,idx,self.camera_dir))):
                assert "The number of lidar and camera files are not matched"
            if len(os.listdir(os.path.join(self.base_dir,idx,self.lidar_dir))) != np.loadtxt(os.path.join(self.base_dir,idx,self.label_dir)).shape[0]+seq_length-1:
                assert "The number of files and labels are not matched"
        

    def __len__(self):
        return sum(self.set_count)

    def __getitem__(self, idx):
        set_idx = 0
        for count in self.set_count:
            if idx < count:
                break
            idx -= count
            set_idx += 1

        x_lidar, x_camera = None, None
        for i in range(self.seq_length):
            x_lidar_name = os.path.join(self.base_dir, self.set_idx[set_idx], self.lidar_dir,"%06d.pickle"%(idx+i))
            x_lidar_tmp = pkl_load(x_lidar_name).transpose((2,0,1))
            x_lidar_tmp = torch.Tensor(x_lidar_tmp.copy()).unsqueeze(0)
            x_camera_name = os.path.join(self.base_dir, self.set_idx[set_idx], self.camera_dir,"%06d.png"%(idx+i))
            x_camera_tmp = np.asarray(Image.open(x_camera_name).resize((416,128))).transpose((2,1,0))
            x_camera_tmp = torch.Tensor(x_camera_tmp.copy()).unsqueeze(0)
            if x_lidar is None:
                x_lidar = x_lidar_tmp
                x_camera = x_camera_tmp
            else:
                x_lidar = torch.vstack([x_lidar,x_lidar_tmp])
                x_camera = torch.vstack([x_camera, x_camera_tmp])
        
        label = np.loadtxt(os.path.join(self.base_dir, self.set_idx[set_idx], self.label_dir),delimiter=" ", dtype=float)[idx:idx+self.seq_length-1]
        label = torch.Tensor(label.copy())

        return x_lidar, x_camera, label