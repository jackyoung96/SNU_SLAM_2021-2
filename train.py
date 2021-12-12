import os
import shutil
import torch
from torch.autograd import backward
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import KITTIDataset, KITTIDatasetRNN
from utils.make_label import make_quat_label, make_euler_label
from model.net import LidarNet, CameraNet, DeepLOAM, ReepLOAM
import datetime

from tqdm import tqdm
import numpy as np
import argparse

def make_Tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_logdir, sub_dir_name)

def del_Tensorboard_dir(dir_name):
    shutil.rmtree(dir_name)

def train(args):
    TB_log_dir = make_Tensorboard_dir('./log')

    writer = SummaryWriter(TB_log_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else 'cpu')
    
    angle_dim = 3
    if args.angle == 'quaternion':
        angle_dim = 4
        make_quat_label()
    elif args.angle =='euler':
        make_euler_label()
    

    lidar_dir = "/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/00/velodyne_npy"
    camera_dir = "/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/00/image_3"
    label_dir = "/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/00/poses_{}.txt".format(args.angle)
    params = {
        'batch_size': 4,
        'shuffle': True,
        "num_workers": 10
    }
    hidden_dim = 512
    hidden_dims=[2*hidden_dim,512,256]
    lr = 5e-4
    MAX_EPOCH = 100


    if args.use_rnn:
        KITTIdataset = KITTIDatasetRNN("velodyne_pkl", "image_3", "poses_{}.txt".format(args.angle), ['00','01','02','03','05','06','07','08','09'], seq_length=5)
    else:
        KITTIdataset = KITTIDataset("velodyne_pkl", "image_3", "poses_{}.txt".format(args.angle), ['00','01','02','03','05','06','07','08','09'])
    train_set, test_set = torch.utils.data.random_split(KITTIdataset, [len(KITTIdataset)-2000, 2000])
    train_generator = DataLoader(train_set, **params)
    test_generator = DataLoader(test_set, **params)

    if args.use_rnn:
        deeploam = ReepLOAM(LidarNet(input_shape=(8,400,400), hidden_dim=hidden_dim, dual=False), \
                        CameraNet(input_shape=(3,416,128), hidden_dim=hidden_dim, dual=False), \
                        hidden_dims,
                        args.angle).to(device)
    else:
        deeploam = DeepLOAM(LidarNet(input_shape=(8,400,400), hidden_dim=hidden_dim), \
                            CameraNet(input_shape=(3,416,128), hidden_dim=hidden_dim), \
                            hidden_dims,
                            args.angle).to(device)
    
    criterion_rot = nn.MSELoss()
    criterion_trans = nn.MSELoss()
    optimizer = optim.Adam(deeploam.parameters(), lr=lr)

    for epoch in range(MAX_EPOCH):
        print("Epoch : {}".format(epoch))
        training_loss = []
        rot_loss = []
        trans_loss = []
        for b_lidar, b_camera, label in tqdm(train_generator):
            if args.use_rnn:
                b_lidar = b_lidar.to(device)
                b_camera = b_camera.to(device)
                label_rot = label[:,:,:angle_dim].to(device)
                label_trans = label[:,:,angle_dim:].to(device)
            else:
                b_lidar = (b_lidar[0].to(device),b_lidar[1].to(device))
                b_camera = (b_camera[0].to(device),b_camera[1].to(device))
                label_rot = label[:,:angle_dim].to(device)
                label_trans = label[:,angle_dim:].to(device)

            optimizer.zero_grad()
            if args.use_rnn:
                pred_rot, pred_trans, _ = deeploam(b_lidar,b_camera)
            else:
                pred_rot, pred_trans = deeploam(b_lidar,b_camera)
            
            loss_trans = criterion_trans(pred_trans, label_trans)
            if args.angle=='quaternion':
                pred_rot = F.normalize(pred_rot, dim=-1)
                loss_rot = 1-(pred_rot*label_rot).sum(-1).mean()
            elif args.angle=='euler':
                loss_rot = criterion_rot(pred_rot, label_rot)

            loss_rot = loss_rot * 100
            loss = loss_trans + loss_rot
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
            trans_loss.append(loss_trans.item())
            rot_loss.append(loss_rot.item())
        
        print("training loss : {}".format(np.mean(training_loss)))
        writer.add_scalars('Loss/train', {'training loss':np.mean(training_loss),
                                        'transform loss':np.mean(trans_loss),
                                        'rotation loss':np.mean(rot_loss)}, epoch)

        with torch.no_grad():
            test_loss = []
            test_trans_loss = []
            test_rot_loss = []
            for b_lidar, b_camera, label in tqdm(test_generator):
                if args.use_rnn:
                    b_lidar = b_lidar.to(device)
                    b_camera = b_camera.to(device)
                    label_rot = label[:,:,:angle_dim].to(device)
                    label_trans = label[:,:,angle_dim:].to(device)
                else:
                    b_lidar = (b_lidar[0].to(device),b_lidar[1].to(device))
                    b_camera = (b_camera[0].to(device),b_camera[1].to(device))
                    label_rot = label[:,:angle_dim].to(device)
                    label_trans = label[:,angle_dim:].to(device)

                if args.use_rnn:
                    pred_rot, pred_trans, _ = deeploam(b_lidar,b_camera)
                else:
                    pred_rot, pred_trans = deeploam(b_lidar,b_camera)
            
                loss_trans = criterion_trans(pred_trans, label_trans)
                if args.angle=='quaternion':
                    pred_rot = F.normalize(pred_rot)
                    loss_rot = 1-(pred_rot*label_rot).sum(-1).mean()
                elif args.angle=='euler':
                    loss_rot = criterion_rot(pred_rot, label_rot)

                loss_rot = loss_rot * 100
                loss = loss_trans + loss_rot

                test_loss.append(loss.item())
                test_trans_loss.append(loss_trans.item())
                test_rot_loss.append(loss_rot.item())

        print("valid loss : {}".format(np.mean(test_loss)))
        writer.add_scalars('Loss/test', {'training loss':np.mean(test_loss),
                                        'transform loss':np.mean(test_trans_loss),
                                        'rotation loss':np.mean(test_rot_loss)}, epoch)
        if (epoch+1)%10 == 0:
            torch.save(deeploam.state_dict(), "/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/checkpoint/ckpt_%03d.pth"%(epoch+1))
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", choices=['quaternion', 'euler'], default='euler')
    parser.add_argument("--use_rnn", action="store_true")
    args = parser.parse_args()
    train(args)