import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import KITTIDataset, KITTIDatasetRNN
from utils.make_label import make_quat_label, make_euler_label
from model.net import LidarNet, CameraNet, DeepLOAM, ReepLOAM
import datetime
import pytorch_model_summary

from tqdm import tqdm, trange
import numpy as np
import argparse

def test(args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else 'cpu')

    angle_dim = 3
    if args.angle == 'quaternion':
        angle_dim = 4
        make_quat_label()
    elif args.angle =='euler':
        make_euler_label()

    lidar_dir = "/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/01/velodyne_npy"
    camera_dir = "/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/01/image_3"
    label_dir = "/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/01/poses_{}.txt".format(args.angle)
    params = {
        'batch_size': 1,
        'shuffle': False,
        "num_workers": 4
    }
    hidden_dim = 512
    hidden_dims=[2*hidden_dim,512,256]

    if args.use_rnn:
        KITTIdataset = KITTIDatasetRNN("velodyne_pkl", "image_3", "poses_{}.txt".format(args.angle), ['10'], seq_length=2)
    else:
        KITTIdataset = KITTIDataset("velodyne_pkl", "image_3", "poses_{}.txt".format(args.angle), ['10'])
    test_generator = iter(KITTIdataset)

    if args.use_rnn:
        deeploam = ReepLOAM(LidarNet(input_shape=(8,400,400), hidden_dim=hidden_dim, dual=False), \
                            CameraNet(input_shape=(3,416,128), hidden_dim=hidden_dim, dual=False), \
                            hidden_dims,
                            args.angle)
        print(pytorch_model_summary.summary(deeploam, *[torch.zeros(1, 2, 8, 400, 400),torch.zeros(1, 2, 3, 416, 128), None], show_input=True))

    else:
        deeploam = DeepLOAM(LidarNet(input_shape=(8,400,400), hidden_dim=hidden_dim), \
                            CameraNet(input_shape=(3,416,128), hidden_dim=hidden_dim), \
                            hidden_dims,
                            args.angle)
        print(pytorch_model_summary.summary(deeploam, *[[torch.zeros(1, 8, 400, 400),torch.zeros(1, 8, 400, 400)],[torch.zeros(1, 3, 416, 128),torch.zeros(1, 3, 416, 128)]], show_input=True))
    
    # deeploam.load_state_dict(torch.load("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/checkpoint/ckpt_{}_best.pth".format(args.angle)))
    if args.use_rnn:
        deeploam.load_state_dict(torch.load("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/checkpoint/ckpt_ReepLOAM_euler_best_all_normalize.pth", map_location='cpu'))
    else:
        deeploam.load_state_dict(torch.load("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/checkpoint/ckpt_DeepLOAM_euler_best_all_normalize.pth", map_location='cpu'))
    deeploam.to(device)
    deeploam.eval()
    rt_np = None
    with torch.no_grad():
        loss_list, loss_rot_list, loss_trans_list = [], [], []
        h_n = None
        for i in trange(len(KITTIdataset)):
            try:
                b_lidar, b_camera, label = next(test_generator)
                if args.use_rnn:
                    b_lidar = b_lidar.unsqueeze(0).to(device)
                    b_camera = b_camera.unsqueeze(0).to(device)
                    label_rot = label[0,:angle_dim].to(device)
                    label_trans = label[0,angle_dim:].to(device)
                else:
                    b_lidar = (b_lidar[0].unsqueeze(0).to(device),b_lidar[1].unsqueeze(0).to(device))
                    b_camera = (b_camera[0].unsqueeze(0).to(device),b_camera[1].unsqueeze(0).to(device))
                    label_rot = label[:angle_dim].to(device)
                    label_trans = label[angle_dim:].to(device)

                if args.use_rnn:
                    if h_n is None:
                        pred_rot, pred_trans, h_n = deeploam(b_lidar, b_camera, h_n)
                    else:
                        pred_rot, pred_trans, h_n = deeploam(b_lidar[:,1:2], b_camera[:,1:2], h_n)
                    pred_rot, pred_trans = pred_rot.view((1,-1)), pred_trans.view((1,-1))
                else:
                    pred_rot, pred_trans = deeploam(b_lidar, b_camera)
                if args.angle=='quaternion':
                    pred_rot = nn.functional.normalize(pred_rot)

                loss_trans = nn.functional.mse_loss(pred_trans.squeeze(), label_trans)
                loss_rot = nn.functional.mse_loss(pred_rot.squeeze(), label_rot)*100
                loss = loss_trans.item()+loss_rot.item()
                loss_trans_list.append(loss_trans.item())
                loss_rot_list.append(loss_rot.item())
                loss_list.append(loss)


                rot = pred_rot.detach().cpu().numpy()
                trans = pred_trans.detach().cpu().numpy()
                rt = np.concatenate([rot,trans], axis=1)
                if rt_np is None:
                    rt_np = rt
                else:
                    rt_np = np.vstack([rt_np, rt])

            except StopIteration:
                print("finish")
        
    np.savetxt('result/{}.txt'.format(args.angle),rt_np, delimiter=' ')
    print("sigma total : ", np.mean(loss_list))
    print("sigma trans : ", np.mean(loss_trans_list))
    print("sigma rot : ", np.mean(loss_rot_list))
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", choices=['quaternion', 'euler'], default='euler')
    parser.add_argument("--use_rnn", action="store_true")
    args = parser.parse_args()
    test(args)