import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class LidarNet(nn.Module):
    def __init__(self, input_shape=(8,400,400), hidden_dim=1024, dual=True):
        super().__init__()
        self.dual = dual
        self.input_shape = input_shape
        resnet = models.resnet18(pretrained=False)
        self.layer0 = nn.Sequential(
            nn.Conv2d(2*input_shape[0] if dual else input_shape[0],64,3,1,1),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        dim = self.cal_dim()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU()
        )
        del resnet
        
    def cal_dim(self):
        with torch.no_grad():
            x = torch.zeros(1,*self.input_shape)
            if self.dual:
                x = torch.cat((x,x), 1)
            x = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x)))))
            x = self.avgpool(x)
        return int(np.prod(x.shape))
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

class CameraNet(nn.Module):
    def __init__(self, input_shape=(3,416,128), hidden_dim=1024, dual=True):
        super().__init__()
        self.dual = dual
        self.input_shape = input_shape
        resnet = models.resnet18(pretrained=True)
        self.layer0 = nn.Sequential(
            nn.Conv2d(2*input_shape[0] if dual else input_shape[0],64,3,1,1),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        dim = self.cal_dim()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU()
        )
        

    def cal_dim(self):
        with torch.no_grad():
            x = torch.zeros(1,*self.input_shape)
            if self.dual:
                x = torch.cat((x,x), 1)
            x = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x)))))
            x = self.avgpool(x)
        return int(np.prod(x.shape))

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

class DeepLOAM(nn.Module):
    def __init__(self, lidarnet=None, cameranet=None, hidden_dims=[2048,1024,256], angle='euler'):
        super().__init__()
        self.lidarnet = lidarnet
        self.cameranet = cameranet
        if lidarnet is None:
            self.lidarnet = LidarNet()
        if cameranet is None:
            self.cameranet = CameraNet()
        
        self.transform = nn.Sequential(
            nn.Linear(*hidden_dims[0:2]),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(*hidden_dims[1:3]),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1],3)
        )
        self.rotation = nn.Sequential(
            nn.Linear(*hidden_dims[:2]),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(*hidden_dims[1:3]),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1], 3 if angle=='euler' else 4) # quaternion
        )
        
    def forward(self, x_lidar, x_camera):
        x_lidar = torch.cat(x_lidar, 1)
        x_camera = torch.cat(x_camera, 1)
        x_lidar = self.lidarnet(x_lidar)
        x_camera = self.cameranet(x_camera)
        x = torch.cat((x_lidar,x_camera),1)
        
        transform = self.transform(x)
        rotation = self.rotation(x)
        # rotation = rotation.div((torch.norm(rotation, dim=1)+0.0001).view(-1,1))

        return rotation, transform


class ReepLOAM(nn.Module):
    def __init__(self, lidarnet=None, cameranet=None, hidden_dims=[2048,1024,256], angle='euler'):
        super().__init__()
        self.lidarnet = lidarnet
        self.cameranet = cameranet
        if lidarnet is None:
            self.lidarnet = LidarNet()
        if cameranet is None:
            self.cameranet = CameraNet()

        self.rnn = nn.RNN(input_size = hidden_dims[0],
                          hidden_size = hidden_dims[1],
                          num_layers = 2,
                          batch_first = True,
                          dropout= 0.5)
        
        self.transform = nn.Sequential(
            nn.Linear(*hidden_dims[1:3]),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1],3)
        )
        self.rotation = nn.Sequential(
            nn.Linear(*hidden_dims[1:3]),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1], 3 if angle=='euler' else 4) # quaternion
        )
        
    def forward(self, x_lidar, x_camera, h_0 = None):
        
        batchsize = x_lidar.shape[0]
        seq_length = x_lidar.shape[1]
        N = batchsize * seq_length
        x_lidar = x_lidar.view((N, *x_lidar.shape[2:]))
        x_camera = x_camera.view((N, *x_camera.shape[2:]))

        x_lidar = self.lidarnet(x_lidar)
        x_camera = self.cameranet(x_camera)
        x = torch.cat((x_lidar,x_camera),-1).view((batchsize, seq_length, -1))
        
        
        if h_0 is None:
            x,h_n = self.rnn(x)
            x = x[:,1:]
        else:
            x,h_n = self.rnn(x,h_0)
            
        x = x.reshape((-1,x.shape[-1]))

        transform = self.transform(x)
        rotation = self.rotation(x)
        if h_0 is None:
            transform = transform.view((batchsize, seq_length-1, -1))
            rotation = rotation.view((batchsize, seq_length-1, -1))
        else:
            transform = transform.view((batchsize, seq_length, -1))
            rotation = rotation.view((batchsize, seq_length, -1))

        return rotation, transform, h_n