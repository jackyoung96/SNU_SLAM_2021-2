import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R


def plot3d():
    label = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/10/poses.txt")
    x,y,z = label[:,3],label[:,7],label[:,11]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,z,y,c=y, s=20, alpha=0.5)
    plt.savefig('/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/img/3d_gt.png')

def plot2d():
    label = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/10/poses.txt")
    x,y,z = label[:,3],label[:,7],label[:,11]
    fig = plt.figure()
    plt.scatter(x,z,c=y, s=20, alpha=0.5)
    plt.savefig('/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/img/2d_gt.png')

def quat2pos():
    pred = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/result/quaternion.txt")
    T = np.eye(4)
    x,y,z = [0],[0],[0]
    for i in range(pred.shape[0]):
        quat = pred[i,:4]
        trans = pred[i,4:]
        r = R.from_quat(quat)
        T_p = np.eye(4)
        T_p[:3,:3] = r.as_matrix()
        T_p[:3,3] = trans.reshape(3)
        T = np.matmul(T,T_p)

        x.append(T[0,3])
        y.append(T[1,3])
        z.append(T[2,3])
    
    fig = plt.figure()
    plt.scatter(x,z,c=y, s=20, alpha=0.5)
    plt.savefig('/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/img/quat2d.png')

def quat2pos3d():
    pred = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/result/quaternion.txt")
    T = np.eye(4)
    x,y,z = [0],[0],[0]
    for i in range(pred.shape[0]):
        quat = pred[i,:4]
        trans = pred[i,4:]
        r = R.from_quat(quat)
        T_p = np.eye(4)
        T_p[:3,:3] = r.as_matrix()
        T_p[:3,3] = trans.reshape(3)
        T = np.matmul(T,T_p)

        x.append(T[0,3])
        y.append(T[1,3])
        z.append(T[2,3])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,z,y,c=y, s=20, alpha=0.5)
    plt.savefig('/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/img/quat3d.png')

def euler2pos():
    # pred = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/00/poses_euler.txt")
    pred = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/result/DeepLOAM_euler.txt")
    T = np.eye(4)
    x,y,z = [0],[0],[0]
    for i in range(pred.shape[0]):
        euler = pred[i,:3]
        trans = pred[i,3:]
        r = R.from_euler('zyx',euler)
        T_p = np.eye(4)
        T_p[:3,:3] = r.as_matrix()
        T_p[:3,3] = trans.reshape(3)
        T = np.matmul(T,T_p)

        x.append(T[0,3])
        y.append(T[1,3])
        z.append(T[2,3])
    
    fig = plt.figure()
    plt.scatter(x,z,c=y, s=20, alpha=0.5)
    plt.savefig('/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/img/euler2d.png')

def euler2pos3d():
    pred = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/result/DeepLOAM_euler.txt")
    T = np.eye(4)
    x,y,z = [0],[0],[0]
    for i in range(pred.shape[0]):
        euler = pred[i,:3]
        trans = pred[i,3:]
        r = R.from_euler('zyx',euler)
        T_p = np.eye(4)
        T_p[:3,:3] = r.as_matrix()
        T_p[:3,3] = trans.reshape(3)
        T = np.matmul(T,T_p)

        x.append(T[0,3])
        y.append(T[1,3])
        z.append(T[2,3])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,z,y,c=y, s=20, alpha=0.5)
    plt.savefig('/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/img/euler3d.png')

# quat2pos()
# quat2pos3d()
# euler2pos()
# euler2pos3d()
# plot3d()
# plot2d()

def euler2pos_dpco():
    # pred = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/00/poses_euler.txt")
    pred = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/result/simple_relative_10.txt")
    T = np.eye(4)
    x,y,z = [0],[0],[0]
    for i in range(pred.shape[0]):
        euler = pred[i,:3]
        trans = pred[i,3:]
        r = R.from_euler('zyx',euler)
        T_p = np.eye(4)
        T_p[:3,:3] = r.as_matrix()
        T_p[:3,3] = trans.reshape(3)
        T = np.matmul(T_p,T)

        x.append(T[0,3])
        y.append(T[1,3])
        z.append(T[2,3])
    
    fig = plt.figure()
    plt.scatter(x,z,c=y, s=20, alpha=0.5)
    plt.savefig('/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/img/DPeuler2d.png')

def euler2pos3d_dpco():
    pred = np.loadtxt("/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/result/simple_relative_10.txt")
    T = np.eye(4)
    x,y,z = [0],[0],[0]
    for i in range(pred.shape[0]):
        euler = pred[i,:3]
        trans = pred[i,3:]
        r = R.from_euler('zyx',euler)
        T_p = np.eye(4)
        T_p[:3,:3] = r.as_matrix()
        T_p[:3,3] = trans.reshape(3)
        T = np.matmul(T_p,T)

        x.append(T[0,3])
        y.append(T[1,3])
        z.append(T[2,3])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,z,y,c=y, s=20, alpha=0.5)
    plt.savefig('/home/jaekyungcho/2021_2/SLAM/finalProject/DeepLOAM/img/DPeuler3d.png')

euler2pos3d_dpco()
euler2pos_dpco()