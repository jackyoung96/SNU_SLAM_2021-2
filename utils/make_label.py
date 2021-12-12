import numpy as np
from scipy.spatial.transform import Rotation as R


def make_quat_label():
    for k in range(11):
        with open("/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/%02d/poses.txt"%k, 'r') as f:
            results = []
            lines = f.readlines()
            for i in range(len(lines)-1):
                pose_oa = lines[i].strip().split(" ")
                pose_ob = lines[i+1].strip().split(" ")
                pose_oa_np = np.array(pose_oa, dtype=float).reshape((3,4))
                pose_ob_np = np.array(pose_ob, dtype=float).reshape((3,4))

                Rot_ab = np.matmul(pose_oa_np[:,:3].transpose(), pose_ob_np[:,:3])
                Trans_ab = np.matmul(pose_oa_np[:,:3].transpose(),pose_ob_np[:,3:4]-pose_oa_np[:,3:4])
                Trans_ab = Trans_ab.squeeze()

                r = R.from_matrix(Rot_ab)
                Rot = r.as_quat()
                RT = np.concatenate([Rot,Trans_ab])
                results.append(RT)
            
            result = np.vstack(results)
            np.savetxt("/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/%02d/poses_quaternion.txt"%k,result)

def make_euler_label():
    for k in range(11):
        with open("/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/%02d/poses.txt"%k, 'r') as f:
            results = []
            lines = f.readlines()
            for i in range(len(lines)-1):
                pose_oa = lines[i].strip().split(" ")
                pose_ob = lines[i+1].strip().split(" ")
                pose_oa_np = np.array(pose_oa, dtype=float).reshape((3,4))
                pose_ob_np = np.array(pose_ob, dtype=float).reshape((3,4))

                Rot_ab = np.matmul(pose_oa_np[:,:3].transpose(), pose_ob_np[:,:3])
                Trans_ab = np.matmul(pose_oa_np[:,:3].transpose(),pose_ob_np[:,3:4]-pose_oa_np[:,3:4])
                Trans_ab = Trans_ab.squeeze()

                r = R.from_matrix(Rot_ab)
                Rot = r.as_euler('zyx')
                RT = np.concatenate([Rot,Trans_ab])
                results.append(RT)
            
            result = np.vstack(results)
            np.savetxt("/home/jaekyungcho/2021_2/SLAM/finalProject/KITTI/%02d/poses_euler.txt"%k,result)