import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

err_mean = 0.10


for seq in range(11): 
    data = pd.read_csv(f"../select_fusion/euroc_new/{seq:02d}/poses.txt", header = None).to_numpy()
    #data = pd.read_csv(f"D:/M2DGR/{seq:02d}/poses.txt", header = None, sep = " |,", engine="python").to_numpy()
    gt_x = data[:,3]
    gt_y = data[:,7]
    gt_z = data[:,11]

    gt = np.stack((gt_x, gt_y, gt_z), axis = 1)
    print(gt.shape)

    def UWB_simulate(anchor_num, anchor_pos, ground_truth_pos):
        dis = np.empty((anchor_num))
        for i in range(anchor_num):
            dis[i] = np.round(np.linalg.norm(anchor_pos[i]-ground_truth_pos)+random.gauss(0,err_mean), 2) 
        return dis

    # anchor_pos = np.array([[-5, -10, 5], [-5, 20, 5], [20, 20, 5], [20, -10, 5]]) # euroc MH
    if seq < 5:
        # anchor_pos = np.array([[-2, -2, 2], [-2, 2, 2], [2, 2, 2], [2, -2, 2]]) # m2dgr 
        anchor_pos = np.array([[-5, -10, 5], [-5, 20, 5], [20, 20, 5], [20, -10, 5]]) # euroc MH

    else:
        anchor_pos = np.array([[-5, -5, 3], [-5, 5, 3], [5, 5, 3], [5, -5, 3]]) # euroc vicon room


    uwb = np.empty((len(gt), 4))
    # with open(f"./data/test{seq:02d}.csv", "w") as f:
    # with open(f"D:/M2DGR/{seq:02d}/uwbs_new.txt", "w") as f:
    with open(f"../select_fusion/euroc_new/{seq:02d}/uwbs.txt", "w") as f:
        for i in range(len(gt)):
            uwb[i] = UWB_simulate(4, anchor_pos, gt[i])
            f.write(f"{uwb[i][0]},{uwb[i][1]},{uwb[i][2]},{uwb[i][3]},")
            f.write(f"{anchor_pos[0][0]},{anchor_pos[0][1]},{anchor_pos[0][2]},")
            f.write(f"{anchor_pos[1][0]},{anchor_pos[1][1]},{anchor_pos[1][2]},")
            f.write(f"{anchor_pos[2][0]},{anchor_pos[2][1]},{anchor_pos[2][2]},")
            f.write(f"{anchor_pos[3][0]},{anchor_pos[3][1]},{anchor_pos[3][2]},")
            f.write(f"{gt[i][0]},{gt[i][1]},{gt[i][2]}\n")

