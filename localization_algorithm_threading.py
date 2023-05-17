import numpy as np
from scipy import optimize
import sys, collections, time
from scipy.optimize import lsq_linear, root, minimize

def ToA_stage_one(distances_to_anchors, anchor_positions): ## x 和 y 軸是準的 ，z軸在共平面時又大誤差
    distances_to_anchors, anchor_positions = np.array(distances_to_anchors), np.array(anchor_positions)
    if not np.all(distances_to_anchors):
        raise ValueError('Bad uwb connection. distances_to_anchors must never be zero. ' + str(distances_to_anchors))
    anchor_offset = anchor_positions[0]
    anchor_positions = anchor_positions[1:] - anchor_offset
    K = np.sum(np.square(anchor_positions), axis=1)   # ax=1 列加
    squared_distances_to_anchors = np.square(distances_to_anchors)
    squared_distances_to_anchors = (squared_distances_to_anchors - squared_distances_to_anchors[0])[1:]
    b = (K - squared_distances_to_anchors) / 2.
    res = lsq_linear(anchor_positions, b, lsmr_tol='auto', verbose=0)
    return res.x + anchor_offset

def ToA(distances_to_anchors, anchor_num, anchor_positions, max_xyz): ## 找較準確的z 
    distances_to_anchors, anchor_positions = np.array(distances_to_anchors), np.array(anchor_positions)
    tag_pos = ToA_stage_one(distances_to_anchors, anchor_positions)
    anc_z_ls_mean = np.mean(np.array([i[2] for i in anchor_positions]) )  
    new_z = (np.array([i[2] for i in anchor_positions]) - anc_z_ls_mean).reshape(anchor_num, 1)
    new_anc_pos = np.concatenate((np.delete(anchor_positions, 2, axis = 1), new_z ), axis=1)
    new_disto_anc = np.sqrt(abs(distances_to_anchors[:]**2 - (tag_pos[0] - new_anc_pos[:,0])**2 - (tag_pos[1] - new_anc_pos[:,1])**2))
    new_z = new_z.reshape(anchor_num,)

    a = (np.sum(new_disto_anc[:]**2) - 3*np.sum(new_z[:]**2))/len(anchor_positions)
    b = (np.sum((new_disto_anc[:]**2) * (new_z[:])) - np.sum(new_z[:]**3))/len(anchor_positions)
    cost = lambda z: np.sum(((z - new_z[:])**4 - 2*(((new_disto_anc[:])*(z - new_z[:]))**2 ) + new_disto_anc[:]**4))/len(anchor_positions) 

    function = lambda z: z**3 - a*z + b
    derivative = lambda z: 3*z**2 - a

    ranges = (slice(-4.55, 0, 1), )
    # ranges = (slice(-2, 0, 0.05), )
    resbrute = optimize.brute(cost, ranges, full_output = True, finish = optimize.fmin)
    # if resbrute[0] > 0 or resbrute[0] < -2 :
    if resbrute[0] > 0 or resbrute[0] < -4.55 :
        # print("before", [resbrute[0]])
        resbrute = optimize.brute(cost, ranges, full_output = True, finish = None)
        # print("----------------")
        resbrute = np.array([[resbrute[0]]])
        return np.array([0,0,0])
    # print([resbrute[0]])
    new_tag_pos = np.concatenate((np.delete(np.array(tag_pos), 2), resbrute[0] + anc_z_ls_mean))
    
    return np.around(new_tag_pos, 4)         


def TDoA_stage_one(anchor_num, anchor_positions, distances_differences):

    # setting which distances diffrences to use for calculate
    # (i , j, k) means using D_ik and D_jk for linearlization

    # target = ((4,1,0),(3,1,0),(2,1,0))
    target = ((4,3,2),(4,3,1),(4,3,0),(4,2,3),(4,2,1),(4,2,0),(4,1,3),(4,1,2),(4,1,0),(4,0,3),(4,0,2),(4,0,1),
              (3,2,4),(3,2,1),(3,2,0),(3,1,4),(3,1,2),(3,1,0),(3,0,4),(3,0,2),(3,0,1),(2,1,4),(2,1,3),(2,1,0),
              (2,0,4),(2,0,3),(2,0,1),(1,0,4),(1,0,3),(1,0,2))

    X = [0]*len(target)
    Y = [0]*len(target)
    Z = [0]*len(target)
    
    x = [0]*anchor_num
    y = [0]*anchor_num
    z = [0]*anchor_num

    r = [0]*anchor_num
    for i in range(anchor_num):
        r[i] = anchor_positions[i][0]**2 + \
            anchor_positions[i][1]**2+anchor_positions[i][2]**2
    for i in range(anchor_num):
        x[i] = anchor_positions[i][0]
        y[i] = anchor_positions[i][1]
        z[i] = anchor_positions[i][2]
    k = 0
    N = [0]*len(target)
    for i in target:
        N[k] = (r[i[0]]-r[i[2]])/distances_differences[i[0]][i[2]]-(r[i[1]]-r[i[2]]) / \
            distances_differences[i[1]][i[2]] - \
            distances_differences[i[0]][i[2]]+distances_differences[i[1]][i[2]]
        X[k] = 2*(x[i[0]]-x[i[2]])/distances_differences[i[0]][i[2]] - \
            2*(x[i[1]]-x[i[2]])/distances_differences[i[1]][i[2]]
        Y[k] = 2*(y[i[0]]-y[i[2]])/distances_differences[i[0]][i[2]] - \
            2*(y[i[1]]-y[i[2]])/distances_differences[i[1]][i[2]]
        Z[k] = 2*(z[i[0]]-z[i[2]])/distances_differences[i[0]][i[2]] - \
            2*(z[i[1]]-z[i[2]])/distances_differences[i[1]][i[2]]
        k = k+1
    M = [[0]*3]*len(target)

    for i in range(len(target)):
        M[i] = [X[i], Y[i], Z[i]]

    # print(M)
    Minv = np.linalg.pinv(M)

    Minv_M = np.dot(Minv, M)

    T = np.dot(Minv, N)

    x = float(T[0])
    y = float(T[1])
    z = float(T[2])
    location = [x, y, z]

    return location

def acc_z(anchor_num,anchor_positions,distances_differences,calculate_position,z_max):
    DD = distances_differences
    r_xy = anchor_num*[0]
    r_z = anchor_num*[0]
    for i in range(anchor_num):
        r_xy[i] = (anchor_positions[i][0]-calculate_position[0])**2+(anchor_positions[i][1]-calculate_position[1])**2 
        r_z[i] = anchor_positions[i][2]
    params = DD , r_xy , r_z
    def cost(z,*params):
        cost = 0
        DD,r_xy,r_z = params
        for i in range(anchor_num):
            for j in range(i):
                cost += abs(DD[i][j]-((r_xy[i]+(r_z[i]-z)**2)**0.5)+((r_xy[j]+(r_z[j]-z)**2)**0.5))
        return cost
    # ranges = (slice(0, z_max, 0.05), )
    # resbrute = optimize.brute(cost, ranges,args=params, full_output = True, finish = optimize.fmin)
    # if resbrute[0] > z_max or resbrute[0] < 0 :
    #     resbrute = optimize.brute(cost, ranges,args=params, full_output = True, finish = None)
    # new_cal_position = [calculate_position[0],calculate_position[1],float(resbrute[0])]


    ranges = (slice(-3, 0, 0.05), )
    # resbrute = optimize.brute(cost, ranges,args=params, full_output = True, finish = optimize.fmin)
    # if resbrute[0] > 0 or resbrute[0] < -2 :
        # print("before", [resbrute[0]])
    resbrute = optimize.brute(cost, ranges,args=params, full_output = True, finish = None)
    if -3 <= resbrute[0] < -2:
        
        # print("----------------")
        # resbrute = np.array([[resbrute[0]]])
        return np.array([0,0,0])

    new_cal_position = [calculate_position[0],calculate_position[1],float(resbrute[0])]
    return new_cal_position

def acc_x(anchor_num,anchor_positions,distances_differences,calculate_position,x_max):     
    DD = distances_differences
    r_yz = anchor_num*[0]
    r_x = anchor_num*[0]
    for i in range(anchor_num):
        r_yz[i] = (anchor_positions[i][1]-calculate_position[1])**2+(anchor_positions[i][2]-calculate_position[2])**2 ## (anchor_i_y-y)^2 + (anchor_i_z-z)^2  
        r_x[i] = anchor_positions[i][0]
    params = DD , r_yz , r_x
    def cost(x,*params):
        cost = 0
        DD,r_yz,r_x = params
        for i in range(anchor_num):
            for j in range(i):
                cost += abs(DD[i][j]-((r_yz[i]+(r_x[i]-x)**2)**0.5)+((r_yz[j]+(r_x[j]-x)**2)**0.5))
        return cost
    ranges = (slice(0, x_max, 0.05), )
    resbrute = optimize.brute(cost, ranges,args=params, full_output = True, finish = optimize.fmin)
    if resbrute[0] > x_max or resbrute[0] < 0 :
        resbrute = optimize.brute(cost, ranges,args=params, full_output = True, finish = None)
    new_cal_position = [float(resbrute[0]),calculate_position[1],calculate_position[2]]
    return new_cal_position

def acc_y(anchor_num,anchor_positions,distances_differences,calculate_position,y_max):     
    DD = distances_differences
    r_xz = anchor_num*[0]
    r_y = anchor_num*[0]
    for i in range(anchor_num):
        r_xz[i] = (anchor_positions[i][0]-calculate_position[0])**2+(anchor_positions[i][2]-calculate_position[2])**2 ## (anchor_i_x-x)^2 + (anchor_i_z-z)^2  
        r_y[i] = anchor_positions[i][1]
    params = DD , r_xz , r_y
    def cost(y,*params):
        cost = 0
        DD,r_xz,r_y = params
        for i in range(anchor_num):
            for j in range(i):
                cost += abs(DD[i][j]-((r_xz[i]+(r_y[i]-y)**2)**0.5)+((r_xz[j]+(r_y[j]-y)**2)**0.5))
        return cost
    ranges = (slice(0, y_max, 0.05), )
    resbrute = optimize.brute(cost, ranges,args=params, full_output = True, finish = optimize.fmin)
    if resbrute[0] > y_max or resbrute[0] < 0 :
        resbrute = optimize.brute(cost, ranges,args=params, full_output = True, finish = None)
    new_cal_position = [calculate_position[0],float(resbrute[0]),calculate_position[2]]
    return new_cal_position

def TDoA_stage_two(anchor_num,anchor,distances_differences,calculate_position,max_xyz):
    # max_xyz = max range of x , y ,z
    [x_max,y_max,z_max] = max_xyz
    new_cal_position = acc_z(anchor_num,anchor,distances_differences,calculate_position,z_max)
    calculate_position = new_cal_position
    if calculate_position[2] == 0:
        return np.array([0,0,0])
    new_cal_position = acc_y(anchor_num,anchor,distances_differences,calculate_position,y_max)
    calculate_position = new_cal_position
    new_cal_position = acc_x(anchor_num,anchor,distances_differences,calculate_position,x_max)
    return new_cal_position

def TDoA(anchor_num,anchor_positions,distances_differences,max_xyz):
    calculate_position = TDoA_stage_one(anchor_num, anchor_positions,distances_differences)
    calculate_position = TDoA_stage_two(anchor_num,anchor_positions,distances_differences,calculate_position,max_xyz)
    return calculate_position