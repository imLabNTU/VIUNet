import argparse
import os
from glob import glob
import pandas as pd
from rich.progress import track
from scipy.spatial.transform import Rotation as R


parser = argparse.ArgumentParser(description='Preprocess data from VIU dataset')
parser.add_argument('--dir', type=str, help='Path to the directory containing the data',required=True)
parser.add_argument('--output_dir', type=str, help='Path to the directory to output the data',required=True)
parser.add_argument('--type',type=str, help='Type of dataset, either VIU or Euroc',required=True)
args = parser.parse_args()

dir = args.dir
output_dir = args.output_dir
type = args.type

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cnt = 0
match type:
    case 'VIU':
        dirs = glob(os.path.join(dir, '*'))
        dirs.sort()
        print (f'data dirs: {dirs}, mode {type}')
        for d in dirs:
            dirs = glob(os.path.join(dir, '*'))
        dirs.sort()
        print(f'data dirs: {dirs}, mode {type}')
        cnt = 0
        for d in dirs:
            os.mkdir(os.path.join(output_dir, f'{cnt:02d}'))
            imu_data = pd.read_csv(os.path.join(d,'imu/data.csv'))
            print (f'imu data: {imu_data.columns}')
            print (f'imu data:\n {imu_data.head()}')
            imu_pt = 0
            imu_data = imu_data.values
            for i in range(imu_data.shape[0]):
                imu_data[i][0] = int(imu_data[i][0])
            #imu_data = imu_data[:,1:]
            
            gt_data = pd.read_csv(os.path.join(d,'gt/data.csv'))
            gt_pt = 0
            gt_data = gt_data.values
            print (f'gt data: {gt_data}')
            for i in range(gt_data.shape[0]):
                gt_data[i][0] = int(gt_data[i][0])
            
            uwb_data = pd.read_csv(os.path.join(d,'uwb/data.csv'))
            uwb_data = uwb_data.values
            for i in range(uwb_data.shape[0]):
                uwb_data[i][0] = int(uwb_data[i][0])
            
            
            #gt_data = gt_data[:,1:]
            
            photos = glob(os.path.join(d,'camera/data/*.png'))
            photos.sort()
            times = []
            last_photo = 0
            last_imu = 0
            last_gt = 0
            for f in photos:
                time = f.split('/')[-1].split('.')[0]
                time = int(time)
                times.append(time)
                last_photo = time
            
            last_imu = imu_data[-1][0]
            last_gt = gt_data[-1][0]
            print (f'last photo: {last_photo}, last imu: {last_imu}, last gt: {last_gt}')
            
            last = min(last_photo, last_imu, last_gt)
            
            print (f'begin photo: {times[0]}, begin imu: {imu_data[0][0]}, begin gt: {gt_data[0][0]}')
            begin = max(times[0],imu_data[0][0], gt_data[0][0])
            
            print (f'begin: {begin}, last: {last}')
            assert begin < last
            
            while imu_data[imu_pt][0] < begin:
                imu_pt += 1
            
            #truncate photo
            while times[0] < begin:
                times.pop(0)
                photos.pop(0)
            pcnt = 0
            while pcnt<len(times) and times[pcnt] <= last:
                pcnt += 1
                
            #also remove the last one
            times = times[:pcnt]
            photos = photos[:pcnt-1]

            print (len(times), len(photos))

            gt_pose = []
            uwb_pose = []
            for i in track(range(len(photos))):
                f = photos[i]
                os.system(f'cp {f} {output_dir}/{cnt:02d}/{f.split("/")[-1]}')
                imu_at_time = []
                gt_at_time = 0
                while gt_data[gt_pt][0] < int(f.split('/')[-1].split('.')[0]):
                    gt_pt += 1
                gt_at_time = gt_data[gt_pt-1][1:]
                gt_pose.append(gt_at_time)
                #Since UWB is aligned with GT, we can use gt_pt to get uwb data
                uwb_at_time = uwb_data[gt_pt-1][1:]
                uwb_pose.append(uwb_at_time)
                
                # get imu data till the next photo
                while imu_data[imu_pt][0] < times[i+1]:
                    imu_at_time.append(imu_data[imu_pt][1:])
                    imu_pt += 1
                    if imu_pt >= imu_data.shape[0]:
                        break
                imu_f = open(os.path.join(output_dir,f'{cnt:02d}',f'{f.split("/")[-1].split(".")[0]}.txt'),'w')
                for imu in imu_at_time:
                    imu_f.write(f'{imu[0]},{imu[1]},{imu[2]},{imu[3]},{imu[4]},{imu[5]}\n')
                imu_f.close()
            gt_f = open(os.path.join(output_dir,f'{cnt:02d}','poses.txt'),'w')
            uwb_f = open(os.path.join(output_dir,f'{cnt:02d}','uwb.txt'),'w')
            for gt,uwb in zip(gt_pose, uwb_pose):
                # gt is already in affine form
                gt_f.write(f'{gt[0]},{gt[1]},{gt[2]},{gt[3]},{gt[4]},{gt[5]},{gt[6]},{gt[7]},{gt[8]},{gt[9]},{gt[10]},{gt[11]}\n')
                # uwb is a 16 dim vector
                uwb_f.write(f'{uwb[0]},{uwb[1]},{uwb[2]},{uwb[3]},{uwb[4]},{uwb[5]},{uwb[6]},{uwb[7]},{uwb[8]},{uwb[9]},{uwb[10]},{uwb[11]},{uwb[12]},{uwb[13]},{uwb[14]},{uwb[15]}\n')
            gt_f.close()
            cnt += 1
        
    case 'Euroc':
        dirs = glob(os.path.join(dir, '*'))
        dirs.sort()
        print(f'data dirs: {dirs}, mode {type}')
        cnt = 0
        for d in dirs:
            os.mkdir(os.path.join(output_dir, f'{cnt:02d}'))
            imu_data = pd.read_csv(os.path.join(d,'imu0/data.csv'))
            print (f'imu data: {imu_data.columns}')
            print (f'imu data:\n {imu_data.head()}')
            imu_pt = 0
            imu_data = imu_data.values
            for i in range(imu_data.shape[0]):
                imu_data[i][0] = int(imu_data[i][0])
            #imu_data = imu_data[:,1:]
            
            gt_data = pd.read_csv(os.path.join(d,'state_groundtruth_estimate0/data.csv'))
            print (f'gt data: {gt_data.columns}')
            print (f'gt data:\n {gt_data.head()}')
            gt_pt = 0
            gt_data = gt_data.values
            print (f'gt data: {gt_data[0][0]}')
            for i in range(gt_data.shape[0]):
                gt_data[i][0] = int(gt_data[i][0])
            
            #gt_data = gt_data[:,1:]
            
            photos = glob(os.path.join(d,'cam0/data/*.png'))
            photos.sort()
            times = []
            last_photo = 0
            last_imu = 0
            last_gt = 0
            for f in photos:
                time = f.split('/')[-1].split('.')[0]
                time = int(time)
                times.append(time)
                last_photo = time
            
            last_imu = imu_data[-1][0]
            last_gt = gt_data[-1][0]
            print (f'last photo: {last_photo}, last imu: {last_imu}, last gt: {last_gt}')
            last = min(last_photo, last_imu, last_gt)
    
            begin = max(times[0],imu_data[0][0], gt_data[0][0])
            
            print (f'begin: {begin}, last: {last}')
            
            while imu_data[imu_pt][0] < begin:
                imu_pt += 1
            
            #truncate photo
            while times[0] < begin:
                times.pop(0)
                photos.pop(0)
            pcnt = 0
            while times[pcnt] <= last:
                pcnt += 1
                
            #also remove the last one
            times = times[:pcnt]
            photos = photos[:pcnt-1]
            gt_pose = []
            for i in track(range(len(photos))):
                f = photos[i]
                os.system(f'cp {f} {output_dir}/{cnt:02d}/{f.split("/")[-1]}')
                imu_at_time = []
                gt_at_time = 0
                while gt_data[gt_pt][0] < int(f.split('/')[-1].split('.')[0]):
                    gt_pt += 1
                gt_at_time = gt_data[gt_pt-1][1:]
                gt_pose.append(gt_at_time)
                # get imu data till the next photo
                while imu_data[imu_pt][0] < times[i+1]:
                    imu_at_time.append(imu_data[imu_pt][1:])
                    imu_pt += 1
                imu_f = open(os.path.join(output_dir,f'{cnt:02d}',f'{f.split("/")[-1].split(".")[0]}.txt'),'w')
                for imu in imu_at_time:
                    imu_f.write(f'{imu[0]},{imu[1]},{imu[2]},{imu[3]},{imu[4]},{imu[5]}\n')
                imu_f.close()
            gt_f = open(os.path.join(output_dir,f'{cnt:02d}','poses.txt'),'w')
            for gt in gt_pose:
                #parse to matrix
                rx,ry,rz = gt[0],gt[1],gt[2]
                tx,ty,tz = gt[3],gt[4],gt[5]
                r = R.from_euler('xyz',[rx,ry,rz],degrees=True)
                r = r.as_matrix()
                gt_f.write(f'{r[0][0]},{r[0][1]},{r[0][2]},{tx},{r[1][0]},{r[1][1]},{r[1][2]},{ty},{r[2][0]},{r[2][1]},{r[2][2]},{tz}\n')
            gt_f.close()
            cnt += 1
                
    case _:
        print (f'Dataset type {type} not supported')
        raise NotImplementedError

