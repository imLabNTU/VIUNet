import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse
from re import S
import time
import csv
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
from models import FlowNet, RecFeat, PoseRegressor, RecImu, Fc_Flownet, Hard_Mask, Soft_Mask, UWBNet, VIUWBWeight, WeightLearner
from utils import save_path_formatter, mat2euler
from logger import AverageMeter
from itertools import chain
import torch.nn.functional as F
from data_loader import KITTI_Loader
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from AutomaticWeightedLoss import AutomaticWeightedLoss
from localization_algorithm_threading import *

parser = argparse.ArgumentParser(description='Selective Sensor Fusion on KITTI',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=5)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=300, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')

best_error = -1
n_iter = 0

def quaternion_loss(y_true, y_pred):
    dist1 = torch.mean(torch.abs(y_true-y_pred), axis=-1)
    dist2 = torch.mean(torch.abs(y_true+y_pred), axis=-1)
    loss = torch.where(dist1<dist2, dist1, dist2)
    return torch.mean(loss)

def main():

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    global best_error, n_iter

    args = parser.parse_args()

    n_save_model = 1
    degradation_mode = 0
    fusion_mode = 2
    # 0: vision only 1: direct 2: soft 3: hard

    # set saving path
    abs_path = Path('./results').absolute()
    save_path = save_path_formatter(args, parser)
    args.save_path = abs_path / 'checkpoints' / save_path
    print('=> will save everything to {}'.format(args.save_path))
    (args.save_path/"results").mkdir(parents=True, exist_ok=True)
    (args.save_path/"models").mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(args.seed)

    # image transform
    normalize = custom_transforms.Normalize(mean=[0, 0, 0],
                                            std=[255, 255, 255])
    normalize2 = custom_transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    input_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize,
        normalize2
    ])

    # Data loading code
    print("=> fetching scenes in '{}'".format(args.data))

    test_set = KITTI_Loader(
        args.data,
        transform=input_transform,
        seed=args.seed,
        train=2,
        sequence_length=args.sequence_length,
        data_degradation=degradation_mode, data_random=False
    )


    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)
        # , collate_fn = collate_fn_s)

    if args.epoch_size == 0:
        args.epoch_size = len(test_loader)

    # create pose model
    print("=> creating pose model")

    feature_dim = 256

    if fusion_mode == 0:
        feature_ext = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim*2).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 1:
        feature_ext = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 2:
        # feature_ext = FlowNet(args.batch_size).cuda()
        # fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        # rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        # rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        # selectfusion = Soft_Mask(feature_dim * 2, feature_dim * 2).cuda()
        # pose_net = PoseRegressor(feature_dim * 2).cuda()
        feature_ext = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024 * 3, feature_dim).cuda()
        # fc_flownet = Fc_Flownet(81920, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        uwb_encoder = UWBNet(16, feature_dim, 3).cuda()
        selectfusion = Soft_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()
        awl = AutomaticWeightedLoss(2).cuda()
        weights_model = VIUWBWeight(2).cuda()
        # weights_model = WeightLearner().cuda()

    if fusion_mode == 3:
        # feature_ext = FlowNet(args.batch_size).cuda()
        # # fc_flownet = Fc_Flownet(180224, feature_dim).cuda()
        # fc_flownet = Fc_Flownet(32*1024*3, feature_dim).cuda()
        # # fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        # rec_feat = RecFeat(feature_dim * 3, feature_dim * 3, args.batch_size, 2).cuda()
        # rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        # uwb_encoder = UWBNet(16, int(feature_dim / 2)).cuda()
        # selectfusion = Hard_Mask(feature_dim * 3, feature_dim * 3).cuda()
        # pose_net = PoseRegressor(feature_dim * 3).cuda()

        feature_ext = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024 * 3, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        uwb_encoder = UWBNet(16, feature_dim, 3).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        # selectfusion = Soft_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()
        awl = AutomaticWeightedLoss(2).cuda()
        weights_model = VIUWBWeight(2).cuda()

        # feature_ext = FlowNet(args.batch_size).cuda()
        # fc_flownet = Fc_Flownet(32 * 1024 * 3, feature_dim).cuda()
        # rec_feat = RecFeat(feature_dim * 3, feature_dim * 3, args.batch_size, 2).cuda()
        # rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        # uwb_encoder = UWBNet(16, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        # selectfusion = Hard_Mask(feature_dim * 3, feature_dim * 3).cuda()
        # pose_net = PoseRegressor(feature_dim * 3).cuda()
        # awl = AutomaticWeightedLoss(2).cuda()

    pose_net.init_weights()

    # flownet_model_path = abs_path / ".." / "select_fusion" / "pretrain" / "flownets_EPE1.951.pth"
    # pretrained_models_path = Path('./pretrain/')
    pretrained_models_path = Path('pretrain/')
    flownet_model_path = pretrained_models_path / 'flownets_EPE1.951.pth'
    
    # flownet
    weights = torch.load(str(pretrained_models_path / 'flownets_EPE1.951.pth'))
    model_dict = feature_ext.state_dict()
    update_dict = {k: v for k, v in weights['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    feature_ext.load_state_dict(model_dict)
    print('restore depth model from ' + str(flownet_model_path))

    #pretrained_models_path = Path('D:/myFusion_checkpoint/checkpoints/euroc_new/11-25-08-24/models')

#    pretrained_models_path = Path('C:/Users/imLab/Documents/myFusion/pretrain/all')
    # uwb_encoder.load_state_dict(torch.load(pretrained_models_path/"realdata"/"uwb_encoder_real2.pth"))
    uwb_encoder.load_state_dict(torch.load('pretrain/uwb_encoder.pth'))

    pretrained_models_path = pretrained_models_path / "all" 
    fc_flownet.load_state_dict(torch.load(pretrained_models_path/"fc_flownet_12.pth")) 
    rec_feat.load_state_dict(torch.load(pretrained_models_path/"rec_12.pth"))
    rec_imu.load_state_dict(torch.load(pretrained_models_path/"rec_imu_12.pth"))
    selectfusion.load_state_dict(torch.load(pretrained_models_path/"selectfusion_12.pth"))
    pose_net.load_state_dict(torch.load(pretrained_models_path/"pose_12.pth"))

    cudnn.benchmark = True
    feature_ext = torch.nn.DataParallel(feature_ext)
    rec_feat = torch.nn.DataParallel(rec_feat)
    pose_net = torch.nn.DataParallel(pose_net)
    rec_imu = torch.nn.DataParallel(rec_imu)
    uwb_encoder = torch.nn.DataParallel(uwb_encoder)
    fc_flownet = torch.nn.DataParallel(fc_flownet)
    selectfusion = torch.nn.DataParallel(selectfusion)
    # awl = torch.nn.DataParallel(awl)
    # weights_model = torch.nn.DataParallel(weights_model)

    # print('=> setting adam solver')

    # parameters = chain(rec_feat.parameters(), rec_imu.parameters(), fc_flownet.parameters(), pose_net.parameters(),
    #                     selectfusion.parameters(), awl.parameters(), weights_model.parameters())
    # optimizer = torch.optim.Adam(parameters, args.lr,
    #                             betas=(args.momentum, args.beta),
    #                             weight_decay=args.weight_decay)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    # start training loop
    print('=> training pose model')

    best_val = 100

    best_tra = 10000.0
    best_ori = 10000.0



    for epoch in range(args.epochs):

        # train for one epoch
        # evaluate on validation set
        test(args, test_loader, feature_ext, rec_feat, rec_imu, uwb_encoder, pose_net,
            fc_flownet, selectfusion, awl, weights_model, 0.5, epoch, fusion_mode)


        print('Best: {}, Best Translation {:.5} Best Orientation {:.5}'
            .format(epoch + 1, best_tra, best_ori))


def test(args, test_loader, feature_ext, rec_feat, rec_imu, uwb_encoder, pose_net,
        fc_flownet, selectfusion, awl,  weights_model, temp, epoch, fusion_mode):

    batch_time = AverageMeter()

    # switch to evaluate mode
    feature_ext.eval()
    rec_feat.eval()
    pose_net.eval()
    rec_imu.eval()
    fc_flownet.eval()
    uwb_encoder.eval()
    selectfusion.eval()
    awl.eval()
    weights_model.eval()

    end = time.time()

    aver_loss = 0
    aver_pose_loss = 0
    aver_euler_loss = 0
    aver_global_trans_loss = 0
    aver_global_rot_loss = 0
    aver_n = 0

    # print(len(test_loader))
    for i, (imgs, imus, uwbs, poses) in enumerate(test_loader):

        if i == 0:
            k = 5
        if i == 1:
            k = 7
        if i == 2:
            k = 10

        result = []
        truth_pose = []
        truth_euler = []
        predictions = []

        rec_feat.module.hidden = rec_feat.module.init_test_hidden()

        pose_loss = 0
        euler_loss = 0
        global_trans_loss = 0
        global_rot_loss = 0

        # compute output
        for j in range(0, len(imgs) - 1):

            tgt_img = imgs[j+1].cuda()
            ref_img = imgs[j].cuda()
            imu = imus[j].transpose(0, 1).cuda()
            # uwb = torch.stack([uwbs[j][:, :16].cuda(), uwbs[j+1][:, :16].cuda()], dim=0)
            uwb1 = uwbs[j][:, :16].cuda()
            uwb2 = uwbs[j+1][:, :16].cuda()

            with torch.no_grad():

                rec_imu.module.hidden = rec_imu.module.init_test_hidden()
                # uwb_encoder.module.hidden = uwb_encoder.module.init_test_hidden()
                # print(tgt_img_var.shape, ref_img_var.shape)
                raw_feature_vision = feature_ext(tgt_img, ref_img)
                #print (raw_feature_vision.shape)
                feature_vision = fc_flownet(raw_feature_vision)

                if fusion_mode == 0:

                    feature_weighted = feature_vision

                else:

                    # extract imu features
                    feature_imu = rec_imu(imu)
                    # feature_uwb = uwb_encoder(uwb)
                    # print(feature_vision.shape, feature_imu.shape)
                    if j == 0:
                        # print(uwbs[j][0, :4].numpy())
                        # print([uwbs[j][0, 4+3*N:7+3*N].numpy() for N in range(4)])
                        # uwb_pose1 = ToA(uwbs[j][0, :4].numpy(), 4, [uwbs[j][0, 4+3*N:7+3*N].numpy() for N in range(4)], max_xyz = [5,5,5])
                        # uwb_pose1 = torch.from_numpy(uwb_pose1.astype(np.float32)).cuda()
                        uwb_pose1 = uwb_encoder(uwb1)
                    # uwb_pose2 = ToA(uwbs[j+1][0, :4].numpy(), 4, [uwbs[j+1][0, 4+3*N:7+3*N].numpy() for N in range(4)], max_xyz = [5,5,5])
                    # uwb_pose2 = torch.from_numpy(uwb_pose2.astype(np.float32)).cuda()
                    uwb_pose2 = uwb_encoder(uwb2)
                    # concatenate visual and imu features
                    feature = torch.cat([feature_vision, feature_imu], 2)


                    if fusion_mode == 1:

                        feature_weighted = feature

                    else:

                        if fusion_mode == 2:
                            mask = selectfusion(feature)

                        else:
                            mask = selectfusion(feature, temp)

                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                # recurrent features
                feature_new = rec_feat(feature_weighted)

                pose = pose_net(feature_new)

            # compute pose err
            pose = pose.view(-1, 7)

            # with open('results/result_seq' + str(k) + '_' + str(epoch) + '.csv', "a") as f:
            #     f.write(pose.cpu().detach().numpy())

            trans_pose = compute_trans_pose(poses[j].cpu().data.numpy().astype(np.float64),
                                            poses[j + 1].cpu().data.numpy().astype(np.float64))

            pose_truth = torch.FloatTensor(trans_pose[:, :, -1]).cuda()

            # rot_mat = torch.FloatTensor(trans_pose[:, :, :3]).cuda()

            # euler = mat2euler(rot_mat)

            euler = torch.FloatTensor(R.from_matrix(trans_pose[:, :, :3]).as_quat()).cuda()

            pose[:, 3:] = F.normalize(pose[:, 3:], p=2, dim=1)
            pose[:, 3:] = torch.where((pose[:, 3:][:, -1]>0).unsqueeze(-1).expand(-1, pose[:, 3:].size(-1)), pose[:, 3:], -pose[:, 3:])


            if len(result) == 0:
                result = np.copy(pose.cpu().detach().numpy())
            else:
                result = np.concatenate((result, pose.cpu().detach().numpy()), axis=0)

            euler_loss += F.mse_loss(euler, pose[:, 3:])

            pose_loss += F.mse_loss(pose_truth, pose[:, :3])
            # pose_loss += F.mse_loss(poses[j][:, :, -1].squeeze(1).float().cuda(), pose[:, :3])
            # pose_loss += F.mse_loss(poses[j + 1][:, :, -1].squeeze(1).float().cuda(), pose[:, 3:6])

            if j == 0:
                # pose_t = torch.FloatTensor(poses[j][:, :, -1].float()).cuda().unsqueeze(-1)
                pose_t = uwb_pose1.unsqueeze(-1)
                pose_r = torch.FloatTensor(poses[j][:, :, :3].float()).cuda()
                # predict_pose_t = (torch.bmm(pose_r, pose[:, :3].unsqueeze(-1)) + pose_t) * 0.5 + uwb_pose2.unsqueeze(-1) * 0.5
                predict_pose_t = weights_model(torch.bmm(pose_r, pose[:, :3].unsqueeze(-1)) + pose_t, uwb_pose2.unsqueeze(-1))
                predict_pose_r = torch.bmm(pose_r, torch.FloatTensor(R.from_quat(pose[:, 3:].cpu().data.numpy()).as_matrix()).cuda())
            else:
                pose_t = predict_pose_t
                pose_r = predict_pose_r
                # predict_pose_t = (torch.bmm(pose_r, pose[:, :3].unsqueeze(-1)) + pose_t) * 0.5 + uwb_pose2.unsqueeze(-1) * 0.5
                predict_pose_t = weights_model(torch.bmm(pose_r, pose[:, :3].unsqueeze(-1)) + pose_t, uwb_pose2.unsqueeze(-1))
                predict_pose_r = torch.bmm(pose_r, torch.FloatTensor(R.from_quat(pose[:, 3:].cpu().data.numpy()).as_matrix()).cuda())
            
            if len(predictions) == 0:
                predictions = np.copy(predict_pose_t.squeeze(-1).cpu().detach().numpy())
            else:
                predictions = np.concatenate((predictions, predict_pose_t.squeeze(-1).cpu().detach().numpy()), axis=0)
                    

            global_trans_loss += F.mse_loss(poses[j+1][:, :, -1].float().cuda(), predict_pose_t.squeeze(-1))

            global_rot_loss += quaternion_loss(torch.FloatTensor(R.from_matrix(poses[j+1][:, :, :3]).as_quat()).cuda(), 
                                            torch.FloatTensor(R.from_matrix(predict_pose_r.cpu().data.numpy()).as_quat()).cuda())

            if len(truth_pose) == 0:
                truth_pose = np.copy(pose_truth.cpu().detach().numpy())
            else:
                truth_pose = np.concatenate((truth_pose, pose_truth.cpu().detach().numpy()), axis=0)

            if len(truth_euler) == 0:
                truth_euler = np.copy(euler.cpu().detach().numpy())
            else:
                truth_euler = np.concatenate((truth_euler, euler.cpu().detach().numpy()), axis=0)

        euler_loss /= (len(imgs) - 1)
        # pose_loss /= (len(imgs) - 1)
        global_trans_loss /= (len(imgs) - 1)
        global_rot_loss /= (len(imgs) - 1)

        # loss = pose_loss + euler_loss * 100
        loss = awl(euler_loss + global_rot_loss, global_trans_loss)

        # aver_pose_loss += pose_loss.item()
        aver_loss += loss.item()

        aver_euler_loss += euler_loss.item()
        aver_global_trans_loss += global_trans_loss.item()
        aver_global_rot_loss += global_rot_loss.item()

        aver_n += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test Seq{}: Epoch [{}/{}] Step [{}/{}]: Loss {:.5} '
            'Euler {:.5} Global T {:.5} Global R {:.5}'.
            format(k, epoch + 1, args.epochs, i + 1, len(test_loader), loss.item(),
                    #  euler_loss.item(), global_trans_loss.item(), global_rot_loss.item(), weights_model.params[0].item(), 1-weights_model.params[0].item()))
                    euler_loss.item(), global_trans_loss.item(), global_rot_loss.item()))

        file_name = args.save_path / "results" / f"result_seq{k}_{epoch}.csv" 
        np.savetxt(file_name, result, delimiter=',')

        file_name = args.save_path / "results" / f"truth_pose_seq{k}.csv" 
        np.savetxt(file_name, truth_pose, delimiter=',')

        file_name = args.save_path / "results" / f"truth_euler_seq{k}.csv" 
        np.savetxt(file_name, truth_euler, delimiter=',')

        file_name = args.save_path / "results" / f"result_positions_seq{k}_{epoch}.csv" 
        np.savetxt(file_name, predictions, delimiter=',')

    aver_loss /= aver_n
    aver_pose_loss /= aver_n
    aver_euler_loss /= aver_n
    aver_global_trans_loss /= aver_n
    aver_global_rot_loss /= aver_n

    print('Test Average: {}, Average_Loss {:.5} Euler_loss {:.5}, Global_trnas_loss {:.5}'
        .format(epoch + 1, aver_loss, aver_euler_loss, aver_global_trans_loss))

    return


def compute_trans_pose(ref_pose, tgt_pose):

    # print("ref_pose:", ref_pose.shape)
    tmp_pose = np.copy(tgt_pose)

    tmp_pose[:, :, -1] -= ref_pose[:, :, -1]
    trans_pose = np.linalg.inv(ref_pose[:, :, :3]) @ tmp_pose

    return trans_pose


if __name__ == '__main__':
    main()
