'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: train.py
@Time: 2021/12/02 09:59 AM
'''

import os
import math
import time
import numpy as np
from scipy import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sys import platform
import wandb
import glob
import init
from My_args import parser
import augmentations as aug
from dataset import MeshDataset, PrintDataset
from loss import AdaptiveWingLoss
from util import main_sample, save_vtk
from PAConv_model import PAConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = 'C:/Users/lowes/OneDrive/Skrivebord/DTU/8_Semester/Advaced_Geometric_DL/BU_3DFE_3DHeatmaps_crop_2/'

def train(args):
    
    if platform == "win32":
        root = 'C:/Users/lowes/OneDrive/Skrivebord/DTU/8_Semester/Advaced_Geometric_DL/BU_3DFE_3DHeatmaps_crop_2/'
    else:
        root = "/scratch/s183983/data_cropped/" 
        #if opt.user=="s183983" \
            #else "/scratch/s183986/BU_3DFE_3DHeatmaps_crop/"
            
        
    writer = SummaryWriter('runs/3D_face_alignment')
    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.dataset)
    # Dataset Random partition
    # FaceLandmark = FaceLandmarkData(partition='trainval', data=args.dataset)
    # train_size = int(len(FaceLandmark) * 0.7)
    # test_size = len(FaceLandmark) - train_size
    # torch.manual_seed(args.dataset_seed)
    # Prepare the dateset and dataloader 
    # train_dataset, test_dataset = torch.utils.data.random_split(FaceLandmark, [train_size, test_size])
    # train_loader = DataLoader(train_dataset, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, num_workers=1, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    
    train_set = MeshDataset(root,"train", args.num_points)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True, drop_last=True)
    
    val_set = MeshDataset(root,"val", args.num_points)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, 
                            num_workers=args.num_workers, shuffle=True, drop_last=True)
    
    print_set = PrintDataset(root,"val", args.batch_size, args.num_points)
    print_loader = DataLoader(print_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # data argument
    ScaleAndTranslate = aug.PointcloudScaleAndTranslate()
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5

    # select a model to train
    model = PAConv(args, 83).to(device)   # 68 in FaceScape; 8 in BU-3DFE and FRGC
    model.apply(init.weight_init)
    model = nn.DataParallel(model)

    print('let us use', torch.cuda.device_count(), 'GPUs')
    if args.loss == 'adaptive_wing':
        criterion = AdaptiveWingLoss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.0001)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=40, gamma=0.9)

    
    for epoch in range(args.epochs):
        loss_epoch, loss_val = 0.0, 0.0
        iters = 0
        model.train()
        for point, landmark, seg in train_loader:
            seg = torch.where(torch.isnan(seg), torch.full_like(seg, 0), seg)
            iters = iters + 1
            if args.no_cuda == False:
                point = point.to(device)                   # point: (Batch * num_point * num_dim)
                landmark = landmark.to(device)             # landmark : (Batch * landmark * num_dim)
                seg = seg.to(device)                       # seg: (Batch * point_num * landmark)
            point_normal = aug.normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
            point_normal = ScaleAndTranslate(point_normal)
            opt.zero_grad()
            point_normal = point_normal.permute(0, 2, 1)   # point : (batch * num_dim * num_point)
            pred_heatmap = model(point_normal)

            # Compute the loss fucntion 
            loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
            loss.backward()
            loss_epoch = loss_epoch + loss
            opt.step()
            break
            wandb.log({"loss_step": loss,
                           # "accuracy": acc,
                           # "accuracy_w": acc_w
                           })

            
            print('Epoch: [%d / %d] Train_Iter: [%d /%d] loss: %.4f' % (epoch + 1, args.epochs, iters, len(train_loader), loss))
       
        L1_mean, L2_mean = 0,0
        model.eval()
        for point, landmark, seg in val_loader:
            seg = torch.where(torch.isnan(seg), torch.full_like(seg, 0), seg)
            if args.no_cuda == False:
                point = point.to(device)                   # point: (Batch * num_point * num_dim)
                landmark = landmark.to(device)             # landmark : (Batch * landmark * num_dim)
                seg = seg.to(device)                       # seg: (Batch * point_num * landmark)
            point_normal = aug.normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
            point_normal = ScaleAndTranslate(point_normal)
            
            with torch.no_grad():
                point_normal = point_normal.permute(0, 2, 1)
                pred_heatmap = model(point_normal)
                loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                loss_val = loss_val + loss
                
                L2 = torch.sqrt(torch.pow(pred_heatmap-seg.permute(0, 2, 1),2).sum(2).sum(1))
                L1 = torch.abs(pred_heatmap-seg.permute(0, 2, 1)).sum(2).sum(1)
                L1_mean += L1.sum()
                L2_mean += L2.sum()
                
        wandb.log({"train_loss": loss_epoch,
                       "val_loss": loss_val,
                       "epoch": epoch
                       })   
                             
            
                
        L1_mean /= len(val_loader)
        L2_mean /= len(val_loader)
        wandb.log({"test_L1_mean": L1_mean,
                       "test_L2_mean": L2_mean,
                       })
        print('Epoch: [%d / %d], val L1: [%f], val L2: [%f]' % (epoch + 1, args.epochs, L1_mean, L2_mean))
        
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(),
            './checkpoints/%s/models/model_epoch_%d.pt' % (args.exp_name, epoch+1))
            
            preds = np.full([print_set.batch_size, print_set.mesh_points, 83], np.nan)
            model.eval()
            for point, landmark, seg, choice in print_loader:
                seg = torch.where(torch.isnan(seg), torch.full_like(seg, 0), seg)
                if args.no_cuda == False:
                    point = point.to(device)                   # point: (Batch * num_point * num_dim)
                    landmark = landmark.to(device)             # landmark : (Batch * landmark * num_dim)
                    seg = seg.to(device)                       # seg: (Batch * point_num * landmark)
                point_normal = aug.normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
                point_normal = ScaleAndTranslate(point_normal)
                with torch.no_grad():
                    point_normal = point_normal.permute(0, 2, 1)
                    pred_heatmap = model(point_normal).cpu()
                    
                for i in range(print_set.batch_size):
                    preds[i,choice[i],:] = pred_heatmap[i,:,:].T
                    
            pred_labels = np.nanmean(preds,axis=0)
            save_name = os.path.join('./checkpoints',args.exp_name,'meshes',print_set.file_name+'_'+str(epoch+1)+'_hm5.vtk')
            save_vtk(print_set.pd,pred_labels[:,4], save_name)
            save_name = os.path.join('./checkpoints',args.exp_name,'meshes',print_set.file_name+'_'+str(epoch+1)+'_hm42.vtk')
            save_vtk(print_set.pd,pred_labels[:,41], save_name)
            
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        writer.add_scalar('3D_Face_Alignment_loss', loss_epoch / ((epoch + 1) * len(train_loader)), epoch + 1)
        
    torch.save(model.state_dict(),
    './checkpoints/%s/models/model_epoch_%d.pt' % (args.exp_name, epoch+1))

def test(args):
    
    if platform == "win32":
        root = 'C:/Users/lowes/OneDrive/Skrivebord/DTU/8_Semester/Advaced_Geometric_DL/BU_3DFE_3DHeatmaps_crop_2/'
    else:
        root = "/scratch/s183983/data_cropped/" 
        #if opt.user=="s183983" \
            #else "/scratch/s183986/BU_3DFE_3DHeatmaps_crop/"
            
        
    writer = SummaryWriter('runs/3D_face_alignment')
    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.dataset)
    # Dataset Random partition
    # FaceLandmark = FaceLandmarkData(partition='trainval', data=args.dataset)
    # train_size = int(len(FaceLandmark) * 0.7)
    # test_size = len(FaceLandmark) - train_size
    # torch.manual_seed(args.dataset_seed)
    # Prepare the dateset and dataloader 
    # train_dataset, test_dataset = torch.utils.data.random_split(FaceLandmark, [train_size, test_size])
    # train_loader = DataLoader(train_dataset, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, num_workers=1, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    
    val_set = MeshDataset(root,"test", args.num_points)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=True, drop_last=False)
    
    print_set = PrintDataset(root,"test", args.batch_size, args.num_points)
    print_loader = DataLoader(print_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # data argument
    ScaleAndTranslate = aug.PointcloudScaleAndTranslate()


    # select a model to train
    model = PAConv(args, 83).to(device)   # 68 in FaceScape; 83 in BU-3DFE and FRGC
    model = nn.DataParallel(model)
    names = glob.glob('./checkpoints/%s/models/*.pt' % (args.exp_name))
    names.sort()
    name = names[-1]
    ckpt = torch.load(name, map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt)

    model.eval()
    L1_mean, L2_mean = 0,0
    for point, landmark, seg in val_loader:
        seg = torch.where(torch.isnan(seg), torch.full_like(seg, 0), seg)
        if args.no_cuda == False:
            point = point.to(device)                   # point: (Batch * num_point * num_dim)
            landmark = landmark.to(device)             # landmark : (Batch * landmark * num_dim)
            seg = seg.to(device)                       # seg: (Batch * point_num * landmark)
        point_normal = aug.normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
        point_normal = ScaleAndTranslate(point_normal)
        with torch.no_grad():
            point_normal = point_normal.permute(0, 2, 1)
            pred_heatmap = model(point_normal)
            
            L2 = torch.sqrt(torch.pow(pred_heatmap-seg.permute(0, 2, 1),2).sum(2).sum(1))
            L1 = torch.abs(pred_heatmap-seg.permute(0, 2, 1)).sum(2).sum(1)
            
        L1_mean += L1.sum()
        L2_mean += L2.sum()
        for (l1,l2) in zip(L1,L2):
            wandb.log({"test_L1": l1,
                           "test_L2": l1,
                           })   
            
    L1_mean /= len(val_loader)
    L2_mean /= len(val_loader)
    print("Mean L1:", L1_mean, "Mean L2:", L2_mean)
    wandb.log({"test_L1_mean": L1_mean,
                   "test_L2_mean": L2_mean,
                   })  
    preds = np.full([print_set.batch_size, print_set.mesh_points, 83], np.nan)
    
    for point, landmark, seg, choice in print_loader:
        seg = torch.where(torch.isnan(seg), torch.full_like(seg, 0), seg)
        if args.no_cuda == False:
            point = point.to(device)                   # point: (Batch * num_point * num_dim)
            landmark = landmark.to(device)             # landmark : (Batch * landmark * num_dim)
            seg = seg.to(device)                       # seg: (Batch * point_num * landmark)
        point_normal = aug.normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
        point_normal = ScaleAndTranslate(point_normal)
        model.eval()
        with torch.no_grad():
            pred_heatmap = model(point_normal).cpu()
            
        for i in range(print_loader.batch_size):
            preds[i,choice[i],:] = pred_heatmap[i,:,:]
            
    pred_labels = np.nanmean(preds,axis=0)
    save_name = os.path.join('./checkpoints',args.exp_name,'meshes',print_set.file_name+'_test_hm5.vtk')
    save_vtk(print_set.pd,pred_labels[:,4], save_name)
    save_name = os.path.join('./checkpoints',args.exp_name,'meshes',print_set.file_name+'_test_hm42.vtk')
    save_vtk(print_set.pd,pred_labels[:,41], save_name)
        
       

if __name__ == "__main__":
    # Training settings
        
    args = parser.parse_args()
    init._init_(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    wandb.init(project="3DFA-GCN", entity="s183983")
    wandb.config = {
      "learning_rate": args.lr,
      "epochs": args.epochs,
      "batch_size": args.batch_size,
      "num_points": args.num_points
    }

    train(args)
    print("Doing final test of model")
    test(args)

