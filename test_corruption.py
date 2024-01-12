import sys
from dataset_aug_test_corruption import JigsawsDataSet, JigsawsDataSet_unlabel
from loss_function import bmn_loss_func, get_mask, top_ce_loss, ce_loss
import os
import json
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.mixture import GaussianMixture
import numpy as np
import opts_test_corruption as opts
from ipdb import set_trace
from models_test import BMN, TemporalShift, TemporalShift_random
import pandas as pd
import random
from post_processing import BMN_post_processing
from eval import evaluation_proposal
import scipy.stats
import math
os.environ["CUDA_VISIBLE_DEVICES"] = '4,1,2,3'
def test_BMN(data_loader, model):
    model.eval()
    label = []
    result = []
    for n_iter, (input_data, label_confidence, label_start, label_end, top_br_gt) in enumerate(data_loader):
        input_data = input_data.cuda()
        _, top_br = model(input_data)
        label.append(top_br_gt.detach().cpu().numpy().item())
        result.append(top_br.detach().cpu().numpy().item())
    return label, result

def test_BMN_ema(data_loader, model):

    model.eval()
    label = []
    result = []
    for n_iter, (input_data, label_confidence, label_start, label_end, top_br_gt) in enumerate(data_loader):
        input_data = input_data.cuda()
        _, top_br = model(input_data)
        label.append(top_br_gt.detach().cpu().numpy().item())
        result.append(top_br.detach().cpu().numpy().item())
    return label, result

def BMN_Train_1(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model_ema = BMN(opt)
    model_ema = torch.nn.DataParallel(model_ema, device_ids=[0,1,2,3]).cuda()
    test_loader_1out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=1, feature=1),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_1out, result_1out = test_BMN(test_loader_1out, model)
    label_1out_ema, result_1out_ema = test_BMN_ema(test_loader_1out, model_ema)  
    test_loader_2out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=2, feature=1),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_2out, result_2out = test_BMN(test_loader_2out, model)
    label_2out_ema, result_2out_ema = test_BMN_ema(test_loader_2out, model_ema)
    test_loader_3out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=3, feature=1),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_3out, result_3out = test_BMN(test_loader_3out, model)
    label_3out_ema, result_3out_ema = test_BMN_ema(test_loader_3out, model_ema)  
    test_loader_4out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=4, feature=1),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_4out, result_4out = test_BMN(test_loader_4out, model)
    label_4out_ema, result_4out_ema = test_BMN_ema(test_loader_4out, model_ema)  
    test_loader_5out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=5, feature=1),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_5out, result_5out = test_BMN(test_loader_5out, model)
    label_5out_ema, result_5out_ema = test_BMN_ema(test_loader_5out, model_ema)
    pearson_1out = abs(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) else 0.0
    pearson_2out = abs(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) else 0.0
    pearson_3out = abs(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) else 0.0
    pearson_4out = abs(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) else 0.0    
    pearson_5out = abs(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) else 0.0
    spearman_1out = abs(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) else 0.0        
    spearman_2out = abs(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) else 0.0 
    spearman_3out = abs(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) else 0.0 
    spearman_4out = abs(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) else 0.0 
    spearman_5out = abs(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) else 0.0 
    z_1out = 0.5 * (math.log(1 + pearson_1out) - math.log(1 - pearson_1out))
    z_2out = 0.5 * (math.log(1 + pearson_2out) - math.log(1 - pearson_2out))
    z_3out = 0.5 * (math.log(1 + pearson_3out) - math.log(1 - pearson_3out))
    z_4out = 0.5 * (math.log(1 + pearson_4out) - math.log(1 - pearson_4out))
    z_5out = 0.5 * (math.log(1 + pearson_5out) - math.log(1 - pearson_5out))
    total_pearson = (pearson_1out + pearson_2out + pearson_3out + pearson_4out + pearson_5out) / 5
    total_spearman = (spearman_1out + spearman_2out + spearman_3out + spearman_4out + spearman_5out) / 5
    total_z = (z_1out + z_2out + z_3out + z_4out + z_5out) / 5
    print("total_spearman: ", total_spearman)
    print("total_pearson: ", total_pearson)
    print("total_z: ",total_z)    
def BMN_Train_2(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model_ema = BMN(opt)
    model_ema = torch.nn.DataParallel(model_ema, device_ids=[0,1,2,3]).cuda()
    test_loader_1out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=1, feature=2),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_1out, result_1out = test_BMN(test_loader_1out, model)
    label_1out_ema, result_1out_ema = test_BMN_ema(test_loader_1out, model_ema)  
    test_loader_2out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=2, feature=2),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_2out, result_2out = test_BMN(test_loader_2out, model)
    label_2out_ema, result_2out_ema = test_BMN_ema(test_loader_2out, model_ema)
    test_loader_3out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=3, feature=2),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_3out, result_3out = test_BMN(test_loader_3out, model)
    label_3out_ema, result_3out_ema = test_BMN_ema(test_loader_3out, model_ema)  
    test_loader_4out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=4, feature=2),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_4out, result_4out = test_BMN(test_loader_4out, model)
    label_4out_ema, result_4out_ema = test_BMN_ema(test_loader_4out, model_ema)  
    test_loader_5out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=5, feature=2),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_5out, result_5out = test_BMN(test_loader_5out, model)
    label_5out_ema, result_5out_ema = test_BMN_ema(test_loader_5out, model_ema)
    pearson_1out = abs(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) else 0.0
    pearson_2out = abs(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) else 0.0
    pearson_3out = abs(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) else 0.0
    pearson_4out = abs(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) else 0.0    
    pearson_5out = abs(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) else 0.0
    spearman_1out = abs(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) else 0.0        
    spearman_2out = abs(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) else 0.0 
    spearman_3out = abs(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) else 0.0 
    spearman_4out = abs(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) else 0.0 
    spearman_5out = abs(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) else 0.0 
    z_1out = 0.5 * (math.log(1 + pearson_1out) - math.log(1 - pearson_1out))
    z_2out = 0.5 * (math.log(1 + pearson_2out) - math.log(1 - pearson_2out))
    z_3out = 0.5 * (math.log(1 + pearson_3out) - math.log(1 - pearson_3out))
    z_4out = 0.5 * (math.log(1 + pearson_4out) - math.log(1 - pearson_4out))
    z_5out = 0.5 * (math.log(1 + pearson_5out) - math.log(1 - pearson_5out))
    total_pearson = (pearson_1out + pearson_2out + pearson_3out + pearson_4out + pearson_5out) / 5
    total_spearman = (spearman_1out + spearman_2out + spearman_3out + spearman_4out + spearman_5out) / 5
    total_z = (z_1out + z_2out + z_3out + z_4out + z_5out) / 5
    print("total_spearman: ", total_spearman)
    print("total_pearson: ", total_pearson)
    print("total_z: ",total_z)    
def BMN_Train_3(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model_ema = BMN(opt)
    model_ema = torch.nn.DataParallel(model_ema, device_ids=[0,1,2,3]).cuda()
    test_loader_1out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=1, feature=3),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_1out, result_1out = test_BMN(test_loader_1out, model)
    label_1out_ema, result_1out_ema = test_BMN_ema(test_loader_1out, model_ema)  
    test_loader_2out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=2, feature=3),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_2out, result_2out = test_BMN(test_loader_2out, model)
    label_2out_ema, result_2out_ema = test_BMN_ema(test_loader_2out, model_ema)
    test_loader_3out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=3, feature=3),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_3out, result_3out = test_BMN(test_loader_3out, model)
    label_3out_ema, result_3out_ema = test_BMN_ema(test_loader_3out, model_ema)  
    test_loader_4out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=4, feature=3),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_4out, result_4out = test_BMN(test_loader_4out, model)
    label_4out_ema, result_4out_ema = test_BMN_ema(test_loader_4out, model_ema)  
    test_loader_5out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=5, feature=3),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_5out, result_5out = test_BMN(test_loader_5out, model)
    label_5out_ema, result_5out_ema = test_BMN_ema(test_loader_5out, model_ema)
    pearson_1out = abs(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) else 0.0
    pearson_2out = abs(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) else 0.0
    pearson_3out = abs(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) else 0.0
    pearson_4out = abs(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) else 0.0    
    pearson_5out = abs(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) else 0.0
    spearman_1out = abs(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) else 0.0        
    spearman_2out = abs(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) else 0.0 
    spearman_3out = abs(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) else 0.0 
    spearman_4out = abs(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) else 0.0 
    spearman_5out = abs(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) else 0.0 
    z_1out = 0.5 * (math.log(1 + pearson_1out) - math.log(1 - pearson_1out))
    z_2out = 0.5 * (math.log(1 + pearson_2out) - math.log(1 - pearson_2out))
    z_3out = 0.5 * (math.log(1 + pearson_3out) - math.log(1 - pearson_3out))
    z_4out = 0.5 * (math.log(1 + pearson_4out) - math.log(1 - pearson_4out))
    z_5out = 0.5 * (math.log(1 + pearson_5out) - math.log(1 - pearson_5out))
    total_pearson = (pearson_1out + pearson_2out + pearson_3out + pearson_4out + pearson_5out) / 5
    total_spearman = (spearman_1out + spearman_2out + spearman_3out + spearman_4out + spearman_5out) / 5
    total_z = (z_1out + z_2out + z_3out + z_4out + z_5out) / 5
    print("total_spearman: ", total_spearman)
    print("total_pearson: ", total_pearson)
    print("total_z: ",total_z)    
def BMN_Train_4(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model_ema = BMN(opt)
    model_ema = torch.nn.DataParallel(model_ema, device_ids=[0,1,2,3]).cuda()
    test_loader_1out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=1, feature=4),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_1out, result_1out = test_BMN(test_loader_1out, model)
    label_1out_ema, result_1out_ema = test_BMN_ema(test_loader_1out, model_ema)  
    test_loader_2out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=2, feature=4),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_2out, result_2out = test_BMN(test_loader_2out, model)
    label_2out_ema, result_2out_ema = test_BMN_ema(test_loader_2out, model_ema)
    test_loader_3out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=3, feature=4),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_3out, result_3out = test_BMN(test_loader_3out, model)
    label_3out_ema, result_3out_ema = test_BMN_ema(test_loader_3out, model_ema)  
    test_loader_4out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=4, feature=4),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_4out, result_4out = test_BMN(test_loader_4out, model)
    label_4out_ema, result_4out_ema = test_BMN_ema(test_loader_4out, model_ema)  
    test_loader_5out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=5, feature=4),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_5out, result_5out = test_BMN(test_loader_5out, model)
    label_5out_ema, result_5out_ema = test_BMN_ema(test_loader_5out, model_ema)
    pearson_1out = abs(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) else 0.0
    pearson_2out = abs(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) else 0.0
    pearson_3out = abs(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) else 0.0
    pearson_4out = abs(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) else 0.0    
    pearson_5out = abs(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) else 0.0
    spearman_1out = abs(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) else 0.0        
    spearman_2out = abs(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) else 0.0 
    spearman_3out = abs(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) else 0.0 
    spearman_4out = abs(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) else 0.0 
    spearman_5out = abs(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) else 0.0 
    z_1out = 0.5 * (math.log(1 + pearson_1out) - math.log(1 - pearson_1out))
    z_2out = 0.5 * (math.log(1 + pearson_2out) - math.log(1 - pearson_2out))
    z_3out = 0.5 * (math.log(1 + pearson_3out) - math.log(1 - pearson_3out))
    z_4out = 0.5 * (math.log(1 + pearson_4out) - math.log(1 - pearson_4out))
    z_5out = 0.5 * (math.log(1 + pearson_5out) - math.log(1 - pearson_5out))
    total_pearson = (pearson_1out + pearson_2out + pearson_3out + pearson_4out + pearson_5out) / 5
    total_spearman = (spearman_1out + spearman_2out + spearman_3out + spearman_4out + spearman_5out) / 5
    total_z = (z_1out + z_2out + z_3out + z_4out + z_5out) / 5
    print("total_spearman: ", total_spearman)
    print("total_pearson: ", total_pearson)
    print("total_z: ",total_z)    
def BMN_Train_5(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model_ema = BMN(opt)
    model_ema = torch.nn.DataParallel(model_ema, device_ids=[0,1,2,3]).cuda()
    test_loader_1out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=1, feature=5),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_1out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_1out, result_1out = test_BMN(test_loader_1out, model)
    label_1out_ema, result_1out_ema = test_BMN_ema(test_loader_1out, model_ema)  
    test_loader_2out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=2, feature=5),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_2out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_2out, result_2out = test_BMN(test_loader_2out, model)
    label_2out_ema, result_2out_ema = test_BMN_ema(test_loader_2out, model_ema)
    test_loader_3out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=3, feature=5),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_3out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_3out, result_3out = test_BMN(test_loader_3out, model)
    label_3out_ema, result_3out_ema = test_BMN_ema(test_loader_3out, model_ema)  
    test_loader_4out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=4, feature=5),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_4out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_4out, result_4out = test_BMN(test_loader_4out, model)
    label_4out_ema, result_4out_ema = test_BMN_ema(test_loader_4out, model_ema)  
    test_loader_5out = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation", out=5, feature=5),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    state = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint.pth.tar")
    state_ema = torch.load(opt["checkpoint_path_5out"] + "/BMN_checkpoint_ema.pth.tar")
    model.load_state_dict(state['state_dict'])
    model_ema.load_state_dict(state_ema['state_dict'])
    label_5out, result_5out = test_BMN(test_loader_5out, model)
    label_5out_ema, result_5out_ema = test_BMN_ema(test_loader_5out, model_ema)
    pearson_1out = abs(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_1out), np.array(result_1out))[0]) else 0.0
    pearson_2out = abs(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_2out), np.array(result_2out))[0]) else 0.0
    pearson_3out = abs(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_3out), np.array(result_3out))[0]) else 0.0
    pearson_4out = abs(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_4out), np.array(result_4out))[0]) else 0.0    
    pearson_5out = abs(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label_5out), np.array(result_5out))[0]) else 0.0
    spearman_1out = abs(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_1out), np.array(result_1out))[0]) else 0.0        
    spearman_2out = abs(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_2out), np.array(result_2out))[0]) else 0.0 
    spearman_3out = abs(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_3out), np.array(result_3out))[0]) else 0.0 
    spearman_4out = abs(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_4out), np.array(result_4out))[0]) else 0.0 
    spearman_5out = abs(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label_5out), np.array(result_5out))[0]) else 0.0 
    z_1out = 0.5 * (math.log(1 + pearson_1out) - math.log(1 - pearson_1out))
    z_2out = 0.5 * (math.log(1 + pearson_2out) - math.log(1 - pearson_2out))
    z_3out = 0.5 * (math.log(1 + pearson_3out) - math.log(1 - pearson_3out))
    z_4out = 0.5 * (math.log(1 + pearson_4out) - math.log(1 - pearson_4out))
    z_5out = 0.5 * (math.log(1 + pearson_5out) - math.log(1 - pearson_5out))
    total_pearson = (pearson_1out + pearson_2out + pearson_3out + pearson_4out + pearson_5out) / 5
    total_spearman = (spearman_1out + spearman_2out + spearman_3out + spearman_4out + spearman_5out) / 5
    total_z = (z_1out + z_2out + z_3out + z_4out + z_5out) / 5
    print("total_spearman: ", total_spearman)
    print("total_pearson: ", total_pearson)
    print("total_z: ",total_z)    

def main(opt):
    if opt["mode"] == "train":
        print("Deal with Severity Level: 1")
        BMN_Train_1(opt)
        print("Deal with Severity Level: 2")
        BMN_Train_2(opt)
        print("Deal with Severity Level: 3")
        BMN_Train_3(opt)
        print("Deal with Severity Level: 4")
        BMN_Train_4(opt)
        print("Deal with Severity Level: 5")
        BMN_Train_5(opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)                                                                                                                