import sys
from dataset_aug import JigsawsDataSet, JigsawsDataSet_unlabel
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
import opts
from ipdb import set_trace
from models_new import BMN, TemporalShift, TemporalShift_random
import pandas as pd
import random
from post_processing import BMN_post_processing
from eval import evaluation_proposal
from ipdb import set_trace
import math
import scipy
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
blue = lambda x: '\033[94m' + x + '\033[0m'
sys.dont_write_bytecode = True
global_step = 0
global_step_warmup = 0
eval_loss = []
consistency_rampup = 30
consistency = 0.05  # 30  # 3  # None

criterion_mse = nn.MSELoss().cuda()
criterion_L1 = nn.L1Loss().cuda()
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_param.data * alpha + param.data * (1 - alpha)
        #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data = ema_buffer.data * alpha + buffer.data * (1 - alpha)

def contrastive_loss(x, x_aug, T=0.05):

    """
    :param x: the hidden vectors of original data
    :param x_aug: the positive vector of the auged data
    :param T: temperature
    :return: loss
    """
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    #loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = pos_sim / (sim_matrix.sum(dim=1))
    loss = - torch.log(loss).mean()
    return loss


def NCE_loss(x, x_aug):
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix)
    diagonal_matrix = torch.diag(torch.ones(batch_size))
    matrix = torch.ones(batch_size)
    #set_trace()
    p_true = sim_matrix / sim_matrix.sum(dim=1)
    mask = (matrix - diagonal_matrix).cuda()
    p_negative = sim_matrix * mask
    p_negative = p_negative / p_negative.sum(dim=1)
    loss = -torch.log(p_true / (p_true + (batch_size - 1) * p_negative)).mean()
    return loss

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    # num_classes = input_logits.size()[1]
    # return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes   # size_average=False
    return F.mse_loss(input_logits, target_logits, reduction='mean')


def softmax_kl_loss(target_logits, input_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    #return F.kl_div(input_logits, target_logits, reduction='mean')


def Motion_MSEloss(output,clip_label,motion_mask=torch.ones(100).cuda()):
    z = torch.pow((output-clip_label),2)
    loss = torch.mean(motion_mask*z)
    return loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def contrastive_loss(x, x_aug, T=0.05):

    """
    :param x: the hidden vectors of original data
    :param x_aug: the positive vector of the auged data
    :param T: temperature
    :return: loss
    """
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    #loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = pos_sim / (sim_matrix.sum(dim=1))
    loss = - torch.log(loss).mean()
    return loss

Best_metric=[0,0,0]
Best_metric_ema=[0,0,0]
def test_BMN(data_loader, model, epoch, bm_mask):
    global eval_loss
    global Best_metric
    model.eval()
    best_loss = 1e10
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    label = []
    result = []
    for n_iter, (input_data, label_confidence, label_start, label_end, top_br_gt) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        top_br_gt = top_br_gt.cuda()
        feat, top_br = model(input_data)
        #loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        loss_cls = criterion_mse(top_br.squeeze(dim=1), top_br_gt)
        #label_pred = torch.softmax(top_br,axis=1).detach().cpu().numpy()
        #label_pred = torch.softmax(torch.mean(top_br[0][:25,:], dim=1),axis=0).detach().cpu().numpy()
        #vid_label_id = np.argmax(label_pred)
        #print("vid_label_id:", vid_label_id)
        cls_id = top_br_gt.detach().cpu().numpy().item()
        label.append(cls_id)
        result.append(top_br.detach().cpu().numpy().item())
        epoch_loss_cls = loss_cls.cpu().detach().numpy()
    #set_trace()
    pearson = abs(scipy.stats.pearsonr(np.array(label), np.array(result))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label), np.array(result))[0]) else 0.0
    spearman = abs(scipy.stats.spearmanr(np.array(label), np.array(result))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label), np.array(result))[0]) else 0.0
    z = 0.5 * (math.log(1 + pearson) - math.log(1 - pearson))

    print("pearson:", pearson)
    print("spearman:", spearman)
    print("z:", z)
    print(
        blue("BMN val loss(epoch %d): cls_loss: %.03f " % (
            epoch, epoch_loss_cls / (n_iter + 1))))

    eval_loss.append(epoch_loss / (n_iter + 1))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint.pth.tar")       # ./checkpoint
    flag = 0
    if pearson >= Best_metric[0]:
        flag += 1
    if spearman >= Best_metric[1]:
        flag += 1
    if z >= Best_metric[2]:
        flag += 1
    if flag >= 2:
        Best_metric = [pearson, spearman, z]
        torch.save(state, opt["checkpoint_path"] + "/BMN_best_metric.pth.tar")
        print("Best_metric: ", Best_metric)
    else:
        Best_metric = Best_metric
    #print("flag: ", flag)
    if epoch_loss_cls < model.module.tem_best_loss:
        print("Best BMN")
        model.module.tem_best_loss = epoch_loss_cls
        torch.save(state, opt["checkpoint_path"] + "/BMN_best.pth.tar")
    opt_file = open(opt["checkpoint_path"] + "/output_eval_loss.json", "w")
    json.dump(eval_loss, opt_file)
    opt_file.close()


def test_BMN_ema(data_loader, model, epoch, bm_mask):
    model.eval()
    global Best_metric_ema
    best_loss = 1e10
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    label = []
    result = []
    for n_iter, (input_data, label_confidence, label_start, label_end, top_br_gt) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        top_br_gt = top_br_gt.cuda()
        feat, top_br = model(input_data)
        #loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        loss_cls = criterion_mse(top_br.squeeze(dim=1), top_br_gt)
        # label_pred = torch.softmax(torch.mean(top_br[0][:25,:], dim=1),axis=0).detach().cpu().numpy()
        # vid_label_id = np.argmax(label_pred)
        cls_id = top_br_gt.detach().cpu().numpy().item()
        label.append(cls_id)
        result.append(top_br.detach().cpu().numpy().item())
        epoch_loss_cls = loss_cls.cpu().detach().numpy()
    pearson = abs(scipy.stats.pearsonr(np.array(label), np.array(result))[0]) if not math.isnan(scipy.stats.pearsonr(np.array(label), np.array(result))[0]) else 0.0
    spearman = abs(scipy.stats.spearmanr(np.array(label), np.array(result))[0]) if not math.isnan(scipy.stats.spearmanr(np.array(label), np.array(result))[0]) else 0.0
    z = 0.5 * (math.log(1 + pearson) - math.log(1 - pearson))
    print("pearson:", pearson)
    print("spearman:", spearman)
    print("z:", z)
    print(
        blue("BMN val_ema loss(epoch %d): cls_loss: %.03f " % (
            epoch, epoch_loss_cls / (n_iter + 1))))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint_ema.pth.tar")       # ./checkpoint
    flag = 0
    if pearson >= Best_metric_ema[0]:
        flag += 1
    if spearman >= Best_metric_ema[1]:
        flag += 1
    if z >= Best_metric_ema[2]:
        flag += 1
    if flag >= 2:
        Best_metric_ema = [pearson, spearman, z]
        torch.save(state, opt["checkpoint_path"] + "/BMN_best_metric_ema.pth.tar")
        print("Best_metric_ema: ", Best_metric_ema)
    else:
        Best_metric_ema = Best_metric_ema
    #print("flag: ", flag)
    #if epoch_loss < model.module.tem_best_loss:
    if epoch_loss_cls < model.module.tem_best_loss:
        model.module.tem_best_loss = epoch_loss_cls                                                    
        #model.module.tem_best_loss = epoch_loss
        print("Best BMN ema")
        torch.save(state, opt["checkpoint_path"] + "/BMN_best_ema.pth.tar")

def warm_up(data_loader, model, warm_up_epoch, optimizer):
    for warm_up in range(warm_up_epoch):
        for iter_n, (input_data_1, input_data_2, label_confidence, label_start, label_end, top_br_gt) in enumerate(data_loader):
            print("Warm up Epoch: ", warm_up, "iter: ", iter_n)
            input_data = torch.cat((input_data_1, input_data_2),dim=0).cuda()
            top_br_gt = torch.cat((top_br_gt, top_br_gt), dim=0).cuda()
            feat, top_br = model(input_data)
            loss_cls = criterion_mse(top_br.squeeze(dim=1), top_br_gt) * 0.75 + criterion_L1(top_br.squeeze(dim=1), top_br_gt) * 0.25
            b_size = top_br.shape[0] // 2
            #print(top_br.shape)
            loss_con = softmax_kl_loss(feat[:b_size,:,:].squeeze(), feat[b_size:,:,:].squeeze())    
            #set_trace()
            loss_warm_up = loss_cls + 0.05*(0 - loss_con)
            optimizer.zero_grad()
            loss_warm_up.backward()
            optimizer.step()             
            print('Total Loss: ', loss_warm_up)
            print('Cls Loss: ', loss_cls)
            print('Contrastive Loss: ', loss_con)
    save_mode_path = os.path.join(opt["checkpoint_path"], 'warmup_'+str(int(warm_up)+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)            
def train_semi(data_loader, data_loader_unlabel, model, model_ema, optimizer, epoch, bm_mask):
    global global_step
    model.train()
    epoch_loss = 0
    consistency_loss_all = 0
    consistency_loss_ema_all = 0
    #consistency_criterion = softmax_mse_loss  # softmax_kl_loss
    consistency_criterion = softmax_mse_loss
    temporal_perb = TemporalShift_random(2048, 64)   
    order_clip_criterion = nn.CrossEntropyLoss()
    consistency = True
    clip_order = False
    dropout2d = True
    temporal_re = True
    unlabeled_train_iter = iter(data_loader_unlabel)
    for n_iter, (input_data_1, input_data_2, input_data_3, input_data_4, label_confidence, label_start, label_end, top_br_gt) in enumerate(data_loader):      
        print("Semi train Epoch: ", epoch, "iter: ", n_iter)
        input_data = torch.cat((input_data_1, input_data_3, input_data_2, input_data_4), dim=0).cuda()
        top_br_gt = torch.cat((top_br_gt,top_br_gt,top_br_gt,top_br_gt),dim=0).cuda()
        b_size = input_data_1.shape[0]
        input_data_student = temporal_perb(input_data)
        if dropout2d:
            input_data_student = F.dropout2d(input_data_student, 0.2)
        else:
            input_data_student = F.dropout(input_data_student, 0.2)
        feat, top_br = model(input_data_student)
        feat = feat.squeeze()
        #set_trace()
        loss_cls = criterion_mse(top_br.squeeze(dim=1), top_br_gt) * 0.75 + criterion_L1(top_br.squeeze(dim=1), top_br_gt) * 0.25
        loss_con_1 = softmax_kl_loss(feat[:b_size,:], feat[b_size: 2 * b_size,:])
        loss_con_2 = softmax_kl_loss(feat[2 * b_size:3 * b_size,:], feat[3 * b_size: 4 * b_size,:])
        if temporal_re:
            input_recons = F.dropout2d(input_data.permute(0,2,1), 0.2).permute(0,2,1)
        else:
            input_recons = F.dropout2d(input_data, 0.2)
        recons_feature = model(x=input_recons, recons=True)  
        try:
            input_data_unlabel_1, input_data_unlabel_2,input_data_unlabel_3,input_data_unlabel_4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(data_loader_unlabel)
            input_data_unlabel_1, input_data_unlabel_2,input_data_unlabel_3,input_data_unlabel_4 = unlabeled_train_iter.next()
        input_data_unlabel = torch.cat((input_data_unlabel_1, input_data_unlabel_3,input_data_unlabel_2,input_data_unlabel_4),dim=0).cuda()
        input_data_unlabel_student = temporal_perb(input_data_unlabel)
        if dropout2d:
            input_data_unlabel_student = F.dropout2d(input_data_unlabel_student, 0.2)
        else:
            input_data_unlabel_student = F.dropout(input_data_unlabel_student, 0.2)
        feat_unlabel_student, _ = model(input_data_unlabel_student)
        feat_unlabel_student = feat_unlabel_student.squeeze()
        loss_con_3 = softmax_kl_loss(feat_unlabel_student[:b_size,:], feat_unlabel_student[b_size: 2 * b_size,:])
        loss_con_4 = softmax_kl_loss(feat_unlabel_student[2 * b_size:3 * b_size,:], feat_unlabel_student[3 * b_size: 4 * b_size,:])        
        # label
        input_data_label_student_flip = F.dropout2d(input_data.flip(2).contiguous(), 0.1)     
        feat_label_student_flip, _ = model(input_data_label_student_flip)   
        feat_label_student_flip = feat_label_student_flip.squeeze()
        input_data_unlabel_student_flip = F.dropout2d(input_data_unlabel.flip(2).contiguous(), 0.1)
        feat_unlabel_student_flip,_ = model(input_data_unlabel_student_flip)
        feat_unlabel_student_flip = feat_unlabel_student_flip.squeeze()
        if temporal_re:
            recons_input_student = F.dropout2d(input_data_unlabel.permute(0,2,1), 0.2).permute(0,2,1)
        else:
            recons_input_student = F.dropout2d(input_data_unlabel, 0.2)
        recons_feature_unlabel_student = model(x=recons_input_student, recons=True)        
        loss_recons = 0.0005 * (Motion_MSEloss(recons_feature, input_data) + Motion_MSEloss(recons_feature_unlabel_student, input_data_unlabel))  # 0.0001
        with torch.no_grad():
            input_data_ema = F.dropout(input_data, 0.05)  # 0.3
            feat_teacher, _ = model_ema(input_data_ema)
            feat_teacher = feat_teacher.squeeze()
            input_data_unlabel_teacher = F.dropout(input_data_unlabel, 0.05)  # 0.3
            feat_unlabel_teacher,_ = model_ema(input_data_unlabel_teacher)
            feat_unlabel_teacher = feat_unlabel_teacher.squeeze()
            # add mask
            feat_unlabel_teacher[feat_unlabel_teacher >= 0.9] = 1.0
            feat_unlabel_teacher[feat_unlabel_teacher <= 0.1] = 0.0  # 2_add
            # flip (label)
            feat_label_teacher_flip = feat_teacher.flip(1).contiguous()
            # # flip (unlabel)
            feat_unlabel_teacher_flip = feat_unlabel_teacher.flip(1).contiguous()
            mask = torch.eq((feat_unlabel_teacher.max(1)[0] > 0.6).float(), 1.)
            feat_unlabel_teacher = feat_unlabel_teacher[mask]
            # flip
            feat_unlabel_teacher_flip = feat_unlabel_teacher_flip[mask]
        # add mask
        feat_unlabel_student = feat_unlabel_student[mask]
        # flip add mask
        feat_unlabel_student_flip = feat_unlabel_student_flip[mask]
        if consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            print("consistency_weight: ", consistency_weight)
            # meters.update('cons_weight', consistency_weight)
            # set_trace()
            consistency_loss = consistency_weight * (consistency_criterion(feat, feat_teacher))

            consistency_loss_ema = consistency_weight * (consistency_criterion(feat_unlabel_teacher, feat_unlabel_student))
            # set_trace()
            if torch.isnan(consistency_loss_ema):
                consistency_loss_ema = torch.tensor(0.).cuda()

            consistency_loss_ema_flip = 0.1 * consistency_weight * (
                    consistency_criterion(feat_unlabel_teacher_flip, feat_unlabel_student_flip)) + 0.1 * consistency_weight * (
                    consistency_criterion(feat_label_teacher_flip, feat_label_student_flip))

            # meters.update('cons_loss', consistency_loss.item())
        
        else:
            consistency_loss = torch.tensor(0).cuda()
            consistency_loss_ema = torch.tensor(0).cuda()
            consistency_loss_ema_flip = torch.tensor(0).cuda()
            # meters.update('cons_loss', 0)

        if clip_order:
            input_data_all = torch.cat([input_data, input_data_unlabel], 0)
            batch_size, C, T = input_data_all.size()
            idx = torch.randperm(batch_size)
            input_data_all_new = input_data_all[idx]
            forw_input = torch.cat(
                [input_data_all_new[:batch_size // 2, :, T // 2:], input_data_all_new[:batch_size // 2, :, :T // 2]], 2)
            back_input = input_data_all_new[batch_size // 2:, :, :]
            input_all = torch.cat([forw_input, back_input], 0)
            label_order = [0] * (batch_size // 2) + [1] * (batch_size - batch_size // 2)
            label_order = torch.tensor(label_order).long().cuda()
            out = model(x=input_all, clip_order=True)
            loss_clip_order = order_clip_criterion(out, label_order)
        print("calculating loss_cls")
        print("loss_cls:", loss_cls)
        #set_trace()
        loss_all = consistency_loss + consistency_loss_ema + loss_recons +  consistency_loss_ema_flip + loss_cls + 0.05*(0-(loss_con_1+loss_con_2+loss_con_3+loss_con_4))
        optimizer.zero_grad()
        print("backward")
        loss_all.backward()
        print("step")
        optimizer.step()
        global_step += 1
        print("update")
        with torch.no_grad():
            update_ema_variables(model, model_ema, 0.999, float(global_step/20))   # //5  //25

        consistency_loss_all += consistency_loss.cpu().detach().numpy()
        consistency_loss_ema_all += consistency_loss_ema.cpu().detach().numpy()
        if n_iter % 10 == 0:
            print(
                "training %d (epoch %d): cls_loss: %.03f, consistency_loss: %.03f, consistency_loss_ema: %.03f, consistency_loss_ema_flip: %.05f, contrastive_loss: %.05f, recon_loss: %.05f, total_loss: %.03f" % (global_step,
                    epoch, 5 * loss_cls / (n_iter + 1),
                    consistency_loss / (n_iter + 1),
                    consistency_loss_ema / (n_iter + 1),
                    consistency_loss_ema_flip / (n_iter + 1),
                    0.001*(0-(loss_con_1+loss_con_2+loss_con_3+loss_con_4)) / (n_iter + 1),
                    loss_recons / (n_iter + 1),
                    loss_all / (n_iter + 1)))

    

def BMN_Train(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model_ema = BMN(opt)
    model_ema = torch.nn.DataParallel(model_ema, device_ids=[0,1,2,3]).cuda()
    for param in model_ema.parameters():
        param.detach_()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],         
                           weight_decay=opt["weight_decay"])                               # 1e-4
    optimizer_finetune = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr_finetune"],         
                           weight_decay=opt["weight_decay"])                              
    #####################################Prepare Dataloader###########################
    train_loader_warmup = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=2, transform_strong=False,subset="train"),
                                               batch_size=opt["label_batch_size"], shuffle=True,drop_last=True,
                                               num_workers=8, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=2, transform_strong=True,subset="train"),
                                               batch_size=opt["label_batch_size"], shuffle=True,drop_last=True,
                                               num_workers=8, pin_memory=True)                                               
    if opt['use_semi'] and opt['unlabel_percent'] > 0.:
        train_loader_unlabel = torch.utils.data.DataLoader(JigsawsDataSet_unlabel(opt, transform_weak=2, transform_strong=True, subset="unlabel"),  # [16,400,100]
                                                   batch_size=opt["unlabel_batch_size"], shuffle=True,drop_last=True,
                                                   num_workers=8, pin_memory=True)    

    test_loader = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=0, transform_strong=False, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    ###################################################################################
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])    # 7     0.1
    scheduler_finetune = torch.optim.lr_scheduler.StepLR(optimizer_finetune, step_size=opt["step_size_finetune"], gamma=opt["step_gamma_finetune"])    
    bm_mask = get_mask(opt["temporal_scale"])
    use_semi = opt['use_semi']
    use_warmup = opt['use_warmup']
    print('use {} label for training!!!'.format(1-opt['unlabel_percent']))
    print('training batchsize : {}'.format(opt["label_batch_size"]))
    print('unlabel_training batchsize : {}'.format(opt["unlabel_batch_size"]))
    if use_warmup:
        warm_up(train_loader_warmup, model, opt['warm_up_epochs'], optimizer)
        model.load_state_dict(torch.load(os.path.join(opt["checkpoint_path"], 'warmup_'+str(opt['warm_up_epochs'])+'.pth')))
        test_BMN(test_loader, model, 0, bm_mask)
        #set_trace()
    for epoch in range(opt["train_epochs"]):          # 9
        # scheduler.step()
        if use_semi:
            if opt['unlabel_percent'] == 0.:
                print('use Semi !!! use all label !!!')
                train_BMN_Semi_Full(train_loader, model, model_ema, optimizer, epoch, bm_mask)
                test_BMN(test_loader, model, epoch, bm_mask)
                test_BMN_ema(test_loader, model_ema, epoch, bm_mask)
            else:
                print('use Semi !!!')
                train_semi(train_loader, train_loader_unlabel, model, model_ema, optimizer_finetune, epoch, bm_mask)
                test_BMN(test_loader, model, epoch, bm_mask)
                test_BMN_ema(test_loader, model_ema, epoch, bm_mask)
        else:
            print('use Fewer label !!!')
            train_BMN(train_loader, model, optimizer, epoch, bm_mask)
            test_BMN(test_loader, model, epoch, bm_mask)
        scheduler.step()
        scheduler_finetune.step()


def main(opt):
    if opt["mode"] == "train":
        BMN_Train(opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    main(opt)
