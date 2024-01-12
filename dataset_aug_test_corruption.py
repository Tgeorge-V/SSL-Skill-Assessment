# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import math
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.functional import F
from utils import ioa_with_anchors, iou_with_anchors
from PIL import Image
from ipdb import set_trace
import weak_augs as weak
level_dict = {'six':0, 'seven':1, 'eight':2, 'nine':3, 'ten':4, 'eleven':5, 'twelve':6, 'thirteen':7, 'fourteen':8, 'fifteen':9, 'sixteen':10, 'seventeen':11, 'eighteen':12, 'nineteen':13, 'twenty':14, 'twenty-one':15, 'twenty-two':16, 'twenty-three':17, 'twenty-four':18, 'twenty-five':19, "twenty-six":20, "twenty-seven":21, "twenty-eight":22, "twenty-nine":23, "thirty":24}

def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data
class JigsawsDataSet(data.Dataset):
    def __init__(self, opt, transform_weak, transform_strong, subset="train", out=None,feature=None):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        #self.category = opt['category']
        if feature == 1:
            self.feature_path = opt["feature_path_1"]  
        elif feature ==2:
            self.feature_path = opt["feature_path_2"]  
        elif feature ==3:
            self.feature_path = opt["feature_path_3"]    
        elif feature ==4:
            self.feature_path = opt["feature_path_4"]    
        elif feature ==5:
            self.feature_path = opt["feature_path_5"]     
        if out == 1:
            self.video_info_path = opt["video_info_1out"]  
        elif out ==2:
            self.video_info_path = opt["video_info_2out"]  
        elif out ==3:
            self.video_info_path = opt["video_info_3out"]    
        elif out ==4:
            self.video_info_path = opt["video_info_4out"]    
        elif out ==5:
            self.video_info_path = opt["video_info_5out"]       
        self.video_anno_path = opt["video_anno"]      
        if transform_weak != 0:
            self.transform_weak = weak.weak_img_aug(transform_weak)
        else:
            self.transform_weak = None
        if transform_strong != False:
            self.feature_aug_path = opt["aug_feature_path"]
        else:
            self.feature_aug_path = None
        self._getDatasetDict()                
        self._get_match_map()    
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}            
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                if 'unlabel' not in video_subset:
                    self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())        
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))
    def _get_match_map(self):
        match_map = []
        for idx in range(self.temporal_scale):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.temporal_scale + 1):
                xmax = xmin + self.temporal_gap * jdx         
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map  # duration is same in row, start is same in col  [10000,2]
        self.anchor_xmin = [self.temporal_gap * (i-0.5) for i in range(self.temporal_scale)]  # [-0.5/100,0.5/100,...98.5/100]  
        self.anchor_xmax = [self.temporal_gap * (i+0.5) for i in range(1, self.temporal_scale + 1)]   # [1.5/100,...,100.5/100]    
    def _load_file(self, index):
        video_name = self.video_list[index]
        video_df = pd.read_csv(self.feature_path + video_name + ".csv")
        video_data = video_df.values[:, :]
        if self.subset == "validation":
            video_data = torch.Tensor(video_data)
            video_data = torch.transpose(video_data, 0, 1)
            video_data = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
            return video_data  
        else:
            video_data_weak = self.transform_weak(Image.fromarray(video_data))
            video_data = torch.Tensor(video_data)
            video_data = torch.transpose(video_data, 0, 1)
            video_data = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
            video_data_weak = torch.transpose(video_data_weak, 1, 2)
            video_data_weak = F.interpolate(video_data_weak, size=self.temporal_scale, mode='linear',align_corners=False)[0,...]    
            if self.feature_aug_path == None:
                return video_data, video_data_weak 
            else:
                video_strong_df = pd.read_csv(self.feature_aug_path + video_name + ".csv")
                video_data_strong = video_strong_df.values[:, :]
                video_data_strong_weak = self.transform_weak(Image.fromarray(video_data_strong))
                video_data_strong = torch.Tensor(video_data_strong)
                video_data_strong = torch.transpose(video_data_strong, 0, 1)
                video_data_strong = F.interpolate(video_data_strong.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
                video_data_strong_weak = torch.transpose(video_data_strong_weak, 1, 2)
                video_data_strong_weak = F.interpolate(video_data_strong_weak, size=self.temporal_scale, mode='linear',align_corners=False)[0,...]                     
                return video_data, video_data_weak, video_data_strong, video_data_strong_weak
    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]                        # video_name
        video_info = self.video_dict[video_name]                   
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame
        #print(video_name)
        #print(video_info)
        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):           
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])        # gt_bbox  [0~1]
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)   # [100*100]
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.temporal_scale, self.temporal_scale])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)           # gt [100*100]
        gt_iou_map = np.max(gt_iou_map, axis=0)        
        gt_iou_map = torch.Tensor(gt_iou_map)             # [100,100]
        ##############################################################################################

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)                # gt [start,end]
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens) 
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):                 
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        ###########classification
        # new_mask = np.zeros([self.temporal_scale])
        # for p in range(self.temporal_scale):
        #     new_mask[p] = -1
        # mask_start = int(math.floor(tmp_start * 100))
        # mask_end = int(math.floor(tmp_end * 100))
        # mask_label_idx = level_dict[tmp_info["label"]]
        # new_mask[mask_start+1:mask_end-1] = mask_label_idx
        # for p in range(self.temporal_scale):
        #     if new_mask[p]==-1:
        #         new_mask[p] = 25
        # classifier_branch = torch.Tensor(new_mask).type(torch.LongTensor)
        mask_label_idx = torch.tensor(level_dict[tmp_info["label"]] + 6)
        if self.mode == "kt":
            mask_label_idx = (mask_label_idx - 14.416666666666666) / 5.035292113340264
        elif self.mode == "np":
            mask_label_idx = (mask_label_idx - 14.285714285714286) / 4.734803834050443
        else:
            mask_label_idx = (mask_label_idx - 19.128205128205128) / 5.330990609727177
        return match_score_start, match_score_end, gt_iou_map, mask_label_idx
    def __getitem__(self, index):
        match_score_start, match_score_end, confidence_score, classifier_branch = self._get_train_label(index, self.anchor_xmin, self.anchor_xmax)
        if self.subset == "validation":
            video_data = self._load_file(index)          
            return video_data, confidence_score, match_score_start, match_score_end, classifier_branch                                                                          
        if self.mode == "train":
            if self.feature_aug_path == None:
                video_data, video_data_weak = self._load_file(index)
                return video_data, video_data_weak, confidence_score, match_score_start, match_score_end, classifier_branch  # [400,100],[100,100],[100]
            else:
                video_data, video_data_weak, video_data_strong, video_data_strong_weak = self._load_file(index)
                return video_data, video_data_weak, video_data_strong, video_data_strong_weak, confidence_score, match_score_start, match_score_end, classifier_branch
        else:
            return index, video_data
    def __len__(self):
        return len(self.video_list)
class JigsawsDataSet_unlabel(data.Dataset):
    def __init__(self, opt, transform_weak, transform_strong, subset="unlabel"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]   
        self.video_info_path = opt["video_info"]      
        self.video_anno_path = opt["video_anno"]      
        if transform_weak != 0:
            self.transform_weak = weak.weak_img_aug(transform_weak)
        else:
            self.transform_weak = None
        if transform_strong != False:
            self.feature_aug_path = opt["aug_feature_path"]
        else:
            self.feature_aug_path = None           
        self._getDatasetDict()                
        self.unlabel_percent = opt['unlabel_percent']  
        self._get_match_map()    
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}            
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = 'unseen'
        self.video_list = list(self.video_dict.keys())       
        print("%s unlabeled subset video numbers: %d" % (self.subset, len(self.video_list)))
    def _get_match_map(self):
        match_map = []
        for idx in range(self.temporal_scale):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.temporal_scale + 1):
                xmax = xmin + self.temporal_gap * jdx        
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map  # duration is same in row, start is same in col  [10000,2]
        self.anchor_xmin = [self.temporal_gap * (i-0.5) for i in range(self.temporal_scale)]  # [-0.5/100,0.5/100,...98.5/100]  
        self.anchor_xmax = [self.temporal_gap * (i+0.5) for i in range(1, self.temporal_scale + 1)]   # [1.5/100,...,100.5/100]
    def _load_file(self, index):
        video_name = self.video_list[index]
        video_df = pd.read_csv(self.feature_path + video_name + ".csv")
        video_data = video_df.values[:, :]
        video_data_weak = self.transform_weak(Image.fromarray(video_data))
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
        video_data_weak = torch.transpose(video_data_weak, 1, 2)
        video_data_weak = F.interpolate(video_data_weak, size=self.temporal_scale, mode='linear',align_corners=False)[0,...]    
        if self.feature_aug_path == None:
            return video_data, video_data_weak 
        else:
            video_strong_df = pd.read_csv(self.feature_aug_path + video_name + ".csv")
            video_data_strong = video_strong_df.values[:, :]
            video_data_strong_weak = self.transform_weak(Image.fromarray(video_data_strong))
            video_data_strong = torch.Tensor(video_data_strong)
            video_data_strong = torch.transpose(video_data_strong, 0, 1)
            video_data_strong = F.interpolate(video_data_strong.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
            video_data_strong_weak = torch.transpose(video_data_strong_weak, 1, 2)
            video_data_strong_weak = F.interpolate(video_data_strong_weak, size=self.temporal_scale, mode='linear',align_corners=False)[0,...]                     
            return video_data, video_data_weak, video_data_strong, video_data_strong_weak
    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]                        # video_name
        video_info = self.video_dict[video_name]                   
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame
        #print(video_name)
        #print(video_info)
        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):           
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])        # gt_bbox  [0~1]
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)   # [100*100]
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.temporal_scale, self.temporal_scale])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)           # gt [100*100]
        gt_iou_map = np.max(gt_iou_map, axis=0)        
        gt_iou_map = torch.Tensor(gt_iou_map)             # [100,100]
        ##############################################################################################

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)                # gt [start,end]
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens) 
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):                 
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        ###########classification
        new_mask = np.zeros([self.temporal_scale])
        for p in range(self.temporal_scale):
            new_mask[p] = -1
        mask_start = int(math.floor(tmp_start * 100))
        mask_end = int(math.floor(tmp_end * 100))
        mask_label_idx = level_dict[tmp_info["label"]]
        new_mask[mask_start+1:mask_end-1] = mask_label_idx
        for p in range(self.temporal_scale):
            if new_mask[p]==-1:
                new_mask[p] = 25
        classifier_branch = torch.Tensor(new_mask).type(torch.LongTensor)
        return match_score_start, match_score_end, gt_iou_map, classifier_branch
    def __getitem__(self, index):                                                                         
        if self.mode == "train":
            if self.feature_aug_path == None:
                video_data, video_data_weak = self._load_file(index)
                return video_data, video_data_weak
            else:
                video_data, video_data_weak, video_data_strong, video_data_strong_weak = self._load_file(index)
                return video_data, video_data_weak, video_data_strong, video_data_strong_weak
        else:
            return index, video_data
    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    kt_mean = 0.0605
    kt_std = 0.1154
    if opt['category'] == "kt":
        normalize = transforms.Normalize([kt_mean],[kt_std])
    train_loader = torch.utils.data.DataLoader(JigsawsDataSet(opt, transform_weak=2, transform_strong=True,subset="train"),
                                               batch_size=opt["label_batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=False)
    train_loader_unlabel = torch.utils.data.DataLoader(JigsawsDataSet_unlabel(opt, transform_weak=2, transform_strong=True,subset="unlabel"),  # [16,400,100]
                                                batch_size=opt["unlabel_batch_size"], shuffle=True,drop_last=True,
                                                num_workers=8, pin_memory=False)    
    for aaa,bbb,ccc,ddd in train_loader_unlabel:
        set_trace()                                             
    # for aaa,bbb,ccc,ddd,eee,fff,ggg,hhh in train_loader:           # len(train_loader)=604
    #     set_trace()
        print(aaa.shape,bbb.shape,ccc.shape,ddd.shape)  # torch.Size([16, 400, 100]) torch.Size([16, 100, 100]) torch.Size([16, 100]) torch.Size([16, 100])
        # set_trace()
        break
