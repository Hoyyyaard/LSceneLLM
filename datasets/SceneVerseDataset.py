import numpy as np
import os, sys
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *
from einops import rearrange
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pandas as pd
import glob
from copy import deepcopy
from scipy.spatial import cKDTree
from tqdm import tqdm
import random
import copy
import json
from transformers import AutoTokenizer

SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant.\
The assistant gives helpful, detailed, and polite answers to the human's questions.\
The visual content will be provided with the following format: <scene>visual content</scene> and\
the object bounding box will be provided with the following format: <obj>x,y,z,width,height,length</obj>\n"

TASK_PROMPT = {
    'object_caption': [
        dict(
            instruction='### human: given the 3D scene, describe this object. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: describe this object in the given 3D scene. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, localize and describe this object. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: localize and describe this object in the given 3D scene. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, describe this object first, then localize it. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
        dict(
            instruction='### human: describe then localize the object from the 3D scene. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
    ],
    'scene_qa': [
        dict(
            instruction='### human: given the 3D scene, answer the question: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer this quesiton according to the given 3D scene: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer the question: "{question}" with the related object locations in the input 3D scene. ### assistant:',
            answer='the answer is: {answer}, and the related objects are localized at {locations}',
            do_localize=True
        ),
        dict(
            instruction='### human: given the 3D scene, localize all the related objects first, then answer the question: "{question}" ### assistant:',
            answer='the related objects are localized at {locations}, the answer is: {answer}',
            do_localize=True
        ),
    ],
    'hd_scene_qa': [
        dict(
            instruction='### human: based on the 3D scene with multiple rooms, answer the question: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer this question based on the provided multi-room 3D scene: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer the question: "{question}" using the relevant object locations from the provided multi-room 3D scene. ### assistant:',
            answer='the answer is: {answer}, and the related objects are localized at {locations}',
            do_localize=True
        ),
        dict(
            instruction='### human: given the multi-room 3D scene, first locate all the relevant objects, then answer the question: "{question}" ### assistant:',
            answer='the related objects are localized at {locations}, the answer is: {answer}',
            do_localize=True
        ),
    ],
    'region_caption':[
        dict(
            instruction='### human: Describe the position of this object in relation to the surrounding objects in the 3D scene. ### assistant:',
            answer='{caption}',
            do_localize=False
        ),
        dict(
            instruction='### human: given the 3D scene, localize and describe the position of this object in relation to the surrounding objects in the 3D scene. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
    ]
    
}

def down_sample(points, colors, instance_labels=None, featrues=None, npoint=None):
    pcd_idxs = np.random.choice(len(points), size=npoint, replace=len(points) < npoint)
    points = points[pcd_idxs]
    colors = colors[pcd_idxs]
    instance_labels = instance_labels[pcd_idxs] if not instance_labels is None else None
    featrues = featrues[pcd_idxs] if not featrues is None else None
    return points, colors, instance_labels, featrues

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz

def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    box_size = np.array([xmax-xmin, ymax-ymin, zmax-zmin])
    return center, box_size

def encode_box_coords(gt_box_centers_normalized, gt_box_sizes_normalized):
    grid_size_3d = 255
    # BOX_FORMAT = '<obj><loc{}><loc{}><loc{}><whl{}><whl{}><whl{}></obj>'
    BOX_FORMAT = '<obj>{},{},{},{},{},{}</obj>'
    center_normalized = gt_box_centers_normalized
    size_normalized = gt_box_sizes_normalized
    box_normalized = np.hstack((center_normalized, size_normalized))    # (-1, 6)
    # <cx, cy, cz, w, h, l>
    box_normalized = (box_normalized * grid_size_3d).astype(np.int64)
    return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)

def scale_points(pred_xyz, mult_factor):
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz

def convert_objectpoints_to_bbox_str(points, object_points):
    center, whl = convert_pc_to_box(object_points)
    point_cloud_dims_min = points.min(axis=0)
    point_cloud_dims_max = points.max(axis=0)
    box_centers = center.astype(np.float32)
    center_normalizing_range = [
        np.zeros((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    ]
    box_centers_normalized = shift_scale_points(
        box_centers[None, ...],
        src_range=[
            point_cloud_dims_min[None, ...],
            point_cloud_dims_max[None, ...],
        ],
        dst_range=center_normalizing_range,
    )
    mult_factor = point_cloud_dims_max - point_cloud_dims_min
    box_sizes_normalized = scale_points(
        whl.astype(np.float32)[None, ...],
        mult_factor=1.0 / mult_factor[None, ...],
    )
    boxes_str = encode_box_coords(box_centers_normalized[0], box_sizes_normalized[0])
    return boxes_str

@DATASETS.register_module()
class HD_Hm3dQADataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf', add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained('ckpts/bert-base-uncased')
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        
        self.config = config
        
        self.SCENE_TOKEN = '<scene><scene_placehold></scene>'
        self.VP_TOKEN = '<vp_placehold>'
        
        special_tokens = ['<vp_placehold>', '<scene>', '<scene_placehold>', '</scene>', '<obj>', '</obj>']
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
        
        # If use openscene as encoder
        self.OPENSCENE = config.OPENSCENE
        self._openscene_root = 'data/SceneVerse/OpenScene_Scan_Features'
        
        # Load scene level data
        self._npoint = config.N_POINTS
        self._group_size = config.GROUP_SIZE
        self._num_groups = config.NUM_GROUP
        self._all_dataset_root = 'data/SceneVerse'
        
        qa_source_dir_f = 'data/SceneVerse/HM3D/annotations/qa/{}.json'.format(config.subset)
        with open(qa_source_dir_f, 'r') as f:
            datas = json.load(f)
        self.all_scene_qa = []
        for scan_name, episodes in datas.items():
            for epi in episodes: 
                for qa in epi['qa_pairs']:
                    anno = {
                        'question': qa['question'],
                        'answers': [qa['answer']],
                        'type': qa['type'],
                        'target_id': epi['target_id'],
                        'instance_type': epi['instance_type'],
                        # 'utterance': epi['utterance'],
                    }
                    self.all_scene_qa.append({'dataset_name':'HM3D', 
                                            "scan_name":scan_name, 
                                            'instance_room_id': epi['scan_id'].split('_')[-1],
                                            "anno":anno, 
                                            "task_name": "hd_scene_qa",
                                            'region_id': qa['region_id'],
                                            'episode_id':'{}#{}#{}#{}'.format('HM3D', scan_name, qa['region_id'], qa['question'])
                                            })
        print_log(f'[DATASET] {len(self.all_scene_qa)} scene qa were loaded from HM3D scan data', logger = 'HD_Hm3dQADataset')

        # Prepare corpus for evaluation
        self.corpus = {
            'hd_scene_qa': copy.deepcopy(self.all_scene_qa),
        }

    def _load_scan(self, pcd_path, inst2label_path, scan_name):
        pcd_data = torch.load(os.path.join(pcd_path, f'{scan_name}'))
        try:
            inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}"))
        except:
            inst_to_label = None
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    
        pcds = np.concatenate([points, colors], 1)
        return points, colors, pcds, instance_labels, inst_to_label
    
    def _load_scan_data(self, scan_name, dataset_name):
        dataset_root = os.path.join(self._all_dataset_root, dataset_name)
        scan_data_root = os.path.join(dataset_root, 'scan_data')
        inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
        inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}")) 
        dataset_root = os.path.join(self._openscene_root, dataset_name)
        dict = torch.load(os.path.join(dataset_root, scan_name), map_location='cpu')
        points = dict['points'].numpy().astype(np.float32)
        colors = dict['colors'].numpy()
        features = dict['features'].numpy().astype(np.float32)
        instance_labels = dict['instance_labels'].numpy()
        return points, colors, features, instance_labels, inst_to_label
    
    def __len__(self):
        return len(self.all_scene_qa)
    
    def __getitem__(self, index):
        
        data = self.all_scene_qa[index]
        dataset_name, scan_name, anno, task_name, region_id = data['dataset_name'], data['scan_name'], data['anno'], data['task_name'], data['region_id']
        instance_room_id = data['instance_room_id']
        room_center = torch.load(os.path.join('data/SceneVerse/HM3D/scan_data/room_center', '{}.pth'.format(scan_name)))
        
        self.tokenizer_config = dict(
            max_length=256, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        
        points = []
        colors = []
        features = []
        instance_labels = []
        inst_to_label = {}
        for ri, room_id in enumerate(region_id.split('-')):
            pts, cols, fts, ils, itl = self._load_scan_data(f'{scan_name}_{room_id}.pth', dataset_name)
            points.extend(pts + room_center[room_id]['center'])
            colors.extend(cols)
            features.extend(fts)
            if room_id == instance_room_id:
                instance_labels.extend(ils)
            else:
                instance_labels.extend(np.zeros_like(ils).astype(ils.dtype))
            inst_to_label[room_id] = itl
        
        points = np.array(points)
        points = pc_norm(points)
        colors = np.array(colors)
        features = np.array(features)
        instance_labels = np.array(instance_labels)


        # Get HD Info
        N = 160000
        hd_points, hd_features, hd_colors, hd_instance_labels = down_sample(points, features, colors, instance_labels, npoint=N)
    
        points, colors, instance_labels, features = down_sample(points, colors, instance_labels, features, npoint=self._npoint)

        click_query = np.zeros((1, 3))
        click_mask = np.zeros((1,))
        box_mask = np.zeros((1,))
        box_query = np.zeros((features.shape[-1],))
        ret_dict = {
            'num_groups': self._num_groups,
            'group_size': self._group_size,
            'dataset_name': dataset_name,
            'level': 'scene',
            'scan_name': scan_name,
            'task_name': task_name,
            'episode_id': data['episode_id'],
            'hd_points': hd_points.astype(np.float32),
            'hd_features': hd_features.astype(np.float32),
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
            'features': features.astype(np.float32),
            'click_query': click_query.astype(np.float32),
            'click_mask': click_mask.astype(np.float32),
            'box_mask': box_mask.astype(np.float32),
            'box_query': box_query.astype(np.float32),
            'hd_instance_labels': hd_instance_labels.astype(np.int64)
            
        }
        
        
        question = anno['question']
        prompt = deepcopy(TASK_PROMPT[task_name][0])
        boxes = ''
        intruction = prompt['instruction'].format(locations=boxes, question=question)

        # Add special token 
        intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
        prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
        answers = anno['answers'][0]
        answers = prompt['answer'].format(locations=boxes, answer=answers)
        llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
        )
        
        ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
        ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['gradient_mask'] = \
            (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        return ret_dict
