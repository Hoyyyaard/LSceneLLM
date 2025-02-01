from torch import nn
import torch
import copy
from models.modeling_llama import LlamaForCausalLM
from models.Point_BERT import PointTransformer
from transformers import BitsAndBytesConfig
import torch.nn.functional as nnf
from transformers import AutoTokenizer
from tools.generation_utils import generation
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    transformer_auto_wrap_policy,
    wrap,
)
import random
import einops
from models.dvae import knn_point, square_distance, Group
from utils import misc
import functools
from models.modeling_llama import LlamaDecoderLayer
import numpy as np
import os 
from utils.logger import *
from pointnet2_ops import pointnet2_utils
from tools.position_embedding import PositionEmbeddingCoordsSine
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    InstructBlipQFormerModel,
    InstructBlipQFormerConfig
)

NORM = True 

class OpensceneEncoder(nn.Module):
    def __init__(self, encoder_config):
        super(OpensceneEncoder, self).__init__()
        self.encoder_config = encoder_config
        self.openscene_features_base_dir = encoder_config.openscene_features_base_dir
        # As different levels have different number of groups, we temporally set num_group and group_size to -1
        self.pointcloud_tokenizer = Group(num_group=-1, group_size=-1)
        self.level_group_map = {
            'scene': (self.encoder_config.NUM_GROUP, self.encoder_config.GROUP_SIZE),
            'instance': (self.encoder_config.INSTANCE_NUM_GROUP, self.encoder_config.INSTANCE_GROUP_SIZE),
            'region': (self.encoder_config.REGION_NUM_GROUP, self.encoder_config.REGION_GROUP_SIZE),
        }
        self.scene_npoints = self.encoder_config.N_POINTS
        self.region_npoints = self.encoder_config.REGION_N_POINTS
        self.instance_npoints = self.encoder_config.INSTANCE_N_POINTS
        
    def forward(self, xyzs, pointcloud_features, level):
        B, _, dim = pointcloud_features.shape
        max_token_num = self.level_group_map['scene'][0]
        all_fts = torch.zeros((B, max_token_num, dim), device=pointcloud_features.device, dtype=pointcloud_features.dtype)
        all_fts_mask = torch.ones((B, max_token_num), device=pointcloud_features.device, dtype=pointcloud_features.dtype)
        scene_idx = [li for li,l in enumerate(level) if l=='scene']
        if len(scene_idx) > 0:
            self.pointcloud_tokenizer.num_group, self.pointcloud_tokenizer.group_size = self.level_group_map['scene']
            scene_fts = pointcloud_features[scene_idx][:, :self.scene_npoints, :]
            xyz = xyzs[scene_idx][:, :self.scene_npoints, :]
            scene_pointclouds = torch.cat([xyz, scene_fts], dim=-1).contiguous()
            ## batch_size, num_group, group_size, 768
            scene_neighborhood, scene_center = self.pointcloud_tokenizer(scene_pointclouds, normalize=NORM)
            ## Drop xyz
            neighborhood_xyz = scene_neighborhood[... , :3]
            scene_fts = scene_neighborhood[... , 3:].mean(-2)
            all_fts[scene_idx] = scene_fts
        else:
            assert False
        
        return all_fts, all_fts_mask, scene_center, neighborhood_xyz


class CenterGroup(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, center):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        B, N, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:,:,:3].contiguous()
            rgb = data[:, :, 3:].contiguous()
        
        batch_size, num_points, _ = xyz.shape
        
        # fps the centers out
        # center = misc.fps(xyz, self.num_group) # B G 3
        
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(batch_size, self.num_group, self.group_size, -1).contiguous()
        # normalize
        if NORM:
            neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood

        
class AdaptiveLLM(nn.Module):
    def __init__(self, llm_config, encoder_config, finetune=False, logger=None, args=None):
        super(AdaptiveLLM, self).__init__()
        self._llm_config = llm_config
        self._encoder_config = encoder_config
        self.dtype = torch.float32 if self._encoder_config.DTYPE == 'FP32' else torch.float16
        self.OPENSCENE = self._encoder_config.NAME == 'Openscene'
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf')
        self.logger = get_logger(args.log_name)
        self.args = args
        self.finetune = finetune
        
        self.FLEX = hasattr(self._encoder_config, 'FLEX')
        self._llm_config.FLEX = self.FLEX

        self._llm_config.DENSE_TOKEN_NUM = self._encoder_config.DENSE_TOKEN_NUM if hasattr(self._encoder_config, 'DENSE_TOKEN_NUM') else None
        if hasattr(self._encoder_config, 'DENSE_TOKEN_SELECT_THRESHOLD'):
            print_log("DENSE_TOKEN_SELECT_THRESHOLD is set to {}".format(self._encoder_config.DENSE_TOKEN_SELECT_THRESHOLD), logger=self.logger)
            self._llm_config.DENSE_TOKEN_SELECT_THRESHOLD = self._encoder_config.DENSE_TOKEN_SELECT_THRESHOLD 
        if hasattr(self._encoder_config, 'DENSE_TOKEN_SELECT_TOPK'):    
            self._llm_config.DENSE_TOKEN_SELECT_TOPK = self._encoder_config.DENSE_TOKEN_SELECT_TOPK
            print_log("DENSE_TOKEN_SELECT_TOPK is set to {}".format(self._encoder_config.DENSE_TOKEN_SELECT_TOPK), logger=self.logger)
        if hasattr(self._encoder_config, 'START_FLEX_LAYER'):
            self._llm_config.START_FLEX_LAYER = int(self._encoder_config.START_FLEX_LAYER)
        else:
            self._llm_config.START_FLEX_LAYER = 8
        if hasattr(self._encoder_config, 'FLEX_WARM_UP'):
            self._llm_config.FLEX_WARM_UP = int(self._encoder_config.FLEX_WARM_UP)
        else:
            self._llm_config.FLEX_WARM_UP = 20
        print_log("START_FLEX_LAYER is set to {}".format(self._llm_config.START_FLEX_LAYER), logger=self.logger)
        print_log("FLEX_WARM_UP is set to {}".format(self._llm_config.FLEX_WARM_UP), logger=self.logger)
            
        self.VISION_TOKEN_NUM = self._encoder_config.NUM_GROUP
        self._llm_config.VISION_TOKEN_NUM = self.VISION_TOKEN_NUM
        
        self._llm_config.SELECT_STRATEGY = self._encoder_config.SELECT_STRATEGY if hasattr(self._encoder_config, 'SELECT_STRATEGY') else 'NORMAL'
        print_log("SELECT_STRATEGY is set to {}".format(self._llm_config.SELECT_STRATEGY), logger=self.logger)
        
        model_path = 'ckpts/Llama-2-7b-hf'
        
        select_model = LlamaForCausalLM
        self.llm_layer = LlamaDecoderLayer
        self.llm = select_model.from_pretrained(model_path, 
                                        config=self._llm_config, 
                                        torch_dtype=self.dtype, 
                                        )
        if not self.FLEX :
            self.llm.model.gradient_checkpointing_enable()
            print_log("Gradient checkpointing is enabled", logger=self.logger)
        
        self.encoder = OpensceneEncoder(self._encoder_config)

        special_tokens = ['<vp_placehold>', '<scene>', '<scene_placehold>', '</scene>', '<obj>', '</obj>']
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Visual Prompt version
        self.visual_nquery = 8
        self.box_prompt_projector = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self.visual_nquery * self._llm_config.hidden_size),
        )
        self.click_prompt_projector = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self.visual_nquery * self._llm_config.hidden_size),
        )
        self.pos_emb3d = PositionEmbeddingCoordsSine(
            d_pos=self._encoder_config.trans_dim, 
            pos_type='fourier', 
            normalize=True
        )
        
        self.xyz_projection = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim , self._encoder_config.trans_dim),
            nn.ReLU(),
            nn.Linear(self._encoder_config.trans_dim, self._encoder_config.trans_dim),
        )
        
        self.encoder_to_llm_projection = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim , self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
        )

    def _fps(self, points, number):
        '''
            data B N 3
            number int
        '''
        fps_idx = pointnet2_utils.furthest_point_sample(points.contiguous(), number) 
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return points
    
    def torch_random_choice(self, x, num_samples, replace=True):
        """
        从张量x中随机选择num_samples个元素
        :param x: 输入张量
        :param num_samples: 需要选择的元素数量
        :param replace: 是否允许重复选择元素
        :return: 随机选择的元素
        """
        if replace:
            # 允许重复选择
            random_indices = torch.randint(0, len(x), (num_samples,))
        else:
        # 不允许重复选择
            if num_samples > len(x):
                raise ValueError("num_samples不能超过输入张量的长度")
            random_indices = torch.randperm(len(x))[:num_samples]
        
        return x[random_indices]
    
    def _get_hd_corpus(self, data_dict, center, neighborhood_xyz):
        '''
            neighborhood_xyz: [bs, 128, 384, 3]
        '''
        # Extract {TOKEN_NUM} dense tokens per token
        TOKEN_NUM = self._llm_config.DENSE_TOKEN_NUM
        bs = center.shape[0]
        # 1. Sample the surrounding points of dense point clouds according to the midpoint of sparse point clouds
        # 2. {TOKEN_NUM} dense tokens that are re-sampled for each sampled point cloud group
        DENSE_GROUP_SIZE = 1500
        group_tokenizer = CenterGroup(num_group=center.shape[1], group_size=DENSE_GROUP_SIZE)
        dense_points = data_dict['hd_points']
        dense_features = data_dict['hd_features']
    
        dense_scene_pointclouds = torch.cat([dense_points, dense_features], dim=-1).contiguous()
        dense_neighborhood = group_tokenizer(dense_scene_pointclouds, center) 
        # # [bsz, 128, DENSE_GROUP_SIZE, 3]
        # dense_neighborhood_xyz = dense_neighborhood[... , :3]
        # # [bsz, 128, DENSE_GROUP_SIZE, 768]
        # dense_neighborhood_fts = dense_neighborhood[... , 3:]
        tokenizer = Group(num_group=TOKEN_NUM, group_size=int(DENSE_GROUP_SIZE/TOKEN_NUM))
        dense_token_neighborhood, dense_token_center = tokenizer(dense_neighborhood.view(-1, dense_neighborhood.shape[-2], dense_neighborhood.shape[-1]).contiguous(), normalize=NORM)
        dense_token_neighborhood = dense_token_neighborhood.view(bs, -1, dense_token_neighborhood.shape[-2], dense_token_neighborhood.shape[-1]).contiguous()
        dense_token_center = dense_token_center.view(bs, -1, dense_token_center.shape[-1]).contiguous()
        # [bs, 128*TOKEN_NUM, 768]
        dense_fts = dense_token_neighborhood[... , 3:].mean(-2)

        point_cloud_dims_min, _ = dense_points[..., :3].min(dim=1)
        point_cloud_dims_max, _ = dense_points[..., :3].max(dim=1)
        point_cloud_dims = [
            point_cloud_dims_min,
            point_cloud_dims_max,
        ]
        
        return {
            # 'dense_points':dense_points, 
            'dense_center':dense_token_center,
            'point_cloud_dims': point_cloud_dims,
            'dense_features':dense_fts.view(bs, -1, TOKEN_NUM, dense_fts.shape[-1]).contiguous()
        }
        
    def wrap_model(self):
        if self._encoder_config.distributed == 'DDP':
            return self.wrap_ddp()
        elif self._encoder_config.distributed == 'FSDP':
            return self.wrap_fsdp()
        else:
            assert False
    
    def wrap_fsdp(self):
        
        llama_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                self.llm_layer,
            },
        )
        self.llm = FSDP(self.llm, device_id=torch.cuda.current_device(), auto_wrap_policy=llama_auto_wrap_policy)
        self.encoder = self.encoder.to(torch.cuda.current_device())
        self.encoder_to_llm_projection = self.encoder_to_llm_projection.to(torch.cuda.current_device())
        self.xyz_projection = self.xyz_projection.to(torch.cuda.current_device())
        self.pos_emb3d = self.pos_emb3d.to(torch.cuda.current_device())
        self.box_prompt_projector = self.box_prompt_projector.to(torch.cuda.current_device())
        self.click_prompt_projector = self.click_prompt_projector.to(torch.cuda.current_device())
        print_log('Using FSDP')
        return self
    
    def wrap_ddp(self):
        self = self.to(torch.cuda.current_device())
        self = nn.parallel.DistributedDataParallel(self, 
                                                    device_ids=[self.args.local_rank % torch.cuda.device_count()]
                                                    )
        print_log('Using Distributed Data parallel' )
        return self
    
    def _find_all_linear_names(self, model):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
    
    def load_model_from_ckpt(self, bert_ckpt_path, args, finetune=False):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        ckpt = torch.load(bert_ckpt_path, map_location='cpu')
        
        # tmp_ckpt = {k:v for k,v in ckpt['base_model'].items() if not k.find('llm.') != -1}
        # ckpt = {'base_model': tmp_ckpt}
        
        if not finetune:
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                    base_ckpt[k.replace("transformer_q.", "encoder.")] = base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k.replace("base_model.", "encoder.")] = base_ckpt[k]
                del base_ckpt[k]
        else:
            base_ckpt = ckpt['base_model']
            
        # uclip2
        # base_ckpt = {k.replace("module.point_encoder.", "encoder."): v for k, v in ckpt['state_dict'].items() if k.find('point_encoder') != -1}
       
        incompatible = self.load_state_dict(base_ckpt, strict=False)
        print_log(incompatible, logger = 'Transformer')
        

        if incompatible.missing_keys and torch.cuda.current_device() == 0:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys and torch.cuda.current_device() == 0:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )
        if torch.cuda.current_device() == 0:
            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')
    
    def _loss_caption(self, logits, target, mask):
        loss_per_word = nnf.cross_entropy(
            logits.permute(0, 2, 1).contiguous(),
            target,
            reduction='none',
        )
        final_loss = torch.sum(loss_per_word * mask) / torch.sum(mask + 1e-6)

        return final_loss
    
    def expand_prompt_representation(self, prompt_feature, prompt_mask=None):
        # input:
        #   prompt_feature: batch x nprompt x (ntkn x channel)
        #   prompt_mask: batch x nprompt
        # output:
        #   prompt_feature: batch x (nprompt x ntkn) x channel
        #   prompt_mask: batch x (nprompt x ntkn)
        batch_size, nprompt = prompt_feature.shape[:2]
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_feature[..., 0])
        prompt_mask = prompt_mask.unsqueeze(-1).repeat(1, 1, self.visual_nquery)
        prompt_mask = prompt_mask.reshape(batch_size, nprompt * self.visual_nquery)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt, self.visual_nquery, self._llm_config.hidden_size)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt * self.visual_nquery, self._llm_config.hidden_size)
        return prompt_feature, prompt_mask
    
    def forward(self, data_dict, eval=False):
    # def forward(self, **data_dict):
    #     eval=False
        with torch.no_grad():
            points = data_dict['points']
            level = data_dict['level']
            features = data_dict['features']
            vision_embed, vision_mask, center, neighborhood_xyz = self.encoder(points, features, level)
                
        # Visual prompt code
        batch_size = vision_embed.shape[0]
        # # Generate box and click prompt
        box_mask = data_dict['box_mask']
        click_prompt = data_dict['click_query']
        click_mask = data_dict['click_mask']
        box_prompt = data_dict['box_query'].unsqueeze(1)
        visual_prompt = [torch.zeros(batch_size, 0, self._llm_config.hidden_size).to(vision_embed.device)]
        visual_mask = [torch.zeros(batch_size, 0).to(vision_embed.device)]
        box_prompt = self.box_prompt_projector(box_prompt)
        box_prompt, box_mask = self.expand_prompt_representation(box_prompt, box_mask)
        visual_prompt.append(box_prompt)
        visual_mask.append(box_mask) 
        
        point_cloud_dims_min, _ = data_dict['points'][..., :3].min(dim=1)
        point_cloud_dims_max, _ = data_dict['points'][..., :3].max(dim=1)
        point_cloud_dims = [
            point_cloud_dims_min,
            point_cloud_dims_max,
        ]
        click_xyz = click_prompt     # batch x nquery x 3
        click_prompt = self.pos_emb3d(click_xyz, input_range=point_cloud_dims)
        click_prompt = self.click_prompt_projector(click_prompt.permute(0, 2, 1))
        click_prompt, click_mask = self.expand_prompt_representation(click_prompt, click_mask)
        visual_prompt.append(click_prompt)
        visual_mask.append(click_mask)

        ## concat box and click prompts as well as prompt masks
        prompt_feature = torch.cat(visual_prompt, dim=1)   # batch x (2 x ntoken) x channel
        prompt_mask = torch.cat(visual_mask, dim=1)        # batch x (2 x ntoken)
        if self.FLEX:
            hd_corpus = self._get_hd_corpus(data_dict, center, neighborhood_xyz)
            hd_center = hd_corpus['dense_center']
            hd_point_cloud_dims = hd_corpus['point_cloud_dims']
            hd_vision_embed = hd_corpus['dense_features']
            bsz, _, tn, dim = hd_vision_embed.shape
            TN = self._encoder_config.DENSE_TOKEN_NUM
            assert tn == TN
            hd_vision_embed = hd_vision_embed.view(bsz, -1, dim).contiguous()
            
            # hd_pos_embed = self.pos_emb3d(hd_center, input_range=hd_point_cloud_dims)
            # pos_embed = self.pos_emb3d(center, input_range=point_cloud_dims)
            # all_pos_embed = torch.cat((pos_embed, hd_pos_embed), dim=-1)
            # all_pos_embed = self.xyz_projection(all_pos_embed.permute(0, 2, 1))
            # pos, hd_pos = all_pos_embed.split([vision_embed.shape[1], hd_vision_embed.shape[1]], dim=1)
            # all_vision_embed = torch.cat((vision_embed+pos, hd_vision_embed+hd_pos), dim=1)

            all_vision_embed = torch.cat((vision_embed, hd_vision_embed), dim=1)
            all_vision_embed = self.encoder_to_llm_projection(all_vision_embed)
            vision_embed, hd_vision_embed = all_vision_embed.split([vision_embed.shape[1], hd_vision_embed.shape[1]], dim=1)


            for i in range(len(self.llm.model.layers)):
                self.llm.model.layers[i].self_attn.hd_features = hd_vision_embed.view(bsz, -1, TN, self._llm_config.hidden_size).contiguous()
                self.llm.model.layers[i].self_attn.step_ratio = data_dict.get('step_ratio', None)
        else:
            pos_embed = self.pos_emb3d(center, input_range=point_cloud_dims)
            pos_embed = self.xyz_projection(pos_embed.permute(0, 2, 1))
            vision_embed = self.encoder_to_llm_projection(vision_embed + pos_embed) 
    
        if not eval:
            input_mask = data_dict['attention_mask']
            input_ids = data_dict['input_ids']

            outputs = self.llm(
                vision_embeds=vision_embed.to(self.dtype),
                vision_mask=vision_mask.to(self.dtype),
                visual_prompt_embeds=prompt_feature.to(self.dtype),
                visual_prompt_mask=prompt_mask.to(self.dtype),
                input_ids=input_ids,
                attention_mask=input_mask.to(self.dtype),
                tokenizer = self.tokenizer,
                output_attentions=False,
            )
            
            gradient_mask = data_dict['gradient_mask']
            
            # if os.getenv('ADD_SCENE_LOSS','False') == 'True':
            # Add <scene> and </scene> logits to supervision
            scene_token_start_index = self.llm.config.scene_token_start_index
            start_scene_logits = outputs.logits[:, scene_token_start_index-1-1]
            end_scene_logits = outputs.logits[:, scene_token_start_index-1-1+self.VISION_TOKEN_NUM+1]
            start_scene_id = torch.tensor([self.tokenizer.convert_tokens_to_ids('<scene>')]).to(input_ids.device).unsqueeze(0).repeat(start_scene_logits.shape[0], 1)
            end_scene_id = torch.tensor([self.tokenizer.convert_tokens_to_ids('</scene>')]).to(input_ids.device).unsqueeze(0).repeat(start_scene_logits.shape[0], 1)
            
            new_logits = torch.cat((outputs.logits[:, -(gradient_mask.shape[1]+1): -1], start_scene_logits.unsqueeze(1), end_scene_logits.unsqueeze(1)), dim=1)
            new_input_ids = torch.cat((input_ids, start_scene_id, end_scene_id), dim=1)
            new_gradient_mask = torch.cat((gradient_mask, torch.ones_like(start_scene_id), torch.ones_like(end_scene_id)), dim=1)
            loss = self._loss_caption(
                logits = new_logits,
                target = new_input_ids,
                mask = new_gradient_mask.to(self.dtype),
            )
            # else:
            # loss = self._loss_caption(
            #     logits = outputs.logits[:, -(gradient_mask.shape[1]+1): -1],
            #     target = input_ids,
            #     mask = gradient_mask.to(self.dtype),
            # )
            
            return loss
        
        else:
            caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': None   # 4, None
            }
            
            # attentions = [None] * vision_embed.shape[0]
            instruction = data_dict['instruction']
            instruction_mask = data_dict['instruction_mask']
            
            output_ids = []
            attentions = []
            
            for batch_id in range(vision_embed.shape[0]):
                sample_instruction = instruction[batch_id]     
                sample_mask = instruction_mask[batch_id]     # ntoken
                
                output = generation(
                    self.llm, 
                    vision_embeds=vision_embed[batch_id].unsqueeze(0).to(self.dtype),
                    vision_mask=vision_mask[batch_id].unsqueeze(0).to(self.dtype),
                    input_ids=sample_instruction[sample_mask == 1].unsqueeze(0),
                    attention_mask=torch.ones_like(sample_instruction[sample_mask == 1].unsqueeze(0)).to(self.dtype),
                    visual_prompt_embeds=prompt_feature[batch_id].unsqueeze(0).to(self.dtype),
                    visual_prompt_mask=prompt_mask[batch_id].unsqueeze(0).to(self.dtype),
                    tokenizer = self.tokenizer,
                    max_length=20,   
                    **caption_config,
                )
                output_ids.append(output['output_ids'])
                if caption_config['num_beams'] is None:
                    if not output['attentions'] is None:
                        attn = torch.cat(output['attentions'], dim=0)
                        attentions.append(attn)
            
            output_ids = torch.cat(output_ids, dim=0)
            if len(attentions) > 0:
                attentions = torch.stack(attentions, dim=0)
            
            return output_ids, attentions, center