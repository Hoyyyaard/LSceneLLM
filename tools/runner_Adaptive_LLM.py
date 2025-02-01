import torch
import torch.nn as nn
import os
import json
import datetime
from tools import builder
from utils import misc, dist_utils
import time
from tqdm import tqdm
from utils.logger import *
from utils.AverageMeter import AverageMeter
from collections import defaultdict, OrderedDict
from utils.metrics import Metrics
from models.adaptive_llm import AdaptiveLLM
from transformers import AutoConfig
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from tools.dist import all_gather_dict, barrier, is_primary
import torch.cuda.amp as amp
import wandb

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

# def safe_all_reduce(loss):
#     # Check if the current loss is NaN and create a mask
#     mask = (~torch.isnan(loss)).float()  # 1 if loss is not NaN, 0 if loss is NaN
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

#     # Perform all_reduce on loss and mask
#     dist.all_reduce(loss, op=dist.ReduceOp.SUM)
#     dist.all_reduce(mask, op=dist.ReduceOp.SUM)

#     # Compute the average loss considering only valid (non-NaN) values
#     avg_loss = loss / (mask + 1e-6)  # Add a small value to avoid division by zero

#     return avg_loss

def run_net(args, config, train_writer=None, val_writer=None, test=False):
    
    logger = get_logger(args.log_name)
    
    finetune = config.get('finetune', False)
    
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)

    # build model
    llm_config  = AutoConfig.from_pretrained('ckpts/Llama-2-7b-hf/config.json')
    base_model = AdaptiveLLM(llm_config, config.model, finetune=finetune, logger=logger, args=args)
   
    if not finetune:
        for n, p in base_model.llm.named_parameters():
            p.requires_grad = False
        for n, p in base_model.encoder.named_parameters():
            p.requires_grad = False

    # else:
    #     for n, p in base_model.llm.named_parameters():
    #         p.requires_grad = True
    #     for n, p in base_model.encoder.named_parameters():
    #         p.requires_grad = False
        
    
    # if args.use_gpu:
    #     base_model.to(args.local_rank)
    # print(torch.cuda.memory_summary())
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # if finetune:
    #     base_model.wrap_fsdp()
    #     print_log('Using FSDP to fully finetune LLM')
    #     base_model.wrap_lora()
    
    # if not test and args.local_rank == 0 and int(os.environ["RANK"]) == 0:
    #     if os.path.exists(os.path.join(args.experiment_path, 'wandb_id.json')):
    #         with open(os.path.join(args.experiment_path, 'wandb_id.json'), "r") as f:
    #             id = json.load(f)['id']
    #             wandb.init(project='3DLLM', id=id, resume="must")
    #         print('Resume wandb experiment')
    #     else:
    #         run = wandb.init(project='3DLLM', name=args.exp_name.split('_')[0])
    #         id = run.id
    #         # Save id to resume
    #         with open(os.path.join(args.experiment_path, 'wandb_id.json'), "w") as f:
    #             json.dump({'id':id}, f) 
    #         print('Start new wandb experiment')

    # resume ckpts
    if args.resume:
        
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    else:
        
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts, args, finetune=(finetune | test))
        else:
            print_log('Training from scratch', logger = logger)
    # print(torch.cuda.memory_summary())

    # if not test and not finetune:
    #     optimizer, scheduler = builder.build_llm_opti_sche(base_model, config, train_dataloader, finetune)
    #     for n, p in base_model.llm.named_parameters():
    #         p.requires_grad = True
    
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
            
        base_model = base_model.wrap_model()

    else:
        # print_log('Using Data parallel ...' , logger = logger)
        # base_model = nn.DataParallel(base_model).cuda()
        base_model = base_model.cuda()

    # Train
    if not test :
        print_log("Trainable parameters", logger = logger)
        if torch.cuda.current_device() == 0:
            for n, p in base_model.named_parameters():
                if p.requires_grad:
                    print_log(n, logger = logger)
        
        # print(torch.cuda.memory_summary())
        
        # scaler = amp.GradScaler()
        
        # optimizer & scheduler
        # if finetune:
        optimizer, scheduler = builder.build_llm_opti_sche(base_model, config, train_dataloader, finetune)
        
        if args.resume:
            builder.resume_optimizer(optimizer, args, logger = logger)
        
        # 启用自动微分异常检测
        # torch.autograd.set_detect_anomaly(True)
        
        # training
        max_tolerant_nan = 5
        curr_nan_times = 0
        epoch_tqdm = tqdm(total = config.max_epoch, desc = 'Epoch', position = 0)
        base_model.zero_grad()
        total_iter = config.max_epoch * len(train_dataloader)
        for epoch in range(start_epoch, config.max_epoch): 
        
            epoch_tqdm.update(1)
            if args.distributed:
                train_sampler.set_epoch(epoch)
            base_model.train()

            epoch_start_time = time.time()
            batch_start_time = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(['Loss'])
            from utils.misc import SmoothedValue
            time_delta = SmoothedValue(window_size=10)

            num_iter = 0

            base_model.train()  # set model to training mode
            n_batches = len(train_dataloader)
            pbar = tqdm(total = n_batches)
            for idx, data_dict in enumerate(train_dataloader):
                pbar.update(1)          
                
                curr_time = time.time()
                num_iter += 1
                n_itr = epoch * n_batches + idx
                
                for k,v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        data_dict[k] = v.cuda()
                        if torch.isnan(data_dict[k]).any():
                            assert False

                data_time.update(time.time() - batch_start_time)
                data_dict['step_ratio'] = n_itr / total_iter
                
                # with amp.autocast():
                # from calflops import calculate_flops

                # flops, macs, params = calculate_flops(model=base_model,
                #                         kwargs=data_dict,
                #                         print_results=False)
                # print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
                # assert False 

                loss = base_model(data_dict)
                
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / dist.get_world_size() 
                if not math.isfinite(loss.item()):
                    if curr_nan_times < max_tolerant_nan:
                        print_log("Loss in not finite. Skip this training step.", logger = logger)
                        curr_nan_times += 1
                        continue
                    else:
                        print_log("Loss in not finite. Terminate training.", logger = logger)
                        exit(-1)
                
                loss.backward()
                curr_nan_times = 0

                if 'clip_grad_norm' in os.environ.keys():
                    torch.nn.utils.clip_grad_norm_(parameters=base_model.parameters(), max_norm=1)
                    
                # forward
                if num_iter == config.step_per_update:
                    num_iter = 0
                    # scaler.step(optimizer)
                    optimizer.step()
                    base_model.zero_grad()
                    
                acc_step = int(len(train_dataloader) * epoch + idx)
                if isinstance(scheduler, list):
                    for item in scheduler:
                        item.step(acc_step)
                else:
                    scheduler.step(acc_step)

                if args.distributed:
                    loss = dist_utils.reduce_tensor(loss, args)
                    losses.update([loss.item()])
                else:
                    losses.update([loss.item()])

                if args.distributed:
                    torch.cuda.synchronize()

                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
                    
                # if not test and args.local_rank == 0 and int(os.environ["RANK"]) == 0:
                #     wandb.log({"step": n_itr, "loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})

                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()
                
                time_delta.update(time.time() - curr_time)

                if idx % 20 == 0:
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                                (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                                ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
                    eta_seconds = (n_batches - idx) * time_delta.avg
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print_log(f'ETA: {eta_str}', logger = logger)
                
            epoch_end_time = time.time()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
            if epoch <= 3:
                builder.save_checkpoint_pretrain_llm(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger, finetune=finetune)  
            if epoch > 1 :
                builder.save_checkpoint_pretrain_llm(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger, finetune=finetune)     

            torch.distributed.barrier()
            exit()
            
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
            
    # Validate the model
    else:
        print('Only validating ...')
        validate(base_model, test_dataloader, -1, val_writer, args, config, logger = logger, finetune=finetune)


def score_captions(corpus: dict, candidates: dict):
    
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    
    score_per_caption = {
        "bleu-1": [float(s) for s in bleu[1][0]],
        "bleu-2": [float(s) for s in bleu[1][1]],
        "bleu-3": [float(s) for s in bleu[1][2]],
        "bleu-4": [float(s) for s in bleu[1][3]],
        "cider": [float(s) for s in cider[1]],
        "rouge": [float(s) for s in rouge[1]],
        "meteor": [float(s) for s in meteor[1]],
    }
    
    message = '\n'.join([
        "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
        ),
        "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
        ),
        "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
        ),
        "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
        ),
        "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            cider[0], max(cider[1]), min(cider[1])
        ),
        "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            rouge[0], max(rouge[1]), min(rouge[1])
        ),
        "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            meteor[0], max(meteor[1]), min(meteor[1])
        )
    ])
    
    eval_metric = {
        "BLEU-4": bleu[0][3],
        "CiDEr": cider[0],
        "Rouge": rouge[0],
        "METEOR": meteor[0],
    }
    return score_per_caption, message, eval_metric


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, finetune=False):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    
    if args.visualization_attn:
        pred_corpus = json.load(open("experiments/Qformer-Adaptive-LLM-finetune-Openscene-test-FLEX-QueryProb-threshold-HD-hm3dqa/qformer/test_Exp0107_0726_WHdAugBbox_DiffPrompt_FlexWarmUp-1_Threshold127_QueryProb_QformerAttnLayer0_From[LL3DA]_Epoch4/hd_scene_qa_-1_qa_pred_gt_val.json"))
    # if not finetune:
    candidates = {}
    test_pbar = tqdm(total = len(test_dataloader))
    with torch.no_grad():
        for idx, data_dict in enumerate(test_dataloader):
            
            test_pbar.update(1)
            
            for k,v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.cuda()
            
            with amp.autocast():
                output_ids, attentions, center = base_model(data_dict, eval=True)
            outputs = dict(
                output_ids=output_ids,
            )
            
            gather_data_dict = {'episode_id': data_dict['episode_id'], 'task_name': data_dict['task_name']}
            if args.distributed:
                outputs = all_gather_dict(outputs)
                gather_data_dict = all_gather_dict(gather_data_dict)
                
            # Flatten 2d list to 1d
            for k,v in gather_data_dict.items():
                if isinstance(v, list):
                    gather_data_dict[k] = [item for sublist in zip(*v) for item in sublist]
            
            output_ids = outputs["output_ids"]  # batch x max_length
            # if not finetune:
            if hasattr(base_model, 'tokenizer'):
                answers = base_model.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            else:
                answers = base_model.module.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            
            print(answers)
            
            for idx in range(output_ids.shape[0]):
                key = gather_data_dict['episode_id'][idx]
                task_name = gather_data_dict['task_name'][idx]
                answer = answers[idx]
                answer = ' '.join(filter(lambda w: w, answer.split(' ')))
                if not task_name in candidates:
                    candidates[task_name] = {key: [answer]}
                else:
                    candidates[task_name][key] = [answer]

            barrier()
         
        for k, v in candidates.items():
            corpus = test_dataloader.dataset.corpus[k]
            new_corpus = {}
            for cor in corpus:
                new_corpus[cor['episode_id']] = [cor['anno']['utterance']] if 'utterance' in cor['anno'].keys() else cor['anno']['answers']
            corpus = new_corpus
            score_per_caption, message, eval_metric = score_captions(
                OrderedDict([(key, corpus[key]) for key in v]), v
            )
        
            if is_primary():
                print_log(f"\n----------------------Evaluation {k}-----------------------\n", logger = logger)
                print_log(message, logger = logger)
                _log_to_disk(args, message, corpus, v, score_per_caption, k, epoch)
            
        if args.distributed:
            torch.cuda.synchronize()
    
    return Acc_Metric(0.)



def _log_to_disk(args, message, corpus, candidates, score_per_caption, task_name, epoch):
    with open(os.path.join(args.experiment_path, f"{task_name}_{epoch}_qa_scores.json"), "w") as f: 
                json.dump(message, f)
            
    with open(os.path.join(args.experiment_path, f"{task_name}_{epoch}_qa_corpus_val.json"), "w") as f: 
        json.dump(corpus, f, indent=4)
    
    with open(os.path.join(args.experiment_path, f"{task_name}_{epoch}_qa_pred_val.json"), "w") as f:
        json.dump(candidates, f, indent=4)
    
    with open(os.path.join(args.experiment_path, f"{task_name}_{epoch}_qa_pred_gt_val.json"), "w") as f:
        pred_gt_val = {}
        for scene_object_id, scene_object_id_key in enumerate(candidates):
            pred_gt_val[scene_object_id_key] = {
                'pred': candidates[scene_object_id_key],
                'gt': corpus[scene_object_id_key],
                'score': {
                    'bleu-1': score_per_caption['bleu-1'][scene_object_id],
                    'bleu-2': score_per_caption['bleu-2'][scene_object_id],
                    'bleu-3': score_per_caption['bleu-3'][scene_object_id],
                    'bleu-4': score_per_caption['bleu-4'][scene_object_id],
                    'CiDEr': score_per_caption['cider'][scene_object_id],
                    'rouge': score_per_caption['rouge'][scene_object_id],
                    'meteor': score_per_caption['meteor'][scene_object_id]
                }
            }
        json.dump(pred_gt_val, f, indent=4)

def test_net():
    pass
