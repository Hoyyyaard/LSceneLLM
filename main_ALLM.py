from tools import runner_Adaptive_LLM as llm_main
from utils import parser, dist_utils, misc
from tools.runner_Adaptive_LLM import run_net 
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        if args.launcher == 'pytorch':
            dist_utils.init_dist(args.launcher)
        elif args.launcher == 'slurm':
            dist_utils.init_dist(args.launcher, local_rank=args.local_rank)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        config.dataset.train.others.tbs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size 
        config.dataset.val.others.bs = config.total_bs // world_size 
        config.dataset.val.others.tbs = config.total_bs
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size 
            config.dataset.test.others.tbs = config.total_bs
    else:
        config.dataset.train.others.bs = config.total_bs
        config.dataset.train.others.tbs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs 
        config.dataset.val.others.bs = config.total_bs 
        config.dataset.val.others.tbs = config.total_bs
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs 
            config.dataset.test.others.tbs = config.total_bs
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed and not args.launcher == 'slurm':
        assert args.local_rank == torch.distributed.get_rank() 

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold
        
    # run
    if args.test:
        run_net(args, config, None, None, test=True)
    else:
        run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
