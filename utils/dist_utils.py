import os

import torch
import torch.multiprocessing as mp
from torch import distributed as dist



def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    print(f'init distributed in rank {torch.distributed.get_rank()}')


def _init_dist_slurm(backend, **kwargs):
    """
    Initialize the distributed environment from SLURM
    """
    local_rank = kwargs['local_rank']
    torch.cuda.set_device(local_rank)
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    num_gpus = torch.cuda.device_count()
    # addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # # specify master port
    # if "MASTER_PORT" in os.environ:
    #     pass  # use MASTER_PORT in the environment variable
    # else:
    #     os.environ["MASTER_PORT"] = "29500"
    # if "MASTER_ADDR" not in os.environ:
    #     os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(proc_id)
    port = os.environ["MASTER_PORT"]
    addr = os.environ["MASTER_ADDR"]
    print(f"local rank: {str(local_rank)}, global rank: {(proc_id*num_gpus)+local_rank}, world size: {ntasks * num_gpus}, node: {ntasks}, port: {port}, addr: {addr}")
    torch.distributed.init_process_group(
        backend=backend, rank=(proc_id*num_gpus)+local_rank, world_size=ntasks * num_gpus
    )
    torch.distributed.barrier()

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def reduce_tensor(tensor, args):
    '''
        for acc kind, get the mean in each gpu
    '''
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def gather_tensor(tensor, args):
    output_tensors = [tensor.clone() for _ in range(args.world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat
