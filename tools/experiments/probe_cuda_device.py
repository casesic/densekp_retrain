"""Small isolated device check; never loads DenseKP or resets a GPU."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--gpu-uuid',help='Use stable NVML UUID instead of CUDA ordinal')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid or str(a.physical_gpu)
    import torch
    start=time.monotonic();torch.set_num_threads(2)
    report=dict(physical_gpu=a.physical_gpu,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                logical_gpu=0,torch=torch.__version__)
    try:
        report['name']=torch.cuda.get_device_name(0)
        report['bf16_supported']=torch.cuda.is_bf16_supported()
        report['memory_free_total_before']=torch.cuda.mem_get_info(0)
        torch.cuda.reset_peak_memory_stats(0)
        x=torch.randn(512,512,device='cuda:0')
        checks=[]
        for dtype in [torch.float32,torch.bfloat16]:
            y=x.to(dtype)@x.to(dtype).T
            torch.cuda.synchronize()
            if not torch.isfinite(y).all():raise ValueError('Nonfinite test matrix')
            checks.append(dict(dtype=str(dtype),finite=True))
        report.update(status='passed',checks=checks,peak_bytes=torch.cuda.max_memory_allocated(0))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        report['seconds']=time.monotonic()-start
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
