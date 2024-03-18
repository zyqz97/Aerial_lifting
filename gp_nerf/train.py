from argparse import Namespace

import torch
from torch.distributed.elastic.multiprocessing.errors import record

import sys
sys.path.append('.')

from gp_nerf.opts import get_opts_base



def _get_train_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)


    return parser.parse_args()

@record
def main(hparams: Namespace) -> None:

    if hparams.dataset_type == 'memory_depth_dji_instance_crossview_process':
        hparams.enable_semantic=True
        hparams.enable_instance=True


    if hparams.enable_semantic or hparams.enable_instance:
        hparams.freeze_geo=True
        hparams.use_subset=True
        hparams.depth_dji_loss=False
        hparams.wgt_depth_mse_loss=0
        
        hparams.lr=0.01
        assert hparams.ckpt_path is not None
        if hparams.enable_instance:
            hparams.freeze_semantic=True


    if 'nr3d' in hparams.network_type:
        assert hparams.contract_new == True
        assert hparams.use_scaling == False

    if 'linear_assignment' in hparams.instance_loss_mode:
        assert hparams.num_instance_classes > 30
    else:
        hparams.num_instance_classes = 25
        assert hparams.num_instance_classes < 30


    from gp_nerf.runner_gpnerf import Runner
    print(f"stop_semantic_grad:{hparams.stop_semantic_grad}")

    hparams.bg_nerf = False


    if hparams.detect_anomalies:
        with torch.autograd.detect_anomaly():
            Runner(hparams).train()
    else:
        Runner(hparams).train()


if __name__ == '__main__':
    main(_get_train_opts())
