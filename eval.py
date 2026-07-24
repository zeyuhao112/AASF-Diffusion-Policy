"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
from omegaconf import OmegaConf
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # DP seed override patch: allow independent evaluation seeds without editing checkpoint config.
    try:
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_readonly(cfg, False)
    except Exception:
        pass

    if os.environ.get("DP_TEST_START_SEED") is not None:
        cfg.task.env_runner.test_start_seed = int(os.environ["DP_TEST_START_SEED"])
    if os.environ.get("DP_N_TEST") is not None:
        cfg.task.env_runner.n_test = int(os.environ["DP_N_TEST"])
    if os.environ.get("DP_N_TEST_VIS") is not None:
        cfg.task.env_runner.n_test_vis = int(os.environ["DP_N_TEST_VIS"])
    if os.environ.get("DP_N_ENVS") is not None:
        cfg.task.env_runner.n_envs = int(os.environ["DP_N_ENVS"])
    if os.environ.get("DP_N_TRAIN") is not None:
        cfg.task.env_runner.n_train = int(os.environ["DP_N_TRAIN"])
    if os.environ.get("DP_N_TRAIN_VIS") is not None:
        cfg.task.env_runner.n_train_vis = int(os.environ["DP_N_TRAIN_VIS"])

    print(
        f"[eval override] test_start_seed={cfg.task.env_runner.test_start_seed}, "
        f"n_test={cfg.task.env_runner.n_test}, "
        f"n_train={cfg.task.env_runner.n_train}"
    )
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
