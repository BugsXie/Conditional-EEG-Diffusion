import os

import torch as th
import torch.nn as nn
import pandas as pd
from braindecode.models import EEGNetv4
from skorch import NeuralNet
from sampling import get_sampling_fn
from huggingface_hub import hf_hub_download

from data_utils.data_utils import Sampling
from data_utils.dataset import Lee2019Dataset, EEGDataset
from metrics import Metrics
from data_utils.plot_util import Plotting
from framework import SDEFramework


def run_inference(
    dataset: EEGDataset,
    amplitude_latency_channel: str,
    model_ckpt: str,
    EMA: bool,
    sampling_fn_cfg: dict,
    combination: dict,
):
    """Runs the sampling with a trained diffusion model

    Args:
        dataset (EEGDataset): Initialized dataset.
        amplitude_latency_channel (str): Which channel is used to compute the PAD and PLD metrics on. 
        model_ckpt (str): Relative path to the checkpoint file.
        EMA (bool): Whether to use the EMA weights instead of regular weights.
        sampling_fn_cfg (dict): Configuration for certain parameters regarding the samplers. 
        combination (dict): The combination which is sampled. A dict for the preloaded model should have the keys "label", "session", and "subject".
    """

    framework = SDEFramework.load_from_checkpoint(model_ckpt)

    sampling_fn = get_sampling_fn(
        framework.sde,
        sampling_fn_cfg["sampler_name"],
        sampling_fn_cfg["predictor_name"],
        sampling_fn_cfg["corrector_name"],
        sampling_fn_cfg["noise_removal"],
        sampling_fn_cfg["snr"],
        sampling_fn_cfg["n_steps_each"],
        sampling_fn_cfg["probability_flow"],
        framework.hparams.continuous,
        framework.hparams.device,
        sampling_fn_cfg["eps"],
    )

    # if conditional, configure the labels and create possible combinations
    if isinstance(DATASET_CFG["user_conditions"], list):
        MODEL_CFG["model__conditionals_combinations"] = dataset.condition_combinations

    # load pre-trained EEGNet
    path_params = "EEGNetv4_Lee2019_ERP/params.pt"
    path_optimizer = "EEGNetv4_Lee2019_ERP/optimizer.pt"
    path_history = "EEGNetv4_Lee2019_ERP/history.json"
    path_criterion = "EEGNetv4_Lee2019_ERP/criterion.pt"

    model = EEGNetv4(
        n_chans=19,
        n_outputs=2,
        n_times=128,
    )

    net = NeuralNet(
        model,
        criterion=nn.CrossEntropyLoss(weight=th.tensor([1, 1])),
    )
    net.initialize()
    net.load_params(
        path_params,
        path_optimizer,
        path_criterion,
        path_history,
    )

    metrics = Metrics(
        dataset.X,
        net.module,
        amplitude_latency_channel,
        DATASET_CFG["fmin"],
        DATASET_CFG["fmax"],
    )

    if isinstance(combination, pd.DataFrame) and not combination.empty:
        subject = combination["subject"].iloc[0]
        session = combination["session"].iloc[0]
        # 构造前缀，例如 "subject6_session_1"
        output_prefix = f"subject{subject}_{session}"
    else:
        output_prefix = None

    plotting = Plotting(
        SAMPLING_CFG["plot_channels"],
        output_prefix=output_prefix,
    )

    sampling = Sampling(
        dataset.X,
        sampling_fn,
        dataset.mne_info,
        metrics,
        plotting,
        combination,
        SAMPLING_CFG["split_condition"],
        dataset.reverse_mapping,
        dataset.y_df,
    )

    # set model to eval mode
    framework.eval()
    with th.no_grad():
        sampling_model = framework.ema_model if EMA else framework.model

        sampling.sampling_logging(
            pl_module=framework,
            sampling_shape=framework.hparams.sampling_shape,
            sampling_model=sampling_model,
        )


EMA = True
from configs.Lee2019_ERP_CFG import (
    DATASET_CFG,
    MODEL_CFG,
    RUN_CFG,
    SAMPLING_CFG,
)

amplitude_latency_channel = "O1"

dataset = Lee2019Dataset(**DATASET_CFG)

sampling_fn_cfg = {
    "sampler_name": "pc",
    "predictor_name": "euler_maruyama",
    "corrector_name": "langevin",
    "noise_removal": True,
    "probability_flow": False,
    "snr": 0.16,
    "n_steps_each": 1,
    "eps": 1e-5,
}

model_ckpt = "EEGWave_step600000/model.ckpt"

combination = pd.DataFrame(
    {
        "label": ['Target', "NonTarget"],
        "subject": [3, 3],
        "session": ['session_2', 'session_2'],
    }
)

# if one wants to sample the full dataset:
# combination = dataset.conditionals_combinations

run_inference(
    dataset=dataset,
    amplitude_latency_channel=amplitude_latency_channel,
    model_ckpt=model_ckpt,
    EMA=EMA,
    sampling_fn_cfg=sampling_fn_cfg,
    combination=combination,
)
