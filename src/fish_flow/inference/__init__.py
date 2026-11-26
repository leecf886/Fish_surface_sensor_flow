"""推理子包：提供编码器独立推理、解码器独立推理、串联推理及单配置执行入口。"""
from fish_flow.inference.core import (
    CascadePrediction,
    DecoderPrediction,
    EncoderPrediction,
    LoadedEncoderBundle,
    LoadedPPINNBundle,
    infer_cascade,
    infer_decoder,
    infer_encoder,
    load_encoder_checkpoint,
    load_ppinn_checkpoint,
)
from fish_flow.inference.data import (
    DecoderInferenceInputs,
    EncoderInferenceInputs,
    load_decoder_inference_inputs,
    load_encoder_inference_inputs,
)
from fish_flow.inference.runner import load_infer_config, run_inference_from_config

__all__ = [
    "CascadePrediction",
    "DecoderInferenceInputs",
    "DecoderPrediction",
    "EncoderInferenceInputs",
    "EncoderPrediction",
    "LoadedEncoderBundle",
    "LoadedPPINNBundle",
    "infer_cascade",
    "infer_decoder",
    "infer_encoder",
    "load_decoder_inference_inputs",
    "load_encoder_checkpoint",
    "load_encoder_inference_inputs",
    "load_infer_config",
    "load_ppinn_checkpoint",
    "run_inference_from_config",
]
