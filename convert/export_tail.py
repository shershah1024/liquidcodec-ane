"""Write models/lc_tail_f32.bin from Liquid-Codec's model.safetensors: decoder.lin.weight [1282, 512] row-major, then decoder.lin.bias [1282],
float32 little-endian (the CPU tail used by examples/; part of Liquid AI's weights, LFM Open License v1.0).
usage: python export_tail.py LIQUID_MODEL_DIR OUT.bin"""
import sys, numpy as np
from safetensors.numpy import load_file
sd = load_file(f"{sys.argv[1]}/model.safetensors"); W = sd["decoder.lin.weight"].astype(np.float32); b = sd["decoder.lin.bias"].astype(np.float32)
assert W.shape == (1282, 512) and b.shape == (1282,), (W.shape, b.shape)
np.concatenate([W.reshape(-1), b]).astype("<f4").tofile(sys.argv[2]); print("wrote", sys.argv[2], W.size + b.size, "floats")
