"""Decode Liquid-Codec codes to a 24 kHz wav with the Core ML decoder on the Apple Neural Engine (fp16 LFM stack) + a CPU fp32 tail.

  python examples/decode_wav.py --wav in.wav --out out.wav [--compare] [--units ane|cpu]
      encodes in.wav with the original PyTorch Liquid-Codec (CPU), decodes the codes on the ANE; --compare also decodes them with PyTorch
      and prints the SNR of the ANE audio against it.
  python examples/decode_wav.py --codes codes.npy --out out.wav      (codes: int array [12, frames], values 0-2015, 12.5 frames/s)

Needs: macOS 15+, Apple silicon, python 3.10+, numpy, torch, soundfile, coremltools (>= 8); for --wav/--compare also transformers and
huggingface_hub (the Liquid-Codec repo, LFM Open License v1.0, is downloaded from Hugging Face)."""
import argparse, os, sys, time
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "convert"))
from lc_ane import latent_from_codes, chunks, CTX
import coremltools as ct
REV = "b5abfac46ef92275f191a52c41013315f237301d"
ap = argparse.ArgumentParser(); ap.add_argument("--wav"); ap.add_argument("--codes"); ap.add_argument("--out", required=True)
ap.add_argument("--compare", action="store_true"); ap.add_argument("--units", default="ane", choices=["ane", "cpu"])
ap.add_argument("--models", default=os.path.join(HERE, "..", "models"))
ap.add_argument("--model", default="lc_dec_ane_int8.mlmodelc", help="lc_dec_ane_int8.mlmodelc (default, 36 MB) or lc_dec_ane.mlmodelc (fp16, 72 MB)"); a = ap.parse_args()
def liquid():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    d = os.environ.get("LIQUID_DIR") or snapshot_download("LiquidAI/Liquid-Codec", revision=REV)   # LIQUID_DIR: a local copy of the HF repo
    return AutoModel.from_pretrained(d, trust_remote_code=True, dtype=torch.float32).eval()
ref = None
if a.wav:
    x, sr = sf.read(a.wav, dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
    if sr != 24000:
        import torchaudio; x = torchaudio.functional.resample(torch.from_numpy(x), sr, 24000).numpy()
    ref = liquid()
    with torch.no_grad(): codes = ref.encode(torch.from_numpy(x)[None]).audio_codes[0].numpy()
else:
    codes = np.load(a.codes)
assert codes.shape[0] == 12, codes.shape
# CPU fp32 tail: linear 512 -> 1282 (log-magnitude, angle), exp/polar, inverse STFT (n_fft 1280, hop 320, Hann), window-envelope divide
tw = np.fromfile(os.path.join(a.models, "lc_tail_f32.bin"), "<f4"); W, b = tw[:1282 * 512].reshape(1282, 512), tw[1282 * 512:]
win = torch.hann_window(1280, dtype=torch.float32)
def tail(hid):
    y = torch.from_numpy(W @ hid + b[:, None])[None]; lm, ang = torch.chunk(y, 2, 1); spec = torch.polar(lm.exp(), ang); n = spec.shape[-1]
    fr = torch.fft.irfft(spec, 1280, dim=1, norm="backward") * win[None, :, None]; out = (n - 1) * 320 + 1280; fold = dict(output_size=(1, out), kernel_size=(1, 1280), stride=(1, 320))
    s = F.fold(fr, **fold)[:, 0, 0]; env = F.fold(win.square().expand(1, n, -1).transpose(1, 2), **fold)[0, 0, 0]
    return (s[:, 480:-480] / env[480:-480])[0].numpy()
units = ct.ComputeUnit.CPU_AND_NE if a.units == "ane" else ct.ComputeUnit.CPU_ONLY
t0 = time.perf_counter(); m = ct.models.CompiledMLModel(os.path.join(a.models, a.model), compute_units=units); t_load = time.perf_counter() - t0
lat = latent_from_codes(codes); hs = []; t0 = time.perf_counter()
for w, v, keep in chunks(lat):   # 20 codec frames of left context + 20 new frames per call (1.6 s of audio each)
    h = m.predict({"lat": w.astype(np.float16), "valid": v.astype(np.float16)})["hidden"].astype(np.float32)[0, :, 0]   # (512, 240) at 75 Hz
    hs.append(h[:, CTX * 6: CTX * 6 + keep])
t_stack = time.perf_counter() - t0; t0 = time.perf_counter(); y = tail(np.concatenate(hs, 1)); t_tail = time.perf_counter() - t0
sf.write(a.out, np.clip(y, -1, 1), 24000, subtype="PCM_16"); dur = len(y) / 24000
print(f"wrote {a.out}: {dur:.2f} s audio; load {t_load:.2f} s; decode {t_stack + t_tail:.3f} s ({dur / (t_stack + t_tail):.0f}x real time; stack {t_stack:.3f} s, tail {t_tail:.3f} s)")
if a.compare:
    ref = ref or liquid()
    with torch.no_grad(): r = ref.decode(torch.from_numpy(codes)[None]).audio_values[0].numpy()
    n = min(len(r), len(y)); e = y[:n] - r[:n]; print(f"vs PyTorch fp32 decode: SNR {10 * np.log10((r[:n] ** 2).sum() / max((e ** 2).sum(), 1e-20)):.1f} dB, max abs diff {np.abs(e).max():.2e}")
