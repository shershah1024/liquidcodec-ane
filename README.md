# Liquid-Codec decoder on the Apple Neural Engine

A Core ML conversion of the decoder of [Liquid AI's Liquid-Codec](https://huggingface.co/LiquidAI/Liquid-Codec) (24 kHz speech codec, 12 codebooks at
12.5 frames/s), built to run on the Apple Neural Engine (ANE). It turns Liquid-Codec codes into 24 kHz audio.

**Licence:** the model weights and the decoder design are Liquid AI's and are used under the **LFM Open License v1.0**
(`LICENSE-LFM-Open-v1.0.txt`). That includes its commercial-use limit (Section 5: commercial use is licensed only below USD 10M annual revenue).
What we changed is listed in `NOTICE` and at the top of `convert/lc_ane.py`.

## Speed: CPU vs Neural Engine
**5.9x faster on the Neural Engine (int8) than Liquid's original model on the CPU.**

MacBook Air, Apple M3. One set of 10 speech clips (60.2 s of audio, n = 10), the same codes for every row.

| decoder | size | x real time |
|---|---|---|
| Liquid's original PyTorch model, fp32, on the CPU (4 threads) | 147 MB (whole codec) | 53x |
| this conversion, fp16, on the Neural Engine | 72 MB | 227x |
| **this conversion, int8 weights (recommended), on the Neural Engine** | **36 MB** | **315x** |

All 464 operations of both packages run on the Neural Engine (Core ML compute plan). The first call after loading is slower: compiling for the
ANE takes ~3-5 s once per process.

## Does it match the original?
- Against Liquid's PyTorch fp32 decode of the same codes (same 10 clips): fp16 SNR 38.9 dB (sd 5.4, min 26.0); int8 27.6 dB (sd 3.2, min 20.3).
- int8 was judged by ear as unchanged from fp16 and from PyTorch fp32 ("absolutely fine, no change in the voice"; 6 clips, one listener).
- In PyTorch fp32, the re-implemented decoder matches Liquid's hidden states to 1.4e-5 (max abs). Chunked decoding matches full-sequence decoding
  to 3.2e-5. A negative control (chunks without their left context) differs by 3.2, so the check can fail.
- The Swift example matches the Python example on the same codes and package at 71.6 dB SNR.
- Lower bit widths did not hold up without retraining: plain post-training int4 (per-block) dropped to 9.8 dB and left the ANE (386 of 464 ops on
  the CPU); 4-bit and 6-bit k-means palettes gave 4.4 and 16.6 dB.

## What is in the box
- `models/lc_dec_ane_int8.mlmodelc` (recommended, 36 MB): the same package with weight-only int8 (linear, symmetric, per-channel; coremltools).
- `models/lc_dec_ane.mlmodelc` (fp16, 72 MB): the decoder's LFM stack (8 layers: 5 short convolutions, 3 sliding-window attentions) as an fp16 Core ML program.
  Fixed shape: each call takes 40 codec frames (20 of left context + 20 new = 1.6 s of new audio) and a validity mask, and returns the 512-d hidden
  states at 75 Hz. It is stateless, so chunked output equals full decoding: the receptive field is 117 frames, under the 120-frame window.
- `models/lc_tail_f32.bin`: the output projection (512 -> 1282, float32). The CPU tail turns hidden states into audio: linear -> exp/polar ->
  inverse STFT (n_fft 1280, hop 320, Hann) -> overlap-add with the window envelope. It stays on the CPU in fp32 for precision.
- `convert/lc_ane.py`: re-implementation and conversion (`check` compares against Liquid's PyTorch model; `convert` writes the .mlpackage).
  `convert/export_tail.py` writes the tail file.
- `examples/decode_wav.py`: Python, wav -> codes (Liquid's PyTorch encoder) -> ANE decode -> wav, with `--compare`.
- `examples/swift/`: a standalone Swift decoder (`LiquidANEDecoder`) and a CLI: codes (int32 [12, T]) -> 24 kHz wav.
- `samples/`: an original clip and its ANE decode, plus `codes_example.npy`. The voice is our own synthetic voice (Qwen3-TTS VoiceDesign, Apache-2.0).

## Changes needed for the ANE
- The quantizer's two output projections (48 -> 256 -> 512) are folded into one, and linears are written as 1x1 convolutions on (1, C, 1, T) tensors.
- Each RMSNorm gets a power-of-2 prescale (from activation statistics), so x² cannot overflow fp16. The residual stream reaches ~2,300.
- Fixed-shape stateless chunks with a validity mask replace the KV-cache streaming path: the mask zeroes conv inputs before the start and masks attention keys.
- The STFT head (complex exp, inverse FFT, overlap-add) stays outside the package, on the CPU in fp32.

## Use
```
# Python (macOS 15+, Apple silicon): pip install numpy torch soundfile coremltools transformers huggingface_hub
python examples/decode_wav.py --wav samples/voice_original.wav --out out.wav --compare
python examples/decode_wav.py --codes samples/codes_example.npy --out out.wav            # int8 by default; --model lc_dec_ane.mlmodelc for fp16

# Swift
swiftc -O -o decode examples/swift/LiquidANEDecoder.swift examples/swift/main.swift
python -c "import numpy as np; np.load('samples/codes_example.npy').astype('<i4').tofile('codes.bin')"
./decode models codes.bin 39 out.wav

# Re-convert from Liquid's weights (needs coremltools; check first, then convert)
python convert/lc_ane.py check  /path/to/Liquid-Codec
python convert/lc_ane.py convert /path/to/Liquid-Codec lc_dec_ane.mlpackage
xcrun coremlcompiler compile lc_dec_ane.mlpackage models/
python convert/export_tail.py /path/to/Liquid-Codec models/lc_tail_f32.bin
```

## Limits
- Decoder only. The encoder is not converted; use Liquid's PyTorch encoder to get codes.
- Not bit-exact: fp16 is ~39 dB and int8 ~28 dB SNR against fp32 (int8 judged unchanged by ear, n=6).
- Numbers come from one machine (M3 Air) and one set of 10 clips. Other chips will differ. The first call per process includes ANE compilation.
- macOS 15 or later (the package targets macOS 15).
