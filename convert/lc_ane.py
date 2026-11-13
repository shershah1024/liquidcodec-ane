"""Liquid-Codec decoder (36.4M) -> Core ML for the Apple Neural Engine.

MODIFIED WORK NOTICE: this file re-implements the decoder of Liquid AI's Liquid-Codec (modeling_liquid_codec.py / modeling_lfm3.py,
https://huggingface.co/LiquidAI/Liquid-Codec, revision b5abfac46ef92275f191a52c41013315f237301d) in an ANE-friendly layout and loads Liquid AI's
weights. Changes from the original: projections folded into one 48->512 linear; 1x1 convs instead of linears; fixed-shape, stateless chunking
with a validity mask; per-norm power-of-2 prescale so fp16 x^2 cannot overflow; the STFT head runs outside the package (CPU, fp32).
Liquid-Codec and its weights are licensed under the LFM Open License v1.0 (see LICENSE-LFM-Open-v1.0.txt).

ANE part (fp16): 48-d latent at 12.5 Hz -> folded project_out/from_quantizer (48->512) -> 6x nearest repeat
-> 8 LFM3 layers (conv k=7 x5, sliding attention w=30 x3) -> final RMSNorm -> hidden (1,512,1,T).
CPU part (fp32, NOT in the package): decoder.lin -> exp/polar -> iSTFT -> envelope divide.

Fixed shape, stateless chunking: each call sees CTX=20 codec frames (120 x 75 Hz) of left context plus NEW=20
new frames. The decoder is causal and its receptive field is 5*6 + 3*29 = 117 frames < 120, so chunked output
equals the full-sequence output up to rounding. A `valid` input (1 at real frames) reproduces the zero padding
at the sequence start: it zeroes the conv input and masks attention keys.

usage:
  python lc_ane.py check  MODEL_DIR            # fp32 torch: chunked vs full, vs HF if transformers is present; act stats
  python lc_ane.py convert MODEL_DIR OUT.mlpackage
"""
import json, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

CTX, NEW = 20, 20                      # codec frames (12.5 Hz)
FR = CTX + NEW                         # 40 codec frames per call
T = FR * 6                             # 240 frames at 75 Hz
WIN, KC, H, KVH, HD, D = 30, 7, 8, 2, 64, 512
ATT = (2, 4, 6)
EPS = 1e-5


def load_sd(model_dir):
    from safetensors.torch import load_file
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    return {k: v.float() for k, v in sd.items()}


class RMS(nn.Module):
    """x * rsqrt(mean(x^2)+eps) * w over dim `dim`, with a prescale s so fp16 x^2 cannot overflow."""
    def __init__(self, w, dim, s=1.0):
        super().__init__(); self.w = nn.Parameter(w, requires_grad=False); self.dim = dim; self.s = float(s)

    def forward(self, x):
        u = x * self.s
        r = torch.rsqrt((u * u).mean(self.dim, keepdim=True) + EPS * self.s * self.s)
        shp = [1] * x.dim(); shp[self.dim] = -1
        return u * r * self.w.view(shp)


def c1(w, b=None):
    m = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=b is not None)
    m.weight.data = w[:, :, None, None].clone()
    if b is not None: m.bias.data = b.clone()
    m.requires_grad_(False); return m


class Layer(nn.Module):
    def __init__(self, sd, i, scales):
        super().__init__()
        p = f"decoder.lfm.layers.{i}."
        self.att = i in ATT
        g = lambda n: sd[p + n]
        self.n_op = RMS(g("operator_norm.weight"), 1, scales.get(f"{i}.op", 1.0))
        self.n_pop = RMS(g("post_operator_norm.weight"), 1, scales.get(f"{i}.pop", 1.0))
        self.n_ffn = RMS(g("ffn_norm.weight"), 1, scales.get(f"{i}.ffn", 1.0))
        self.n_pffn = RMS(g("post_ffn_norm.weight"), 1, scales.get(f"{i}.pffn", 1.0))
        self.w1, self.w3, self.w2 = c1(g("feed_forward.w1.weight")), c1(g("feed_forward.w3.weight")), c1(g("feed_forward.w2.weight"))
        if self.att:
            a = "self_attn."
            self.q, self.k, self.v = c1(g(a + "q_proj.weight")), c1(g(a + "k_proj.weight")), c1(g(a + "v_proj.weight"))
            self.o, self.gp = c1(g(a + "o_proj.weight")), c1(g(a + "gate_proj.weight"))
            self.qn = RMS(g(a + "q_norm.weight"), 2); self.kn = RMS(g(a + "k_norm.weight"), 2)
        else:
            self.inp, self.out = c1(g("conv.in_proj.weight")), c1(g("conv.out_proj.weight"))
            self.dw = nn.Conv2d(D, D, (1, KC), groups=D, bias=False)
            self.dw.weight.data = g("conv.conv.weight")[:, :, None, :].clone(); self.dw.requires_grad_(False)

    def forward(self, x, valid, mask, cos, sin, stats=None, tag=""):
        def st(name, t):
            if stats is not None:
                k = f"{tag}.{name}"; stats[k] = max(stats.get(k, 0.0), float(t.abs().max()))
        st("in", x)
        h = self.n_op(x)
        if self.att:
            n = x.shape[-1]
            q = self.qn(self.q(h).reshape(1, H, HD, n)); k = self.kn(self.k(h).reshape(1, KVH, HD, n))
            v = self.v(h).reshape(1, KVH, HD, n)
            rot = lambda t: torch.cat((-t[:, :, HD // 2:], t[:, :, :HD // 2]), 2)
            q = q * cos + rot(q) * sin; k = k * cos + rot(k) * sin
            k = torch.cat([k[:, 0:1]] * 4 + [k[:, 1:2]] * 4, 1); v = torch.cat([v[:, 0:1]] * 4 + [v[:, 1:2]] * 4, 1)
            s = torch.matmul(q.transpose(2, 3), k) * (HD ** -0.5) + mask           # (1,H,n,n)
            st("score", s.clamp(min=-1e3))
            a = torch.softmax(s, -1)
            o = torch.matmul(a, v.transpose(2, 3)).transpose(2, 3)                  # (1,H,HD,n)
            o = o * torch.sigmoid(self.gp(h)).reshape(1, H, 1, n)
            h = self.o(o.reshape(1, D, 1, n))
        else:
            bcx = self.inp(h); b, c, xx = bcx[:, :D], bcx[:, D:2 * D], bcx[:, 2 * D:]
            u = b * xx * valid
            st("bx", u)
            u = self.dw(F.pad(u, (KC - 1, 0)))
            h = self.out(c * u)
        st("op", h)
        x = x + self.n_pop(h)
        st("mid", x)
        f = self.n_ffn(x)
        f1 = self.w1(f); f3 = self.w3(f); st("w1", f1); st("w3", f3)
        m = F.silu(f1) * f3; st("prod", m)
        f = self.w2(m); st("ffo", f)
        return x + self.n_pffn(f)


class DecANE(nn.Module):
    """lat (1,48,1,FR) fp, valid (1,1,1,T) 0/1  ->  hidden (1,512,1,T)."""
    def __init__(self, sd, n=T, scales=None):
        super().__init__()
        scales = scales or {}
        Wp, bp = sd["quantizer.fsq.project_out.weight"], sd["quantizer.fsq.project_out.bias"]
        Wf, bf = sd["from_quantizer.weight"], sd["from_quantizer.bias"]
        self.inp = c1(Wf @ Wp, Wf @ bp + bf)
        self.layers = nn.ModuleList([Layer(sd, i, scales) for i in range(8)])
        self.fn = RMS(sd["decoder.lfm.embedding_norm.weight"], 1, scales.get("final", 1.0))
        self.set_len(n)

    def set_len(self, n):
        inv = 1.0 / (1e6 ** (torch.arange(0, HD, 2).float() / HD))
        fr = torch.arange(n).float()[:, None] * inv[None]
        emb = torch.cat((fr, fr), -1).T                                             # (HD, n)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)
        qi, ki = torch.arange(n)[:, None], torch.arange(n)[None]
        ok = (ki <= qi) & (qi - ki < WIN)
        self.register_buffer("base", torch.where(ok, 0.0, -1e4)[None, None], persistent=False)

    def forward(self, lat, valid, stats=None):
        x = self.inp(lat)
        x = F.interpolate(x, scale_factor=(1, 6), mode="nearest")
        mask = self.base + (valid - 1.0) * 1e4                                       # (1,1,n,n), keys masked
        for i, l in enumerate(self.layers):
            x = l(x, valid, mask, self.cos, self.sin, stats, str(i))
        if stats is not None: stats["final.in"] = max(stats.get("final.in", 0.0), float(x.abs().max()))
        return self.fn(x)


# ---------------- CPU fp32 tail + chunk driver (shared with the Air runtime) ----------------
class Tail:
    def __init__(self, sd):
        self.W, self.b = sd["decoder.lin.weight"].double().numpy().astype(np.float32), sd["decoder.lin.bias"].numpy()
        self.win = torch.hann_window(1280, dtype=torch.float32)

    def audio(self, hid):                       # hid (512, n75) float32 -> waveform float32, same as HF decode()
        y = torch.from_numpy(self.W @ hid + self.b[:, None])[None]                  # (1,1282,n)
        lm, ang = torch.chunk(y, 2, 1)
        spec = torch.polar(lm.exp(), ang)
        n = spec.shape[-1]
        fr = torch.fft.irfft(spec, 1280, dim=1, norm="backward") * self.win[None, :, None]
        out = (n - 1) * 320 + 1280
        fold = dict(output_size=(1, out), kernel_size=(1, 1280), stride=(1, 320))
        a = F.fold(fr, **fold)[:, 0, 0]
        env = F.fold(self.win.square().expand(1, n, -1).transpose(1, 2), **fold)[0, 0, 0]
        trim = 480
        return (a[:, trim:-trim] / env[trim:-trim])[0].numpy()


def latent_from_codes(codes):                   # codes (12, F) int -> (48, F) float32 grid values
    lv = np.array([8, 7, 6, 6]); basis = np.array([1, 8, 56, 336]); hw = lv // 2
    c = codes.T[:, :, None]
    lat = ((c // basis) % lv - hw) / hw
    return lat.reshape(codes.shape[1], 48).T.astype(np.float32)


def chunks(lat):                                # lat (48, Ftot) -> list of (lat_win (1,48,1,FR), valid (1,1,1,T), keep_frames)
    Ft = lat.shape[1]; out = []
    pad = np.zeros((48, CTX), np.float32)
    L = np.concatenate([pad, lat, np.zeros((48, NEW), np.float32)], 1)
    for s in range(0, Ft, NEW):
        w = L[:, s:s + FR]
        idx = np.arange(s - CTX, s - CTX + FR)
        v = ((idx >= 0) & (idx < Ft)).astype(np.float32).repeat(6)
        keep = min(NEW, Ft - s) * 6
        out.append((w[None, :, None], v[None, None, None], keep))
    return out


def run_chunked(fn, lat):                       # fn(lat_win, valid) -> hidden (512, T) ; returns (512, Ftot*6)
    hs = []
    for w, v, keep in chunks(lat):
        h = fn(w, v)
        hs.append(h[:, CTX * 6: CTX * 6 + keep])
    return np.concatenate(hs, 1)


def main():
    mode, mdir = sys.argv[1], sys.argv[2]
    torch.set_num_threads(int(os.environ.get("NT", "4")))
    sd = load_sd(mdir)
    torch.manual_seed(0)
    if mode == "check":
        g = np.random.default_rng(0)
        Ft = 63
        codes = g.integers(0, 2016, (12, Ft))
        lat = latent_from_codes(codes)
        full = DecANE(sd, n=Ft * 6).eval()
        with torch.no_grad():
            hf_full = full(torch.from_numpy(lat)[None, :, None], torch.ones(1, 1, 1, Ft * 6))[0, :, 0].numpy()
            m = DecANE(sd).eval(); stats = {}
            fn = lambda w, v: m(torch.from_numpy(w), torch.from_numpy(v), stats)[0, :, 0].numpy()
            ch = run_chunked(fn, lat)
            # negative control: too little context must differ
            res = dict(chunk_vs_full_maxabs=float(np.abs(ch - hf_full).max()), hid_absmax=float(np.abs(hf_full).max()))
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(mdir)))
                from transformers import AutoModel
                ref = AutoModel.from_pretrained(mdir, trust_remote_code=True, dtype=torch.float32).eval()
                ct = torch.from_numpy(codes)[None]
                emb = ref.from_quantizer(ref.quantizer.decode(ct).mT)
                hid = ref.decoder.lfm(inputs_embeds=F.interpolate(emb.mT, emb.shape[1] * 6, mode="nearest-exact").mT,
                                      use_cache=False).last_hidden_state[0].T.numpy()
                res["mine_full_vs_hf_hidden_maxabs"] = float(np.abs(hf_full - hid).max())
                tail = Tail(sd)
                a_ref = ref.decode(ct).audio_values[0].numpy(); a_me = tail.audio(ch)
                res["audio_ref_vs_mine_chunked_maxabs"] = float(np.abs(a_ref - a_me).max())
                res["audio_ref_absmax"] = float(np.abs(a_ref).max())
            except Exception as e:
                res["hf"] = f"skipped: {type(e).__name__}: {e}"
            # negative control: drop the context (CTX frames all invalid) -> must differ from full
            def fn0(w, v):
                v = v.copy(); v[..., :CTX * 6] = 0
                return m(torch.from_numpy(w), torch.from_numpy(v))[0, :, 0].numpy()
            res["negctl_noctx_vs_full_maxabs"] = float(np.abs(run_chunked(fn0, lat) - hf_full).max())
            res["stats"] = stats
        print(json.dumps(res, indent=1)); json.dump(res, open("check.json", "w"), indent=1)
    elif mode == "convert":
        import coremltools as ct
        st = json.load(open("check.json"))["stats"]
        # prescale each norm so the scaled max |x| is <= 32 (x^2 mean stays far below fp16 max)
        scales = {}
        for i in range(8):
            for nm, key in (("op", "in"), ("ffn", "mid"), ("pop", "op"), ("pffn", "ffo")):
                mx = st.get(f"{i}.{key}", 1.0)
                scales[f"{i}.{nm}"] = 2.0 ** -max(0, math.ceil(math.log2(max(mx, 1e-6) / 32)))
        scales["final"] = 2.0 ** -max(0, math.ceil(math.log2(st["final.in"] / 32)))
        m = DecANE(sd, scales=scales).eval()
        lat = torch.zeros(1, 48, 1, FR); val = torch.ones(1, 1, 1, T)
        tr = torch.jit.trace(m, (lat, val))
        ml = ct.convert(tr, inputs=[ct.TensorType("lat", (1, 48, 1, FR), dtype=np.float16),
                                    ct.TensorType("valid", (1, 1, 1, T), dtype=np.float16)],
                        outputs=[ct.TensorType("hidden", dtype=np.float16)],
                        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_ONLY)
        ml.short_description = f"Liquid-Codec decoder LFM stack, fp16, CTX={CTX} NEW={NEW} codec frames; tail on CPU fp32"
        ml.save(sys.argv[3])
        # fp16 numeric check on CPU via coremltools is macOS-only; on Linux report the torch fp16-free path only
        json.dump(dict(scales=scales), open(sys.argv[3] + ".scales.json", "w"), indent=1)
        print("SAVED", sys.argv[3], scales)


if __name__ == "__main__":
    main()
