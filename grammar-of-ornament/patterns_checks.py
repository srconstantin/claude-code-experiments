"""Follow-up checks for patterns.py: repeats at motif scale (10–90px), irregular repetition (a ring in the radial autocorrelation), grid-aligned directions, and box-counting controls (single-scale and scale-free noise). Run after patterns.py."""
import sys, csv, json
sys.path.insert(0, "/Users/sarahconstantin/Documents/Documents/claude-code-experiments/grammar-of-ornament")
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
import patterns as P
HERE = P.HERE
rows = list(csv.DictReader(open(HERE / "patterns" / "features.csv")))
rng = np.random.default_rng(0)

def ring(a):
    n = a.shape[0]; c = n // 2
    yy, xx = np.mgrid[0:n, 0:n] - c
    r = np.round(np.hypot(yy, xx)).astype(int)
    prof = np.bincount(r.ravel(), a.ravel()) / np.bincount(r.ravel())
    prof = prof[:100]
    # first minimum after 0, then the highest later maximum
    d = np.diff(prof)
    mins = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    if not len(mins): return 0.0, 0.0, 0.0
    m0 = mins[0]
    k = m0 + np.argmax(prof[m0:])
    return float(prof[k] - prof[m0]), float(k), float(prof[m0])

out = {}
for r in rows:
    x = P.luminance(HERE / r["folder"] / f"{r['name']}.png")
    a = P.autocorr(x)
    lat10 = P.lattice(a, rmin=10, rmax=90)
    rs, rr, rmin = ring(a)
    out[r["name"]] = (r, lat10, rs, rr)

def stride_groups():
    return sorted({int(r["stride"]) for r in rows})

print("LARGE-SCALE REPEATS (peaks at 10–90px): share with autocorrelation ≥ 0.2 / ≥ 0.3, and types among those ≥ 0.2")
types = ["square", "hexagonal", "rectangular", "centered", "oblique", "stripes"]
for label, sel in [("vgg16", lambda r: r["model"] == "vgg16"), ("resnet50", lambda r: r["model"] == "resnet50"), ("convnext", lambda r: r["model"] == "convnext_base")] + \
                  [(f"stride {s}", (lambda s: lambda r: int(r["stride"]) == s)(s)) for s in stride_groups()]:
    items = [v for v in out.values() if sel(v[0])]
    st = np.array([v[1].get("strength", 0) for v in items])
    per = [v for v in items if v[1].get("strength", 0) >= 0.2]
    print(f"  {label:11s} n={len(items):3d}  ≥0.2 {100*np.mean(st>=0.2):3.0f}%  ≥0.3 {100*np.mean(st>=0.3):3.0f}%   " +
          "  ".join(f"{t} {sum(1 for v in per if v[1].get('type')==t)}" for t in types) +
          (f"   period median {np.median([v[1]['period'] for v in per]):.0f}px" if per else ""))

print("\nIRREGULAR REPETITION — ring in the radial autocorrelation (evenly spaced motifs without a lattice)")
for label, sel in [("vgg16", lambda r: r["model"] == "vgg16"), ("resnet50", lambda r: r["model"] == "resnet50"), ("convnext", lambda r: r["model"] == "convnext_base")] + \
                  [(f"stride {s}", (lambda s: lambda r: int(r["stride"]) == s)(s)) for s in stride_groups()]:
    items = [v for v in out.values() if sel(v[0])]
    rs = np.array([v[2] for v in items]); rr = np.array([v[3] for v in items])
    print(f"  {label:11s} ring height ≥0.1: {100*np.mean(rs>=0.1):3.0f}%   ≥0.2: {100*np.mean(rs>=0.2):3.0f}%   ring radius median (those ≥0.1) {np.median(rr[rs>=0.1]) if (rs>=0.1).any() else float('nan'):.0f}px")

print("\nSTRIPE / LATTICE DIRECTIONS (all features with any lattice peak ≥0.2 at 3–80px): share within ±7.5° of 0°, 45°, 90°, 135° vs 33% for uniform")
angs = np.array([float(r["angle1"]) for r in rows if r.get("angle1") and float(r.get("strength") or 0) >= 0.2])
near = np.min(np.abs(((angs[:, None] - np.array([0, 45, 90, 135])[None, :]) + 90) % 180 - 90), axis=1) <= 7.5
print(f"  n={len(angs)}  near grid angles {100*near.mean():.0f}%")

# controls for box counting: thresholded band-pass noise, one scale only, at 896px
print("\nBOX-COUNTING CONTROLS (896px, boxes 2–128px): textures with a single characteristic scale")
for sig in (3, 6, 12, 24):
    z = rng.standard_normal((896, 896))
    b = ndi.gaussian_filter(z, sig) - ndi.gaussian_filter(z, sig * 1.6)
    ink = b > np.quantile(b, 0.45)
    edge = ink ^ ndi.binary_erosion(ink)
    D, r2 = P.box_dimension(edge)
    Di, r2i = P.box_dimension(ink)
    print(f"  blob size ~{sig*2}px: edge D {D:.2f} R² {r2:.3f}   ink D {Di:.2f} R² {r2i:.3f}")

# local slopes per octave for prints vs control
def local_slopes(edge, sizes=(2,4,8,16,32,64,128)):
    n=edge.shape[0]; c=[]
    for s in sizes:
        m=n//s; c.append(edge[:m*s,:m*s].reshape(m,s,m,s).any(axis=(1,3)).sum())
    c=np.log(np.array(c,float)); return -(np.diff(c)/np.log(2))
lw = HERE / "linework" / "vgg16features"
sl = np.array([local_slopes((lambda ink: ink ^ ndi.binary_erosion(ink))(P.ink_of_classes(p))) for p in sorted(lw.glob("*.classes.png"))])
print("\nLOCAL BOX DIMENSION PER OCTAVE (edge), prints median:   " + "  ".join(f"{2**i}-{2**(i+1)}px {v:.2f}" for i, v in enumerate(np.median(sl, 0), 1)))
z = rng.standard_normal((896, 896)); b = ndi.gaussian_filter(z, 6) - ndi.gaussian_filter(z, 9.6); ink = b > np.quantile(b, .45)
print("  single-scale control (~12px blobs):              " + "  ".join(f"{2**i}-{2**(i+1)}px {v:.2f}" for i, v in enumerate(local_slopes(ink ^ ndi.binary_erosion(ink)), 1)))
# multiscale control: sum of band-pass noise over all scales (1/f)
z = rng.standard_normal((896, 896)); F = np.fft.fft2(z); fy = np.fft.fftfreq(896)[:, None]; fx = np.fft.fftfreq(896)[None, :]
f = np.hypot(fy, fx); f[0, 0] = 1; frac = np.real(np.fft.ifft2(F / f ** 1.4)); ink = frac > np.quantile(frac, .45)
print("  scale-free control (fractal noise, β≈2.8):        " + "  ".join(f"{2**i}-{2**(i+1)}px {v:.2f}" for i, v in enumerate(local_slopes(ink ^ ndi.binary_erosion(ink)), 1)))

# save clearest large-scale lattices for a look
json.dump({k: v[1] for k, v in out.items()}, open(HERE / "patterns" / "lattice_10px.json", "w"))
top = sorted(out.values(), key=lambda v: -v[1].get("strength", 0))[:12]
print("\nCLEAREST LARGE-SCALE REPEATS:")
for r, lat, rs, rr in top:
    print(f"  {r['model']:13s} {r['layer']:14s} c{r['name'].split('__')[2][1:]:>5s}  {lat.get('type','–'):11s} strength {lat.get('strength',0):.2f} period {lat.get('period',0):.0f}px  basis angle {lat.get('basis_angle',float('nan')):.0f}°")
