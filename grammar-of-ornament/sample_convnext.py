"""Render N random channels of resnet50, spread evenly across the block layers
(layerX.Y), at the webapp's default settings, and copy them into OUT with
JSON sidecars matching the app's Save format."""
import json, random, re, shutil, sys, time, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8790"
ROOT = Path("/Users/sarahconstantin/Documents/Documents/claude-code-experiments/grammar-of-ornament")
OUT = ROOT / "convnext_basefeatures"
MODEL, N, SEED = "convnext_base", 100, 20261015
S = {"model": MODEL, "size": 224, "steps": 512, "seed": 0, "decay": 1.25, "tv": 0.0}

def api(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

layers = [l for l in api(f"/api/layers?model={MODEL}")["layers"] if re.fullmatch(r"features\.\d+\.\d+", l["name"]) and l["name"] not in ("features.0.0", "features.2.1", "features.4.1", "features.6.1")]
print(f"{len(layers)} block layers:", " ".join(l["name"] for l in layers), flush=True)

rng = random.Random(SEED)
base, extra = divmod(N, len(layers))
bonus = set(rng.sample(range(len(layers)), extra))   # which layers get one more
plan = []
for i, l in enumerate(layers):
    k = base + (1 if i in bonus else 0)
    plan.append((l["name"], sorted(rng.sample(range(l["channels"]), k))))
for name, cs in plan:
    print(f"  {name}: {len(cs)} channels {cs}", flush=True)

# submit one job per layer, then poll until all done
jobs = {}
for name, cs in plan:
    r = api("/api/render", {**S, "layer": name, "targets": [{"c": c} for c in cs]})
    jobs[name] = r
    print(f"submitted {name}: {'cached' if r.get('done') else r['job_id']}", flush=True)

t0 = time.time()
pending = {n: j["job_id"] for n, j in jobs.items() if not j.get("done")}
while pending:
    time.sleep(10)
    for n, jid in list(pending.items()):
        j = api(f"/api/job/{jid}")
        if j["error"]:
            sys.exit(f"job {n} failed: {j['error']}")
        if j["done"]:
            jobs[n] = j; del pending[n]
            print(f"  [{time.time()-t0:6.0f}s] {n} done ({len(j['images'])} images); {len(pending)} layers left", flush=True)
        else:
            print(f"  [{time.time()-t0:6.0f}s] {n} {j['progress']*100:.0f}%", flush=True)

OUT.mkdir(exist_ok=True)
index = []
for name, _ in plan:
    for im in jobs[name]["images"]:
        src = ROOT / im["url"].lstrip("/")
        t = im["target"]
        label = src.stem.split("__")[0]
        out_name = f"{MODEL}__{name}__{label}__{S['size']}px__s{S['seed']}__{src.stem.rsplit('__', 1)[1][:6]}"
        shutil.copyfile(src, OUT / f"{out_name}.png")
        item = {"name": out_name, "url": f"/convnext_basefeatures/{out_name}.png",
                "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "random sample, default settings",
                **S, "layer": name, "target": t}
        (OUT / f"{out_name}.json").write_text(json.dumps(item, indent=1))
        index.append(item)
(OUT / "index.json").write_text(json.dumps(index, indent=1))
print(f"wrote {len(index)} images to {OUT}", flush=True)
