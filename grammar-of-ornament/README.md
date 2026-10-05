# Grammar of Ornament

A local browser for the patterns an image classifier has learned to detect. Pick a model, a layer, and a channel (or one neuron at one position of that channel's feature map), and the app synthesizes the image that makes it fire hardest. Save the ones you like into `saved/`.

The emphasis is on low-level features: layers are listed shallowest first, grouped by feature-map size, and those early layers are where the abstract forms live (edges, gratings, dots, lattices, diagonal weaves). Deeper layers drift toward fur, eyes and faces.

## Setup

```
cd grammar-of-ornament
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Needs Python 3.10+. On an Apple-silicon Mac the GPU is used automatically (PyTorch's MPS backend); set `GOO_DEVICE=cpu` to force the CPU. The first use of each model downloads its weights to the PyTorch cache (ConvNeXt-Base is about 340 MB).

## Run

```
.venv/bin/python server.py
```

or `./launch.sh` to start it in a new Terminal window. The server binds to `127.0.0.1` only and opens `http://127.0.0.1:8790/` in your browser. Leave the window open; Ctrl-C stops it.

## Using it

- **Left pane.** Choose a built-in model (`convnext_base` by default; also `convnext_tiny`, `resnet50`, `vgg16`) or type any `timm` model name. Below it, every convolutional layer and block, shallow to deep.
- **Center.** Thumbnails of the layer's channels, 32 per page. They are quick 128-pixel renders and are cached, so a page you have seen before appears instantly. Click one to open it. A ★ marks channels you have saved from.
- **Right.** The selected channel rendered with the detail settings:
  - **objective**: `channel` maximizes the mean activation over the whole map; `neuron` maximizes one position. In neuron mode, click on the image to choose the position (the yellow box shows it on the layer's H×W grid).
  - **negate**: drive the activation negative instead.
  - **size**, **steps**, **seed**: output resolution, optimization steps, random start. `new seed` (or `r`) re-rolls.
  - **low-freq bias**: how strongly the parameterization favours low frequencies (1.0 is the Lucid/lucent default; higher is smoother and more saturated).
  - **smoothness**: a total-variation penalty, 0 to 10. Both of these reduce high-frequency striping at the cost of detail.
  - **Save** (or `s`) copies the current render into `saved/` with a JSON sidecar recording every setting. **Saved** tab lists them; `open` restores a saved image's exact settings in the browser.
- `←` / `→` step through channels.
- The URL hash tracks the current model, layer and channel (`#model=convnext_base&layer=features.1.0&c=13`), so a neuron can be bookmarked or pasted into a note.

## How it works

`vis.py` is a self-contained implementation of the Lucid/lucent feature-visualization recipe (Olah, Mordvintsev & Schubert, *Feature Visualization*, 2017): the image is parameterized by its Fourier spectrum with a frequency-dependent scale, colours are decorrelated with a fixed ImageNet colour matrix, and each optimization step sees a randomly padded, jittered, scaled and rotated copy of the image so the result is robust to those transforms. Adam maximizes the chosen activation. A forward hook on the target layer stops the forward pass there, so shallow layers are cheap.

Layers are discovered by running one forward pass with hooks on every module and keeping each one whose output is an N×C×H×W map: leaf convolutions and whole blocks (ConvNeXt blocks, ResNet bottlenecks), minus plain containers.

## Files

| file | purpose |
|---|---|
| `vis.py` | model registry, layer catalog, image parameterization, transforms, render loop; also a CLI (`python3 vis.py --help`) |
| `server.py` | local HTTP server, render queue, cache, save/delete |
| `webapp.html` | the page |
| `launch.sh` | opens the server in a new macOS Terminal window |
| `cache/` | every render, keyed by its parameters (gitignored) |
| `saved/` | the images you kept, each with a `.json` sidecar, indexed in `index.json` |
