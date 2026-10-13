# Grammar of Ornament

A local browser for the patterns an image classifier has learned to detect. Pick a model, a layer, and a channel (or one neuron at one position of that channel's feature map), and the app synthesizes the image that makes it fire hardest. Save the ones you like into `features/`.

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
  - **Save** (or `s`) copies the current render into `features/` with a JSON sidecar recording every setting. The **Features** tab lists them; `open` restores a saved image's exact settings in the browser.
- `←` / `→` step through channels.
- The URL hash tracks the current model, layer and channel (`#model=convnext_base&layer=features.1.0&c=13`), or `#tab=features` for the gallery, so a neuron can be bookmarked or pasted into a note.

## Outlines

`outlines.py` turns every image in `features/` into a black-and-white ink drawing in `outlines/`.

```
.venv/bin/python outlines.py                       # everything not yet up to date
.venv/bin/python outlines.py --force               # redo all
.venv/bin/python outlines.py --xdog-eps 0.5 --xdog-p 20 --sigma 1.5        # finer, lighter
.venv/bin/python outlines.py --method contour                               # topo-map isolines instead
.venv/bin/python outlines.py --method contour --levels 0.5 --sigma 3 --width 2.5   # one bold contour per form
```

It starts from the same continuous-tone field as `dithered_forms.py` (each pixel projected onto the image's principal colour axis). The default method is **XDoG** (extended difference of Gaussians), the standard ink-drawing filter: the tone is compared with a blurred copy of itself, and places that are darker than their surroundings are inked, with a soft threshold, so the result reads like a woodcut. `--sigma` sets the scale of the comparison, `--xdog-p` the sharpening, `--xdog-eps` how much of the image gets inked, `--xdog-phi` how hard the threshold is.

`--method contour` draws **isolines** of the smoothed field instead: closed curves where the tone crosses each of the `--levels` (default five levels from 0.2 to 0.8), which run perpendicular to the brightness gradient and so follow the curves of the shapes. Contours shorter than `--min-len` are dropped and lines are drawn at `--width`. `--method canny` is the old sharpest-gradient edge detector, kept for comparison. `--gamma` and `--invert` reshape the tone before any method.

## Forms

`forms.py` turns every image in `features/` into contiguous black-and-white regions in `forms/`.

```
.venv/bin/python forms.py
.venv/bin/python forms.py --only features.3 --sigma 4 --min-area 0.01
```

Rather than converting to grayscale, each image is projected onto the colour axis along which it varies most (the first principal component of its pixels in Lab space, `--space rgb` for RGB), so features that differ in hue but not brightness still split cleanly. The projection is blurred (`--sigma`), thresholded with Otsu's method, and the darker class becomes black (`--invert` swaps). Regions and holes smaller than `--min-area` (a fraction of the image) are removed and the boundaries are rounded with an opening/closing of radius `--smooth`. Output is strictly black and white; `--soft` keeps anti-aliased edges.

## Dithered forms

`dithered_forms.py` renders every image in `features/` as a dithered one-bit image in `dithered-forms/`.

```
.venv/bin/python dithered_forms.py
.venv/bin/python dithered_forms.py --method bayer --gamma 1.3
```

It uses the same principal-colour-axis projection as `forms.py`, oriented so brighter in the original is lighter, but keeps it as continuous tone (stretched between the `--clip` percentiles, shaped by `--gamma`) and dithers it to black and white: Floyd–Steinberg error diffusion by default, or an ordered 8×8 Bayer pattern with `--method bayer`. Output is 2× the feature's size by default (`--scale`) so the dither grain is fine relative to the forms; at native size the grain swamps them.

## Palettes

`palettes.py` picks seven colours from every image in `features/` and writes a strip of swatches with hex codes to `palettes/`, plus a `.json` sidecar listing them.

```
.venv/bin/python palettes.py
.venv/bin/python palettes.py --method spread --n 5
```

Colours are chosen in CIELAB so the spacing is perceptual, and the strip is ordered dark to light. The default `spread` method is farthest-point sampling: start from the pixel colour farthest from the image's mean, then repeatedly add the pixel colour farthest from everything chosen so far. Every swatch is a real pixel colour, stray colours (fewer than `--min-frac` of the pixels, about ten pixels by default) are excluded as candidates, and the set is as widely spread through the image's colour space as seven points can be, so red, yellow and blue blobs each get a swatch. `--method axis` instead takes equal steps along the image's principal colour axis and `--method kmeans` takes cluster centres; both average hues together and come out muted. A light blur (`--sigma`) is applied first so the fine hatching doesn't contribute colours.

## Motifs

The **Motifs** tab turns a feature into a pure black-and-white print and lets you cut single motifs out of it by hand. Pick a feature (or press **motif** on a card in the Features tab). The first time a feature is opened it takes about half a minute to prepare; after that every change is immediate.

What happens to the feature:

1. **Upscale.** The 224px feature is enlarged 4× with Real-ESRGAN (the photo model), a learned super-resolution network that turns the pixel mush of a small image into clean shapes while staying faithful to it. The result is kept in `cache/sr/`; the weights (64 MB) are downloaded once into `models/`.
2. **Put the fine detail back.** The upscaler smooths away the very finest texture, such as dot lattices three pixels apart. The difference between the original's fine detail and the upscaled image's is added back, so dots and hatching return exactly where the feature has them.
3. **One threshold over shading, texture and outlines.** The shading (smoothed without crossing the edges of forms), the image's own fine texture, the restored detail and the colour boundaries between forms are added together and cut at a single level, at 1792px. Because texture is added to shading before the cut, mark width follows the shading: thin white marks in the darks, thin black ones in the lights, solid masses where the tone is extreme.

The controls: **texture** is how much the image's own detail counts against its shading (0 gives flat shapes, higher brings out pattern everywhere); **fine detail** is how much of the restored texture is added; **outlines** is the strength of the black lines where forms meet, including forms that differ only in colour; **threshold** makes the print lighter or darker; **invert** swaps black and white; **reset** returns to the defaults.

To take one motif, drag around it on the print. The outline is smoothed into a closed curve; with **snap outline** on it is rerouted along the strongest contours near where you drew. Everything outside becomes the **ground** colour (white or black), `Esc` cancels a drag, **clear outline** removes it, and **show feature** fades the upscaled colour image in underneath so you can see what you are outlining. **Save motif** writes the print, trimmed to a square around the outline (or the whole print if there is none), into `motifs/<feature>/<id>[-label]/` as `motif.png` (one bit per pixel), `motif.svg` (the black areas as one path) and `motif.json` (the settings and the outline). Saved motifs are listed under the print; **reopen** loads a motif's settings and outline back into the tool.

```
.venv/bin/python motifs.py                                   # every whole feature → prints/<name>.png
.venv/bin/python motifs.py --only c141 --texture 0.3 --restore 0.6 --force
```

The pipeline is deterministic: the same feature and settings always give the same image.

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
| `features/` | the images you kept, each with a `.json` sidecar, indexed in `index.json` |
| `outlines.py` | makes line drawings of every feature image |
| `outlines/` | the line drawings, one per feature image, same file names |
| `forms.py` | makes black-and-white region maps of every feature image |
| `forms/` | the region maps, same file names |
| `dithered_forms.py` | renders every feature image as dithered black and white |
| `dithered-forms/` | the dithered images, same file names |
| `palettes.py` | picks seven colours from every feature image |
| `motifs.py` | the black-and-white print pipeline (upscale, detail restoration, threshold), outlines, saved motifs; also a batch CLI |
| `motifs/` | saved motifs, one folder per motif under its feature |
| `models/` | the upscaler's weights, downloaded on first use (gitignored) |
| `prints/` | whole-feature prints from `motifs.py` run as a script |
| `palettes/` | swatch strips with hex codes, plus `.json` colour lists |
