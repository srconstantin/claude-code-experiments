# Grammar of Ornament

A local browser for the patterns an image classifier has learned to detect. Pick a model, a layer, and a channel (or one neuron at one position of that channel's feature map), and the app synthesizes the image that makes it fire hardest. Save the ones you like into `vgg16features/` (the saved-features folder, `SAVED` in `server.py`).

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
  - **Save** (or `s`) copies the current render into `vgg16features/` with a JSON sidecar recording every setting. The **Features** tab lists them; `open` restores a saved image's exact settings in the browser.
- `←` / `→` step through channels.
- The URL hash tracks the current model, layer and channel (`#model=convnext_base&layer=features.1.0&c=13`), or `#tab=features` for the gallery, so a neuron can be bookmarked or pasted into a note.

## Outlines

`outlines.py` turns every image in `vgg16features/` (or `--src` another folder) into a black-and-white ink drawing in `outlines/`.

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

`forms.py` turns every image in `vgg16features/` (or `--src` another folder) into contiguous black-and-white regions in `forms/`.

```
.venv/bin/python forms.py
.venv/bin/python forms.py --only features.3 --sigma 4 --min-area 0.01
```

Rather than converting to grayscale, each image is projected onto the colour axis along which it varies most (the first principal component of its pixels in Lab space, `--space rgb` for RGB), so features that differ in hue but not brightness still split cleanly. The projection is blurred (`--sigma`), thresholded with Otsu's method, and the darker class becomes black (`--invert` swaps). Regions and holes smaller than `--min-area` (a fraction of the image) are removed and the boundaries are rounded with an opening/closing of radius `--smooth`. Output is strictly black and white; `--soft` keeps anti-aliased edges.

## Dithered forms

`dithered_forms.py` renders every image in `vgg16features/` (or `--src` another folder) as a dithered one-bit image in `dithered-forms/`.

```
.venv/bin/python dithered_forms.py
.venv/bin/python dithered_forms.py --method bayer --gamma 1.3
```

It uses the same principal-colour-axis projection as `forms.py`, oriented so brighter in the original is lighter, but keeps it as continuous tone (stretched between the `--clip` percentiles, shaped by `--gamma`) and dithers it to black and white: Floyd–Steinberg error diffusion by default, or an ordered 8×8 Bayer pattern with `--method bayer`. Output is 2× the feature's size by default (`--scale`) so the dither grain is fine relative to the forms; at native size the grain swamps them.

## Palettes

`palettes.py` picks seven colours from every image in `vgg16features/` (or `--src` another folder) and writes a strip of swatches with hex codes to `palettes/`, plus a `.json` sidecar listing them.

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

To take one motif, drag around it on the print. The outline is smoothed into a closed curve; with **snap outline** on it is rerouted along the strongest contours near where you drew. Everything outside becomes the **ground** colour (white or black), `Esc` cancels a drag, **clear outline** removes it, and **show feature** fades the upscaled colour image in underneath so you can see what you are outlining. **Save motif** writes the print, trimmed to a square around the outline (or the whole print if there is none), into `vgg16motifs/<feature>/<id>[-label]/` (`MOTIFS` in `motifs.py`) as `motif.png` (one bit per pixel), `motif.svg` (the black areas as one path) and `motif.json` (the settings and the outline). Saved motifs are listed under the print; **reopen** loads a motif's settings and outline back into the tool.

```
.venv/bin/python motifs.py                                   # every whole feature → prints/<name>.png
.venv/bin/python motifs.py --only c141 --texture 0.3 --restore 0.6 --force
```

The pipeline is deterministic: the same feature and settings always give the same image.

## Linework

`linework.py` takes the print of every feature in a folder apart into strokes and measures them: the line quality of the feature.

```
.venv/bin/python linework.py                                  # vgg16features/ → linework/vgg16features/
.venv/bin/python linework.py --src resnet50features --only layer2
.venv/bin/python linework.py --max-stroke-width 4 --force
```

It starts from the motif print at the default settings (so the first pass over a feature upscales it, about 15 s, cached in `cache/sr/`) and separates it by scale. Black that survives a morphological opening with a disc of radius half `--max-stroke-width` (6 source pixels by default) is a **mass**; the thin black that remains is **strokes**, except pieces no longer than about twice their width, which are **dots**. The thin white marks the print cuts into the masses are treated the same way as **relief** strokes. Each stroke is skeletonized and cut at its junctions, and every branch becomes a smoothed centerline sampled every 3 print pixels with the mark's width at each sample (twice the distance to the nearest edge).

From those centerlines come the measurements, in source pixels: median width and its spread, median and longest length, straightness (chord over arc length), curvature, **bend** (width over radius of curvature, which is scale-free: how much a line curves relative to its own thickness), inflections per unit length, taper at free ends (end width over middle width), the fraction of closed loops, junctions per unit length, orientation anisotropy and the dominant angle, and **parallelism** (over sampled points, the share of neighbouring strokes within three widths that run within 15° of the same direction), plus how much of the image is mass, stroke and dot.

Per feature it writes `<name>.strokes.svg`, the masses filled and every stroke as an SVG stroked path at its own mean width with round caps (reusable as strokes, recolourable, rescalable), `<name>.outlines.svg`, the same strokes as filled outlines with their variable width (faithful to the print), `<name>.strokes.json`, the per-stroke table (every stroke's centerline and width at each point in print pixels, with its measurements in source pixels, whether it is closed, and whether each end is free or cut at a junction), `<name>.classes.png`, the print with masses grey, strokes black, dots blue, relief strokes orange and relief dots green, for checking the separation, and `<name>.json` with the measurements. Over the folder it writes `summary.json` and `summary.csv` and prints a table of per-layer medians.

## Stroke families

`strokelib.py` sorts every stroke of a feature set into families and picks representative strokes from each.

```
.venv/bin/python strokelib.py                                 # linework/vgg16features/ → strokelib/vgg16features/
.venv/bin/python strokelib.py --ends both-free --dst strokelib/vgg16features-whole
.venv/bin/python strokelib.py --k 16 --per-family 24 --width-scale 0.6
```

Each stroke from the per-stroke tables is described by ten numbers: log length, log width, straightness, bend, inflections, total turn, taper, width unevenness, whether it is closed, and how many of its ends are free (a stroke cut at a junction on both ends is a *segment* of a mesh, cut at one end a *branch*, free at both a whole mark). The descriptors are standardized and clustered with k-means into `--k` families (12 by default), ordered largest first. Each family is named from its typical member, for instance "short tapered wave branch, fine", and its `--per-family` representatives are the members nearest its centre, at most two from any one feature. `--ends one-free` or `both-free` restricts the library to strokes with natural endings; `--pol ink` or `relief` to black marks or white ones; `--min-length` drops the shortest.

Every representative is written as its own SVG, `family-NN-<label>/stroke-MM.svg`: the stroke as a filled outline with its measured width at every point, times `--width-scale` (1 by default, no minimum), in its original orientation, at 8 px per source pixel. The folder also gets `families.png`, a contact sheet with one row per family drawn at a single scale with a scale bar, a `sheet.png` per family, `families.json` (label, size, share, typical measurements, how many strokes each layer and each polarity contributes, and the representatives with their measurements), `families.csv`, and `by_feature.csv` (the share of each feature's strokes in each family). It prints the family table and the share of each layer's strokes by family.

## Compounds

`compounds.py` puts the strokes back together into structures made of several marks, which strokelib cannot show because it files every mark cut at its junctions.

```
.venv/bin/python compounds.py                                 # linework/vgg16features/ → compounds/vgg16features/
.venv/bin/python compounds.py --per-type 24 --gap 3 --min-teeth 5
```

It reads the per-stroke tables and builds a graph of how strokes connect: at junctions, and, for black strokes, through the masses they run into (the thin mask is cut around masses, so a line that passes through a blob is rejoined here, `--attach`). Three kinds of compound come out:

- **Trees**: connected groups of ink or relief strokes. A big network is peeled into pieces, each the heaviest path through it (length times width, so a fat spine wins over a zigzag through thin teeth) plus the branches hanging directly off that path. Each piece is measured by its spine (length, straightness, waviness, width), its side branches (how many per length of spine, how long relative to the spine, their angle to it, whether they hang off one side or all point the same way, how regularly they are spaced), and its cycles. Rules on those give the types: **comb** (many short, regular teeth on one side), **feather** (two-sided), **branching line** (a few long branches), **sprig**, **tree**, **fork** and **burst** (one junction), **loop** and **mesh** (cycles), with the spine described as straight, curved or wavy.
- **Bundles**: groups of near-parallel strokes within `--gap` source px of each other that overlap along their length: **hatching**, **wavy hatching**, and **crosshatching** where two bundles cross at 45° or more in the same place, with "fine" for spacing under 2.5 px. Measured by spacing and its regularity, orientation, stroke length and straightness.
- **Bands**: masses whose spine (traced by linework as a third polarity) is at least `--min-band-aspect` times their width: slow curves at the thick scale, measured like strokes, with the relief hatching that sits on them attached, giving **band**, **hatched band**, **crosshatched band** and **lobed band**.

Each compound is rendered whole as one SVG (masses filled, strokes as variable-width outlines, relief marks white over their mass) into `<type>/NN.svg`; each type gets half its `--per-type` representatives by typicality and half by size, at most two from any one feature, on a `sheet.png`. The folder also gets `compounds.png` (one row per type, one scale, with a scale bar), `compounds.json` (every type's count, typical measurements, counts by layer and the representatives with their measurements) and `by_feature.csv`. Types with fewer than `--min-count` examples are not shown. A table of compounds per feature by layer is printed at the end.

## Colorways

`colorways.py` turns the colour of every feature in a folder into gradients, duotones and mesh backgrounds.

```
.venv/bin/python colorways.py --src convnext_basefeatures     # → colorways/convnext_basefeatures/
.venv/bin/python colorways.py --src vgg16features --mesh-sigma 8
```

Feature visualizations pair opponent colours: the renderer decorrelates the colour channels, so a feature that likes one hue dislikes its opposite and the image swings between the two. For each image the script finds that **pair** as the two ends of the dominant chroma axis (the first principal component of a and b in CIELAB after a light blur; each end is the mean of the 5% of pixels farthest that way), and reports the hue angle between them (180° would be exact opponents) and how much of the image's chroma variance the axis explains. From the pair come a **gradient** of seven stops along the axis, each the chroma-weighted mean colour of one septile of the pixels, so it passes through the image's own intermediate colours, and a pure pair gradient interpolated in Lab; a **duotone**, the pair pushed to a dark shadow and a light highlight and written as an SVG filter that maps any image's luminance onto them; and a **mesh** background, the image blurred heavily with the chroma lifted and summarized as CSS, a base colour plus a few radial gradients at its colour blobs. (A tileable version of the blurred field was tried and dropped: the renders have a vignette, so the tiles read as a grid.)

Per feature it writes `<name>.json` (hex and Lab colours, CSS strings, the SVG filter) and `<name>.card.png` (the feature, the pair, both gradients, the duotone applied to the feature, the mesh field). For the folder it writes `colorways.css` (custom properties `--cw-<short>-a`, `-b`, `-shadow`, `-highlight` and classes `.cw-<short>-gradient`, `-pair`, `-mesh`, `-duotone`), `filters.svg` (the duotone filters, `duo-<short>`), `gallery.html` (everything on one page, opened from that folder), `summary.json`, `summary.csv`, and `pairs.png` (every pair, sorted by hue). It prints a table of which named hue pairs occur in each layer stage.

## Ornament and style tiles

`ornament.py` draws new vector ornaments from the measured proportions of the strokes and compounds, rather than cutting real ones out of features, so they look like the features without copying any, repeat seamlessly, and are reproducible from a seed. Strokes are centerlines (length, turn, waviness) drawn with tapered width profiles. Friezes are single seamless tiles: `comb_frieze` (spacing, spine width and 64° tooth angle from the measured straight comb), `feather_frieze` (barbs on both sides), `hatch_frieze` (the measured fine hatching), `wave_frieze` (the 'short wave' family end to end), `band_frieze` (a woodcut band with gouges cut into its edges) and `dash_frieze` (hairline dashes in glide reflection). `rosette` repeats any element round a centre, with or without mirrors; `check`, `loop`, `caret`, `gouges` and `band_mark` are glyphs for controls.

`styletiles.py` builds three directions for a web component library as one page each with the same components, in `designlib/tiles/`:

```
.venv/bin/python styletiles.py                 # → designlib/tiles/{woodcut,engraving,filigree}.html
.venv/bin/python styletiles.py --screenshot    # also PNGs via headless Chrome
```

Open them through the server, at `http://127.0.0.1:8790/designlib/tiles/`, so saving works.

- **Woodcut**: chipped slabs with relief gouges at the button ends, torn-bottom cards, gouged bands; black on paper with one hot accent; the hero is a motif from `vgg16motifs/`.
- **Engraving**: cartouche buttons with comb fringes, cards with concave corners, a double rule and rosette corners, swallowtail ribbon badges and flag alerts, feather underlines, hatched rules; violet ink on cream with ochre; the hero is the vgg16 feature richest in combs and feathers, as a duotone in a cartouche.
- **Filigree**: ogee-leaf buttons, pointed-arch cards crowned with a double rosette, running-scroll rules and dividers; gold and cream on night blue; the hero is a convnext feature under an arch.

Shapes are not rounded rectangles. `designlib/tiles/shapes.js` draws each shaped element (`data-shape="slab"`, `cartouche`, `ribbon`, `leaf`, `arch` and so on) as an SVG computed from the element's actual size, so points and curves never stretch; it can add an inner rule and clip the element's content (a card's picture) to the shape. Ornaments are named recipes from `ornament.py` with parameters (`SPECS` in `styletiles.py`), so any of them can be regenerated.

Every component and every ornament is a **specimen**: hovering it shows **↻ variant** (a new seed and proportions jittered by up to 18% for its ornaments, and a new seed for its shapes), **Save**, and for ornaments **SVG** (download). Save writes `designlib/saved/<id>/` with `specimen.html` (a standalone page: the theme's styles and fonts, the shapes script, and the element exactly as shown, variants included), `meta.json`, and each of its ornaments as an SVG. `http://127.0.0.1:8790/designlib/saved/` shows everything saved, with links and delete. The page's own controls work too: tabs switch, "Save pattern" saves its specimen, "Export SVG" downloads the divider, "Show layers" outlines the drawn shapes. Opened as a plain file the pages still render and shape variants work, but saving and ornament variants need the server; `designlib_api.py` holds those routes.

Fills are the features themselves (upscaled), duotones of them in the page's inks, and colorway meshes; motifs only in large spots. Ornament sits at the edges of controls and never under words; colours come from the measured colorway pairs; focus and large-text accents are adjusted until they pass, and every text and background pair is checked against WCAG contrast and listed at the foot of the page.

## Atlas and the compound tiles

`atlas.py` collects real compound pieces (and, in group `stroke`, the strokelib families' representative strokes) for design use: nothing is generated. Mark clusters also get a `hull`, their silhouette. From every vgg16 feature it takes the compounds `compounds.py` finds, ranks each type by a score that favours clear examples (combs with many aligned, evenly spaced teeth; crosshatching with many strokes crossing near 90°; large bands and blobs; long branching lines), keeps the top twelve per type with at most two per feature, turns each so its long axis is horizontal (combs with teeth down), smooths mass outlines lightly and writes them to `designlib/atlas/`: `atlas.json` (each piece's ink, relief and outline paths), one SVG per piece under its group (`comb`, `feather`, `crosshatch`, `hatching`, `blob`, `band`, `branch`) and `atlas.png`, a contact sheet.

```
.venv/bin/python atlas.py              # about a minute
.venv/bin/python atlastiles.py         # → designlib/tiles/{herbarium,bloom}.html
```

`sourcetiles.py` builds the current style tiles, one per source of shapes, each using only real atlas pieces and the same components:

```
.venv/bin/python sourcetiles.py [--screenshot]   # → designlib/tiles/{bands,marks,strokes}.html
```

- **Bands** (cream and gold on deep green) is built from the band and blob compounds. Buttons, alerts and badges are band outlines, and each line is a long band repeated. Card pictures are framed by *mostly convex* versions of bands and blobs (`convex` in the atlas: a closing with a disc about half the piece's size fills the bays between lumps but keeps the irregular contour; only shapes at least 95% convex that fill 70% of their bounding box are used), and a grid of feature cards shows the same frames at small size. Checks, dots and bullets are blobs.
- **Marks** (violet ink on cream) is built from hatching, crosshatching, combs, sprigs, loops, feathers and branching lines. Buttons and alerts are the silhouettes of crosshatch clusters and hatching bundles, and card pictures are framed by mostly convex versions of mark clusters (same thresholds as Bands). `atlas.py` computes each silhouette by closing the gaps between a cluster's marks. Each line is a feather, comb or hatching bundle repeated, checks and bullets are crosshatch clusters, the radio ring is the roundest loop, and the spinner is a comb.
- **Strokes** (deep blue and vermilion on white) is built from the strokelib families. Buttons, cards and alerts are framed by four real tapered strokes (`shapes.js` kind `frame`), each line is a wave stroke repeated, the check is a hook, the radio ring is a loop stroke, and bullets are tapered dashes.

Each element uses one shape: outlines never carry attached ornament, alerts have no mark (an error alert has a danger-coloured outline and title), dividers have no centre piece, and a checked box or selected radio is its own shape filled rather than a second mark inside it. Alert text is always inside the outline: for every outline the largest rectangle inside it is computed, and `shapes.js` (`data-contain`) sizes the shape so that rectangle covers the alert's text box; only shapes whose inner rectangle spans at least 55% of their width and 45% of their height are used for alerts, so a shape grows at most about twofold around its text. Every line is a single piece repeated with its ends overlapping. The pictures on each page come from `vgg16motifs/`, `linework/` (outline and stroke drawings, recoloured to the page's inks), the three features folders (vgg16, resnet50, convnext_base, upscaled) and the convnext colorways (meshes and gradients), and each card names its source. ↻ variant draws another piece of the same kind for both shapes and ornaments.

**Section edges.** Each page is a stack of full-width bands, one per section, alternating between the page's ground and a slightly tinted ground (5 points of lightness toward the ink, its hue nudged 7% toward the ink's). Each boundary is the top contour of a real piece repeated across the width: a long band's lumpy top on Bands, the silhouette of a hatching bundle or feather on Marks, a wave stroke on Strokes. The contour is the first ink in each column of the piece, smoothed and tilted so its ends meet, and filled below with the colour of the band it tops; each band starts the repeat at a different offset. The edge replaces the hero rule, and the tinted ground is included in the contrast checks. When the View panel puts a picture behind a band, the picture takes the same contour at its top and the next band's contour at its foot.

Each page has a **View** panel (`designlib/tiles/view.js`; it needs the server):

- **Colorway** recolours the whole page (inks, ground, accent and every ornament) from one of ten convnext colorways, the most vivid of each hue pair. On light pages the darker end of the pair becomes the ink and the other the accent; on dark pages the darker end becomes the ground. Every derived colour is pushed until its contrast check passes. The server route is `POST /api/designlib/palette`.
- **Background on** puts a picture behind the whole page, the hero band, the cards band, or every other section as full-width bands. Text and controls keep a solid panel of the page's ground, so words never sit on the picture.
- **Picture** is the feature in its own colours (the colorway's feature, or the page's), the feature remapped to the page's two colours, or a linework drawing (the outlines of a vgg16 feature) in the page's colours. The page-colour versions are made in the browser from whatever colorway is showing.

`python3 atlas.py --convex-only` recomputes just the mostly convex shapes of an existing atlas in a few seconds.

The earlier experiments (`styletiles.py`: woodcut, engraving; `atlastiles.py`: herbarium, bloom) are still built and listed at the foot of the index, but are no longer in the navigation. Filigree was removed.

Deleting a saved specimen from the gallery moves it to `designlib/saved/.trash/` rather than erasing it.

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
| `vgg16features/` | the images you kept, each with a `.json` sidecar, indexed in `index.json` |
| `outlines.py` | makes line drawings of every feature image |
| `outlines/` | the line drawings, one per feature image, same file names |
| `forms.py` | makes black-and-white region maps of every feature image |
| `forms/` | the region maps, same file names |
| `dithered_forms.py` | renders every feature image as dithered black and white |
| `dithered-forms/` | the dithered images, same file names |
| `palettes.py` | picks seven colours from every feature image |
| `motifs.py` | the black-and-white print pipeline (upscale, detail restoration, threshold), outlines, saved motifs; also a batch CLI |
| `vgg16motifs/` | saved motifs, one folder per motif under its feature |
| `models/` | the upscaler's weights, downloaded on first use (gitignored) |
| `prints/` | whole-feature prints from `motifs.py` run as a script |
| `linework.py` | strokes, dots and masses of every print, their measurements, and SVG exports |
| `linework/` | one folder per feature set: `.strokes.svg`, `.outlines.svg`, `.strokes.json`, `.classes.png`, `.json` per feature, plus `summary.json` / `summary.csv` |
| `strokelib.py` | clusters the strokes of a feature set into families and writes representative strokes |
| `strokelib/` | one folder per feature set: `families.png`, `families.json`, `families.csv`, `by_feature.csv`, and a folder of SVGs per family |
| `compounds.py` | structures made of several marks (combs, hatching, bands and more) with representatives of each type |
| `compounds/` | one folder per feature set: `compounds.png`, `compounds.json`, `by_feature.csv`, and a folder of SVGs per type |
| `ornament.py` | generators for friezes, rosettes and glyphs drawn from the measured stroke proportions |
| `styletiles.py` | three style-tile pages for a component library, in `designlib/tiles/` |
| `atlas.py` | real compound pieces, oriented and cleaned, in `designlib/atlas/` |
| `atlastiles.py` | the earlier Herbarium and Bloom style tiles |
| `sourcetiles.py` | the current style tiles: Bands, Marks and Strokes, one per source of shapes |
| `designlib_api.py` | the server routes for the style tiles: pages, variants, saving, the saved gallery |
| `designlib/` | `tiles/` (pages, `shapes.js`, `app.js`, assets) and `saved/` (saved specimens) |
| `colorways.py` | opponent colour pairs, gradients, duotones and mesh backgrounds for every feature |
| `colorways/` | one folder per feature set: per-feature JSON and card, plus `colorways.css`, `filters.svg`, `gallery.html`, `pairs.png` |
| `sample_resnet50.py`, `sample_convnext.py` | render a fixed random sample of channels spread across a model's block layers into `<model>features/` |
| `palettes/` | swatch strips with hex codes, plus `.json` colour lists |
