# Neuron library

Fonts, colours and ornaments for the web, all taken from the drawings neural networks (VGG16, ResNet50, ConvNeXt)
make when asked to show what one of their channels responds to.

- **Neuron Leaf**: a display typeface, drawn with a leaf-shaped pen on uncial skeletons.
- **Colorways**: 100 palettes, each the opponent colour pair of one ConvNeXt feature, with a paper, an ink, two accents, a dark side and a mesh gradient.
- **Elements**: dividers, lines, friezes, corners and frames, drawn boxes, marks, fleurons, medallions, icons and emblems. Each is drawn from the real shapes in the library: compound strokes, the single strokes they are built from, and hand-cut motifs.

Every element is a single-colour SVG filled with `currentColor`, so any colour or colorway restyles it.

## What's in the folder

```
neuron-library/
  neuron.css            fonts, colorway classes and element classes; link this
  colorways.css         the raw colorway variables (--cw-<id>-a, -b, -shadow, -highlight)
  neuron-elements.js    optional: web components that fit ornaments to any size
  fonts/                Neuron-Leaf (.woff2, .otf); Alegreya Sans, Alegreya Sans SC, DM Mono (.woff2)
  elements/
    dividers/           rule-*, mirror-*: horizontal rules made of compound strokes
    lines/              line-k-t/-b/-l/-r: one single stroke stretched to a line, in four orientations
    tiles/              strip-*, dash-*: small seamless tiles that repeat sideways
    friezes/            f000–f199: whole bands, 1600 × 140
    corners/            corner-*-tl/-tr/-bl/-br: one corner in four turns
    frames/             frame-*: a sample frame (four corners and four stroke edges) as one SVG
    marks/              mark-*: bullets and section marks from compact compounds
    fleurons/           fleuron-*: one piece turned round a centre
    medallions/         medallion-*: a whole motif turned round a centre
    icons/              menu, close, plus, minus, arrows, chevrons, check, search, drawn in single strokes
    emblems/            the motifs: <id>.svg silhouette, <id>-filled.webp (the print's own colours),
                        <id>-lum.webp (a luminance mask for duotones)
    strokes/            all 408 compound strokes, cleaned and recolourable
    sprite.svg          icons, marks and fleurons as <symbol>s
  README.md
```

## Setup

Copy `neuron-library/` into your site and link the stylesheet:

```html
<link rel="stylesheet" href="neuron-library/neuron.css">
<!-- optional, for the adaptive components -->
<script src="neuron-library/neuron-elements.js" defer></script>
```

**Serve the site over http(s).** The element classes use CSS masks, and browsers won't load masks or `<use>` sprites
for a page opened straight from disk (`file://`). For local work:

```
python3 -m http.server -d path/to/site
```

## Type

```html
<h1 class="n-display">Grammar of Ornament</h1>
<body class="n-text"> … </body>
```

Neuron Leaf is a display face. Use it for titles, numbers and short headings at 32px and up. It covers A–Z, a–z, 0–9
and common punctuation (`. , : ; ! ? ' " - – — / ( ) &`).

Body text is set in **Alegreya Sans** (`--n-text`, or class `n-text`), a humanist sans serif built on the same
broad-pen logic as Neuron Leaf's tapering strokes. It has a small x-height, so set it a little larger than you
would another sans (17–18px for reading text). Regular, italic, medium and bold are bundled.

Two more faces cover the small print:
- **Alegreya Sans SC** (`--n-label`, or class `n-label`) is for labels, navigation and metadata. Set it in lowercase
  with a little letterspacing (`n-label` does both) for an even line of small caps.
- **DM Mono** (`--n-mono`) is only for things that are literally code.

All three are under the SIL Open Font License (see `fonts/OFL-*.txt`).

## Colorways

Put `n-cw-<id>` on any element. It sets these variables for everything inside it:

| variable | use |
|---|---|
| `--n-paper` | page background (light) |
| `--n-ink` | text: at least 7:1 against paper |
| `--n-accent`, `--n-accent2` | the feature's opponent colour pair, adjusted to at least 3:1 against paper; use for ornament and large type |
| `--n-dark`, `--n-on-dark` | a dark ground and its text colour |
| `--n-mesh` | the colorway's mesh gradient (a background image) |
| `--n-ramp` | the feature's colours left to right, as a gradient |
| `--n-shadow`, `--n-highlight` | the feature's darkest and lightest tones (the duotone ends) |
| `--n-dark-accent`, `--n-dark-accent2` | the accents lifted to read on the dark ground |

```html
<body class="n-cw-f12-c108 n-page">
  <header class="n-mesh"> … </header>
  <h1 class="n-display n-accent">Title</h1>
</body>
```

**The dark side.** `n-dark` on any container swaps the whole palette: paper becomes the dark ground, ink its light
text colour, and the accents their lifted versions. The features are luminous things, and they look most like
themselves on dark; the dark side is a full theme, not a footer colour.

```html
<section class="n-cw-f10-c100 n-dark"> … everything inside is on the dark palette … </section>
```

**Light and fringe.** Two treatments taken from how the features look:
- `n-glow` gives type a halo in the colorway's highlight; `n-glow-box` does the same for an emblem or image. Made for dark grounds.
- `n-fringe` splits an edge into the opponent pair, the colour fringing the features show at every boundary. Subtle at text size, strong at display size. `n-fringe-box` does it to a box.
- `n-ramp`, `n-ramp-line`, `n-ramp-under`: the feature's colour ramp as a background, a rule or an underline.

The `<id>` is the colorway's short name, such as `f10-c100` (ConvNeXt `features.1.0`, channel 100). All of them are on the
Colorways tab of the site.

## Elements as CSS classes

Each element class paints `currentColor` through the element's shape, so set `color` to recolour it.

```html
<hr class="n-divider-rule-straight-band-13" style="color:var(--n-accent)">
<hr class="n-line-3" style="--n-h:3px">                                   <!-- one stroke, stretched to the width -->
<div class="n-repeat-strip-wavy-band-00" style="--n-h:40px"></div>   <!-- repeats to any length -->
<div class="n-frieze-f017" style="color:var(--n-accent)"></div>       <!-- a whole band, scaled to its width -->
<span class="n-mark-curved-comb-05"></span>  Bullet text
<span class="n-icon-arrow-right"></span>
<span class="n-fleuron-fork-04" style="--n-size:96px"></span>
<div class="n-medallion-f21-c52-1743" style="width:420px"></div>         <!-- a motif turned round a centre -->
<div class="n-emblem-f17-c10-1974" style="width:420px"></div>            <!-- silhouette, in currentColor -->
<div class="n-emblem-f17-c10-1974-filled" style="width:420px"></div>     <!-- the print's own colours inside the cut -->
<div class="n-emblem-f17-c10-1974-duo" style="width:420px"></div>        <!-- duotone: --n-shadow to --n-highlight -->
<div class="n-emblem-f17-c10-1974-duo" style="--n-duo-dark:var(--n-ink);--n-duo-light:var(--n-paper)"></div>

<div class="n-box-3" style="background:var(--n-accent);color:var(--n-paper)">A panel with a hand-drawn edge</div>
<div class="n-drawn-3" style="color:var(--n-accent);--n-line:3px"><div>A drawn border round the content</div></div>

<section class="n-frame-loop-00" style="--n-size:72px">
  Content inside a frame of four corners, with one stroke along each side.
</section>
```

**Lines** (`n-line-1` … `n-line-8`) are the library's answer to a CSS border: a single stroke stretched to the length
of the rule, its taper kept, so no line on the page is dead straight. `n-line-k` runs across, with the heavy end at
the left; `n-line-v-k` runs up a side (set `--n-len` for its length). Frames use the same strokes for their sides.

A frieze with two inks takes its second colour from `--n-frieze2` when the SVG is inlined in the page. As a CSS class
or an `<img>` it's drawn in one colour.

Emblems come in three forms. The silhouette is one path in `currentColor`. The filled form shows the interior of
the motif's own print, the feature's colours inside the cut, upscaled so that it holds at hero size, and is the one
that looks most like the network's work. The duotone paints that interior between two colours, the colorway's shadow
and highlight by default, or any two set as `--n-duo-dark` and `--n-duo-light`.

Medallions turn a whole motif round a centre, its long axis on the radius and its heavier end inward, four- to
eight-fold. Like emblems they are for large spots (300px and up), never small UI.

Drawn boxes (`n-box-1` … `n-box-8`) are panels that are nearly rectangles: each edge wavers the way a single stroke
does, and the corners sit a little off true, like a box drawn by hand. `n-drawn-k` draws the same edge as a border:
the outer element takes the line colour, so wrap the content in one child, and set `--n-line` for the line's weight.
Both work at any size and proportion. `<neuron-box>` (below) draws real tapered strokes instead.

Sizes are set with custom properties:
- `--n-h`: the height of dividers and repeats, and the weight of a line. A frieze band's height follows its width (1600 × 140).
- `--n-len`: the length of a vertical line.
- `--n-size`: the size of marks, icons and fleurons, and the corner size of frames.
- `--n-gap`: the space above and below a divider or line.
- `--n-pad`: the inner padding of frames and boxes.
- `--n-line`: the line weight of a drawn border.

Icons, marks and fleurons are also in the sprite:

```html
<svg width="24" height="24"><use href="neuron-library/elements/sprite.svg#n-icon-menu"/></svg>
```

You can also use any SVG directly: `<img>` it (it will be black), inline it, or open it in a design tool.

## Adaptive elements (neuron-elements.js)

The static files have a fixed shape. The web components draw ornaments to fit the space they're given:
- Dividers and friezes fit a whole number of pieces to any width.
- Frames put their corners at the box's real corners, however its proportions change, and stretch one stroke along each side between them.
- Fleurons are built from any piece, in any number of folds.
- Boxes are drawn with four real strokes, their ends crossing at the corners, at the box's real size.
- Medallions turn any motif round a centre, in any of its three forms, including the motif filled with its own print.

```html
<neuron-divider piece="straight-band-13" mode="repeat" height="24"></neuron-divider>
<neuron-divider piece="wavy-branch-12" mode="mirror" centre="curved-comb-05" height="32"></neuron-divider>
<neuron-frieze piece="wavy-band-00" height="44"></neuron-frieze>
<neuron-frame piece="loop-00" edge="f07-short-wave-branch-fine-15" size="64"><p>Any content</p></neuron-frame>
<neuron-fleuron piece="fork-04" fold="8" mirror size="120"></neuron-fleuron>
<neuron-box weight="3" style="color:var(--n-accent)"><p>Four tapered strokes drawn to the box's real size.</p></neuron-box>
<neuron-medallion motif="f24-c27-3655" form="filled" fold="6" size="480"></neuron-medallion>
<neuron-palette colorway="f30-c85"> … </neuron-palette>
```

To list the pieces, run `NeuronElements.pieces()` in the browser console (`NeuronElements.motifs()` lists the
motifs). Each piece's roles show what it was chosen for:
- rules: long pieces
- compact: centres
- curved: frame corners
- petals: fleurons
- tiles: friezes
- lines: the sides of frames and boxes

Any piece works in any component. Everything is drawn in `currentColor`.

## Guidelines

- **One texture per piece.** Don't layer one feature's image behind another's, or mix two motif families in one composition.
  Use a single source for the ornament, and paper, flat colour, type and plain rules for the rest.
- **Ornament at the edges.** Dividers, friezes and frames belong around text, never under it. Text always sits on a
  solid ground.
- **Motifs and features large.** Emblems, medallions and feature images are for heroes, covers and big panels. In small UI they
  turn to noise. Marks and icons are for small sizes.
- **Surfaces** take a flat colorway colour, the colorway's mesh gradient, or a feature image. Don't use patterned fills.
- **No dead straight lines.** Where a page wants a rule or a border, use a line, a divider or a drawn box, so the
  edges keep the hand of the strokes.
