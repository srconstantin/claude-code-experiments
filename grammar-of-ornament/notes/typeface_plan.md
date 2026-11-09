# Plan: a display typeface built from the compound strokes

Goal: a striking display face for logos, titles and posters, not for body text.
It should carry the biomorphic character of the compound strokes (combs,
feathers, sprigs, branches, crosshatch meshes, bands and blobs) and still work
as type: legible, consistent and well spaced. It should also hold up as a
single word set very large, or very small as a logo.

## 1. Standards: what typographers look for in a display face

These are the criteria the font will be judged against. Each comes with a test
we can actually run.

**Recognizable skeletons.** Every letter has to read as its archetype at a
glance. Ornament can live on the strokes but must not change the structure
that identifies the letter. Example: an R needs its leg, and a G needs its
spur or bar.
- *Test:* a word set at 24 px must still read. An OCR or classifier pass on
  rendered glyphs should recognize every letter; any it confuses (O/Q, I/l/1,
  C/G, U/V) gets redrawn.

**Distinctiveness.** The danger zones for a decorative face are the letters
that differ by one small detail: O and Q, C, G and O, I, l and 1, U and V,
E and F, and in lowercase n and h.
- *Rule:* the distinguishing detail (the Q tail, the G spur, the l ascender)
  is the place to put *more* contrast, never less ornament.

**A shared family resemblance (the "DNA").** A font is a system, not 26
illustrations. Every glyph is built from the same small set of decisions:
- one stroke model (how a stem is dressed);
- one curve model (how a bowl is dressed);
- one terminal (how a stroke ends);
- one junction (what happens where strokes meet);
- one contrast axis (which strokes are heavy).

Each ornamental idea is used consistently everywhere it applies, or not at
all. This is the type equivalent of your "one weird shape per element" rule.
- *Test:* "Hamburgefonstiv" (the standard proofing word, which covers most of
  the alphabet's structures) and "HAMBURGEFONSTIV" should look like one hand
  drew them.

**Even colour and rhythm.** In a set word, each letter should carry about the
same amount of black ("colour"), and the white spaces inside letters
(counters) and between them should alternate evenly.
- Combs and feathers add black, so wide letters (M, W) need sparser teeth and
  narrow letters (I, J) need denser teeth.
- *Test:* measure the ink density of every glyph in its box (target within
  about ±12%). Blur a set word heavily; it should become an even grey bar with
  no dark clots or holes.

**Spacing and kerning.** Spacing does more for how professional a display face
looks than the drawing does.
- Sidebearings (the white set aside on each side of a glyph) are set by the
  optical method: start from the spacing of H, O and n/o, then adjust.
- Teeth sticking out of a letter's sides should be allowed to interlock with
  the next letter, not push it away. So spacing is measured on the letter's
  *core* (its unornamented skeleton), with the ornament allowed to overhang.
- Kerning pairs (individual spacing corrections between specific letters)
  then cover AV, LT, TY, To, Yo, LY and similar.
- *Test:* proof words such as HOHOHO, nnonnoo, LAVATORY, MINIMUM and
  TYPOGRAPHY.

**Optical corrections.** Geometry that is mathematically even looks wrong, so
type is drawn slightly off to look right:
- **Overshoot:** round and pointed letters (O, S, A, V) extend about 1.5–3%
  past the baseline and cap height, or they look short.
- **Crossbars:** horizontals are thinner than verticals at the same weight.
  The bar of H sits a touch above the middle, and the middle arm of E is
  shorter.
- **Ink traps and tapering at joins:** where two strokes meet, the black
  clots. Here that is the junctions, where combs and branches merge, so joins
  get thinned (or a deliberate small knot, used consistently).
- **Apexes** of A, V, W and M are trimmed or extended so they don't look
  blunt or spiky.

**Vertical metrics.** UPM (units per em, the font's coordinate grid) of 1000:
- cap height 700, baseline 0, x-height about 500 (if there is lowercase),
  ascender and descender at 750 and −220;
- ornament may break out of these zones only in specified places (terminals
  and tails) and by a fixed amount, so lines of type don't collide.

**Behaviour at scale.** A logo face lives at two extremes. At 400 px the
fine structure of the teeth has to look deliberate, crisp and not noisy. At
24–32 px the teeth merge into texture and the letter's *silhouette* has to
carry it.
- *Test:* every proof is rendered at both sizes.

**Clean outlines.**
- Contours are closed, with consistent direction, no stray overlaps, and
  points at the extremes (topmost, leftmost and so on).
- Point count is reasonable: an ornamented glyph will be heavy, but it should
  stay under about 2,000 points so the font renders fast.
- Hinting doesn't matter much for a display face.

**Personality with restraint.** The best decorative faces put the strangeness
in one or two places and keep the rest disciplined. Strong candidates here:
- the comb teeth along the outer edge of strokes;
- feather barbs on diagonals;
- a mesh (crosshatch) filling counters.

Not all of them at once.

## 2. What the compound strokes give us

From `compounds.json`, the atlas and `linework/*.strokes.json`:
- 28 compound types (about 31,700 instances), each with typical
  measurements: length, number of teeth, spacing, how much spacing varies,
  straightness, orientation and spread;
- trees with a spine and side branches (combs, feathers, sprigs, branching
  lines), with per-stroke width profiles that taper and are never constant;
- bundles (hatching, fine hatching, crosshatch meshes), bands, and blobs (mass
  nodes at junctions);
- siblings: near-identical variants, which become *alternate glyphs*.

The font takes these **measured statistics** (tooth spacing and how much it
varies, the ratio of tooth length to spine width, the taper curve, how
branches angle off the spine, the rhythm of hatching) and builds them onto
letter skeletons. It does not paste traced pieces onto letters. This keeps the
character of the strokes while giving full control over legibility, which
pasting never would.

## 3. Construction

**3.1 Skeletons (parametric, like Metafont).** Every glyph is a set of
centreline strokes (lines and Bézier curves) with a role on each segment:
stem, bowl, arm, crossbar, diagonal, tail or terminal. Junctions are named
nodes. The skeletons live in a data file (`typeface/skeletons.json`); I
draft them on the 1000-unit grid and refine them by eye on proofs. Shared
parts are drawn once and reused: the bowl of b, d, p and q, the arch of n, h
and m, the round of O, C, G and Q.

**3.2 Dressing rules: compound type for each stroke role.** One table per
direction (see 4):
- **Stem:** spine at the stem weight, plus teeth from comb or feather
  statistics, on the *outer* side only, so counters stay open.
- **Bowl:** a curved comb, with teeth pointing outward and spacing set by
  arc length.
- **Crossbar:** thinner, bare, or a short hatching bundle.
- **Terminal:** a sprig, loop or tapered end from the strokelib width
  profiles.
- **Junction:** a mass node (small blob) or a thinned ink trap. One rule for
  the whole font.

**3.3 Drawing.** For each glyph:
- sweep the width profile along each skeleton stroke, and place the teeth
  using that compound's measured spacing, with seeded jitter for the measured
  variation;
- combine everything by raster union at about 4096 px per em, then trace back
  to Bézier curves (the same contour pipeline as the motifs, plus curve
  fitting).

Raster-then-trace is robust for organic shapes. If the curves come out lumpy,
I'll switch to exact geometric union with a vector library.

**3.4 Imperfect repetition as an OpenType feature.**
- Each letter gets 3–5 alternates, generated with different seeds (siblings).
- A contextual-alternates feature (`calt`) cycles through them, so a doubled
  letter (the two O's in BOOK) never shows identical copies. This is how
  Beowolf and similar "living" fonts work, and it is the frieze idea brought
  to type.
- A stylistic set (`ss01`) turns it off, for logos that need fixed shapes.

**3.5 Font file.** Built with `fontTools` (`pip install fonttools` into the
venv; it isn't installed yet). The output is OTF plus WOFF2 for the web.

**3.6 Optional variable axis, "growth".** It runs from 0, the bare skeleton at
display weight, to 100, full teeth, feathers and mesh. It gives one
font a calm setting and a wild setting, and makes for a striking animated
logo.
- Requires every master to have the same number and order of points, which
  raster tracing doesn't produce. So it waits until the drawing is exact
  geometry.
- Static instances at 0, 50 and 100 are the fallback.

## 4. Three directions to prototype before committing

Each is built only for the proof word **HAMBURGEFONSTIV** plus O, Q, R, S, a
few digits and two set words.

- **A. Comb Grotesk.** Monoline sans skeleton (even, geometric). Every outer
  edge carries short, densely spaced comb teeth that taper out. Junctions are
  small mass knots. Calm structure, bristly silhouette; reads like an
  architectural creature.
- **B. Feather Didone.** High contrast in the Didot or Bodoni style, so it
  can use thick and thin roles:
  - heavy stems become feathers, a spine with barbs angled 30–45° along both
    sides;
  - hairlines and crossbars become single hatching strokes;
  - ball terminals become sprigs or loops.

  The most elegant of the three, and close to the S10 and Haeckel poster
  references.
- **C. Mesh Stencil.** Letters built as filled masses (blob and band shapes)
  with crosshatch-mesh counters and branch-like bridges. Stencil-style gaps
  appear where branches fail to meet. The most "cellular"; strongest at very
  large sizes; riskiest for legibility.

You pick one (or a hybrid) from proof sheets. Then the full character set.

## 5. Scope and phases

Each phase ends with a proof sheet (HTML plus PNG) for review.

1. **Skeletons:** caps A–Z, digits 0–9, and the punctuation `. , : ; ! ? - – & ' "`
   as bare skeletons. Proof: spacing (HOHOHO, nnonnoo) and the ink-density
   chart.
2. **Three directions** on the proof word (section 4). You choose.
3. **Full caps and digits** in the chosen direction, with optical corrections,
   spacing and about 150 kerning pairs. Proofs: specimen, set words, 24 px and
   400 px, blur test, recognition test.
4. **Alternates and OpenType:** 3–5 alternates per glyph, `calt` cycling and
   `ss01`. Build OTF and WOFF2. Proof: a web specimen page using the real font,
   with repeated-letter words.
5. **Lowercase** (optional). Display faces for logos are often caps-only or
   caps-and-small-caps; decide after phase 3.
6. **Applied tests:** logos (wordmarks of 3–8 letters), a poster title, the
   designlib website headings.
7. **Variable "growth" axis** (optional; section 3.6).

Rough effort: phases 1–2 are a session; phase 3 is the bulk of the work
(skeleton tuning and spacing are iterative); phase 4 is a session.

## 6. Decisions for Sarah

- Caps only first, or caps and lowercase eventually?
- Which direction appeals in principle, or all three to be compared? (The
  plan assumes all three.)
- Contextual alternates on by default (living, imperfect) or off by default
  (fixed for logos)?
- Name.

## 7. Lessons from the typefaces in the reference albums (2026-10-22)

The display faces in the albums that work all share three things the first
Neuron prototypes lacked:

- **Smooth, confident outlines.** Even the strangest (the blob letters in `1-047`
  and `1-087`, the swirl in `1-105`, UNSEEN FUTURES in `3-060`) have clean curves.
  The strangeness is in the *construction*, never in a rough edge or a texture.
- **Ornament concentrated where strokes start, end or meet.** In Fantasy
  America (`1-007`) the strokes are calligraphic and smooth; the wildness is all
  in the star-spikes at terminals and the long swashes. UNSEEN FUTURES puts its
  character in notches cut into heavy bodies. Glacier, Desert and Wasteland in
  `1-021` each change one structural habit (spiky joins, split stems,
  swash terminals) and apply it everywhere.
- **One unit, repeated along the skeleton.** "New Type" (`2-049`) is letters
  built entirely by chaining one small motif along each stroke. `3-071`
  builds letters from pipe segments, and the "dreams" type (`3-041`) from beads.
  Repetition of a single unit reads as a system, not as texture.

Also: BAM (`2-073`) sets one biomorphic letter among plain ones. That is a
good use of alternates for logos.

**What this changes for Neuron.**
1. Strokes stay smooth and continuous. Compound pieces supply *terminals and
   joins* (sprigs, loops, forks and comb heads as the ends of strokes, the
   way Fantasy America uses spikes), not texture along every stroke.
2. A **chain** direction: each stroke built from one small compound unit
   (a loop, a blob or a sprig) repeated along the skeleton with its sibling
   variants, as in New Type.
3. A **notch** direction: heavy, smooth bodies with piece shapes cut *out*
   (negative space) at joins and terminals, as in UNSEEN FUTURES.
4. Alternates as **one wild letter per word**, for logos.

## 8. The typography album (https://photos.app.goo.gl/dRCTo2bFjgmSmTLNA, 93 images)

Almost all of it is **high-contrast display serifs with liquid, organic
modulation**: MOLD (`4-052`), Migra (`4-020`), HEAVEN COME (`4-034`, `4-081`),
Blandglos/Endorfina (`4-068`), frivole (`4-079`), NYMPH (`4-017`), the
Didones of `4-018`, `4-057`, `4-062`, and AGE OF (`4-082`). Also a few liquid
calligraphic faces (`4-038`, `4-039`, `4-040`, `4-051`, `4-080`).

What they have in common:
- **The strangeness lives in the thick/thin modulation.** Thick strokes pinch
  to hairlines where they meet other strokes (ink traps as a style: MOLD,
  Blandglos), swell unexpectedly, or split. The *contour* is perfectly
  smooth; the shape is the idea.
- **Hairlines are very thin**, so the heavy strokes read as sculpted masses.
- **Terminals are the signature:** teardrops, spurs, hooks, flicks and split
  ends.
- **Ornament is never applied as texture.** No bristles, no grunge.

**What this means for Neuron:** use the compound pieces for *shape*, not
for edge texture.
- Each thick stroke's width profile (how it swells and pinches along its
  length) is taken from a compound band or blob, smoothed.
- Joins pinch into ink traps, the way the networks' bands narrow at
  junctions.
- Terminals are drawn from sprigs (a curling flick), forks (a split end) and
  loops (a teardrop).
- All outlines are smooth vectors.

This is the "Neuron Serif" direction (`typeface/serif.py`).
