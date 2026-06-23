# Tomographic Reconstruction with Sensitivity-Based Uncertainty Quantification

An interactive web app for CT-style image reconstruction that also estimates **how trustworthy
each pixel of the result is**. You **design the scan geometry** in the sidebar — placing each
X-ray projection where you want it — press **Run**, watch the solver work in a live log, then
explore the reconstruction alongside a per-pixel uncertainty map, right in your browser.

It's a self-contained, single-purpose demo for exploring how scan and reconstruction settings
shape both the *quality* and the *reliability* of a reconstructed image. It is not a
general-purpose CT library: it runs one fixed pipeline on a built-in synthetic test image.

## What it does

You first define the **projection geometry** — one row per beam step, each with its own angle,
radial offset, and number of beams. Then every run carries out the same pipeline, end to end, on
a standard synthetic test image (the Shepp-Logan phantom):

1. **Simulate a scan.** Using the geometry you defined, the app generates synthetic CT
   measurements — a *sinogram* — as if a scanner had imaged the phantom along those beams.
2. **Reconstruct the image.** Working only from those measurements, it rebuilds the image,
   balancing faithfulness to the measurements against a preference for smooth, clean results.
3. **Quantify the uncertainty.** Finally, it measures how sensitive each reconstructed pixel is
   to the measurements, turning that into a per-pixel uncertainty map and a single overall
   reliability score.

Steps 1 and 2 are optimization solves; the uncertainty in step 3 is read directly from the
solved reconstruction. Because the geometry is yours to set, you can explore how beam placement
changes both the reconstruction and its reliability.

## What you get

When a run finishes, the app lays out **four figures**:

- the **original phantom** — the ground-truth image
- the app's **reconstruction**
- a **per-pixel uncertainty map** — where the reconstruction is more or less reliable (log scale)
- a **beam / measurement view** — the projection geometry you defined, drawn over the phantom

Alongside the figures you also get:

- a single **D-optimality score** that summarizes overall reconstruction uncertainty in one
  number — smaller means tighter, more reliable — handy for comparing one set of settings
  against another; and
- **run details**: solver status plus counts such as the number of reconstructed pixels and
  measurements used.

A live, scrolling solver log streams the whole time a run is working.

## Using it

1. Define the beam geometry (and other parameters) in the sidebar; a live preview shows where
   the beams point as you edit.
2. Click **Run**.
3. Follow the solver log as it streams.
4. Review the four figures and the D-optimality score — then tweak and run again to compare.

Results stay on screen while you adjust other controls; the app only re-solves when you press
**Run** again.

## Parameters you can tune

The sidebar shapes the virtual scan and the reconstruction:

- **Image resolution** — pixels per side of the image grid (default 30×30). Larger is slower.
- **Projection geometry** — an editable table where each row is one beam step: its **angle**, a
  radial **offset** (where the ray bundle is centered), and the **number of beams**. A live dial
  previews where the beams point, and a seed button fills in evenly-spaced angles to start from
  (the default is 9 evenly-spaced full-fan projections over 0–180°).
- **Beam-degradation settings** — model how the beam weakens as it passes through the object.
- **Smoothness weight** — how strongly the reconstruction favors a clean image over exactly
  matching the measurements.
- **Measurement-noise level** — the assumed uncertainty in the measurements, which feeds the
  uncertainty estimate.
- **Solver settings** — the optimizer's iteration limit and which underlying numerical solver to
  use.

## Good to know

- A full run at the default size takes **a few minutes** — two optimization solves plus a
  sensitivity analysis. Keep the browser tab open while it works.
- Everything runs on a built-in synthetic phantom; there is no uploading of real scans or images.
- This is a focused demonstration of sensitivity-based uncertainty quantification for
  tomographic reconstruction, built around one fixed pipeline.
