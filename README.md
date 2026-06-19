# Tomographic Reconstruction with Sensitivity-Based Uncertainty Quantification

An interactive web app for CT-style image reconstruction that also estimates **how trustworthy
each pixel of the result is**. You set up a virtual scan in the sidebar, press **Run**, watch the
solver work in a live log, then explore the reconstruction — alongside a per-pixel uncertainty
map — right in your browser.

It's a self-contained, single-purpose demo for exploring how scan and reconstruction settings
shape both the *quality* and the *reliability* of a reconstructed image. It is not a
general-purpose CT library: it runs one fixed pipeline on a built-in synthetic test image.

## What it does

Every run carries out the same four-stage pipeline, end to end, on a standard synthetic test
image (the Shepp-Logan phantom):

1. **Simulate a scan.** Starting from the known phantom, the app generates synthetic CT
   measurements — a *sinogram* — as if a scanner had imaged it from several angles.
2. **Reconstruct the image.** Working only from those measurements, it rebuilds the image,
   balancing faithfulness to the measurements against a preference for smooth, clean results.
3. **Compare with classical methods.** For reference, it reconstructs the same image two
   well-known conventional ways — filtered back-projection (FBP) and SART.
4. **Quantify the uncertainty.** Finally, it measures how sensitive each reconstructed pixel is
   to the measurements, turning that into a per-pixel uncertainty map and a single overall
   reliability score.

Steps 1 and 2 are optimization solves; the uncertainty in step 4 is read directly from the
solved reconstruction.

## What you get

When a run finishes, the app lays out **six figures** in a two-row, three-column grid:

- the **original phantom** — the ground-truth image
- the app's **reconstruction**
- a **per-pixel uncertainty map** — where the reconstruction is more or less reliable (log scale)
- the **simulated sinogram** — the raw measurements, merged across angles
- the **FBP** reconstruction — classical baseline
- the **SART** reconstruction — classical baseline

Alongside the figures you also get:

- a single **D-optimality score** that summarizes overall reconstruction uncertainty in one
  number — smaller means tighter, more reliable — handy for comparing one set of settings
  against another; and
- **run details**: solver status plus counts such as the number of reconstructed pixels and
  measurements used.

A live, scrolling solver log streams the whole time a run is working.

## Using it

1. Set the parameters in the sidebar.
2. Click **Run**.
3. Follow the solver log as it streams.
4. Review the six figures and the D-optimality score — then tweak and run again to compare.

Results stay on screen while you adjust other controls; the app only re-solves when you press
**Run** again.

## Parameters you can tune

The sidebar shapes the virtual scan and the reconstruction:

- **Image resolution** — pixels per side of the image grid (default 30×30). Larger is slower.
- **Projection angles** — how many directions the virtual scanner images from (default 9) and
  the angle range they span (0–180°).
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
