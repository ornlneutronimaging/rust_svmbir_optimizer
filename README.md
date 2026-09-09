# SVMBIR Optimizer

Standalone GUI to tune SVMBIR reconstruction parameters on a pre-processed
CT checkpoint (the HDF5 written by `rust_ct_reconstruction`: attenuation
data with `/angles_rad` and `/center_of_rotation`). Modeled on the
`marimo_optimize_svmbir_parameters` notebook.

## Workflow

1. Open a checkpoint (command-line argument or the 📂 button).
2. Pick two slices on the projection view (red and cyan lines).
3. Adjust the parameters — **sharpness** in the open section; SNR,
   iterations, resolutions, positivity and the center offset behind the
   password-locked **Advanced** section (same password as the notebook).
4. Choose the **Test data** resolution: the checkpoint's own (the default),
   or the projections n×n block-averaged first (2x2, 3x3, 4x4, 6x6 — the
   same block mean as the pre-processing rebin step). svmbir runs on the
   CPU and is slow on wide data; smaller sinograms reconstruct much faster,
   at a coarser resolution. This is a speed shortcut for the test only: the
   parameters are saved in the checkpoint's pixel units and the full
   reconstruction runs on the checkpoint as is (only the center offset is
   converted for the rebinned test run, following the pixel grid like the
   pre-processing rebin does).
5. **▶ Evaluate** reconstructs a 4-slice band around each line through the
   real `svmbir` (0.4.0, from the `all_ct_reconstruction_development` pixi
   environment) and shows the two middle slices side by side. Repeat until
   satisfied — every run lands in the **Run history**, whose `use` buttons
   restore the parameters (and test resolution) of a previous run.
6. **💾 Save** writes `svmbir_config` (JSON) into the checkpoint's
   `/metadata`; `rust_ct_reconstruction` restores it automatically when the
   file is loaded, and later SVMBIR reconstructions use these parameters.

## Center of rotation and tilt

The projection view draws the **center of rotation**: the checkpoint's
value as a dashed orange line and, once the center offset is changed in
the Advanced section, the current one as a solid green line, with a
readout of both values and the move between them (`↺ checkpoint value`
goes back). Drag the offset field or use the arrow keys; holding Shift
moves it 10× faster. The center offset defaults to
`-(width/2 - center_of_rotation)` from the checkpoint, matching the
notebook's convention.

Under the projection, **Tilt correction** lists what the checkpoint
records: the pre-processing step of `rust_ct_reconstruction`
(`tilt_correction`) and every run of the standalone tool
(`metadata/tilt_center_of_rotation`, JSON records), plus the
pre-processing rebin when there was one.

**🎯 Open the tilt & center-of-rotation tool** launches
`rust_tilt_center_of_rotation` on the checkpoint itself (the more robust
estimators: sub-pixel 0°/180° registration, all-pairs consensus, gridrec
test slices). Applying & saving there rewrites the checkpoint's
projections and center of rotation; when the tool closes, this window
detects the new correction record, reloads the file and re-seeds the
center offset from the new center — save the parameters to keep it.
Closing the tool without applying changes nothing.

## Running

```bash
./launch_svmbir_optimizer.sh [checkpoint.h5]
```

Requires a graphical session; the launch script rebuilds when sources
changed. `--called-from-app` additionally prints the saved JSON on stdout
for a driving application.
