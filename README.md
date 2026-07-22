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
4. **▶ Evaluate** reconstructs a 4-slice band around each line through the
   real `svmbir` (0.4.0, from the `all_ct_reconstruction_development` pixi
   environment) and shows the two middle slices side by side. Repeat until
   satisfied — every run lands in the **Run history**, whose `use` buttons
   restore the parameters of a previous run.
5. **💾 Save** writes `svmbir_config` (JSON) into the checkpoint's
   `/metadata`; `rust_ct_reconstruction` restores it automatically when the
   file is loaded, and later SVMBIR reconstructions use these parameters.

The center offset defaults to `-(width/2 - center_of_rotation)` from the
checkpoint, matching the notebook's convention.

## Running

```bash
./launch_svmbir_optimizer.sh [checkpoint.h5]
```

Requires a graphical session; the launch script rebuilds when sources
changed. `--called-from-app` additionally prints the saved JSON on stdout
for a driving application.
