//! SVMBIR parameters, the test reconstruction on two slice bands (through
//! the real svmbir of the `all_ct_reconstruction_development` pixi
//! environment), and saving the parameters back into the checkpoint HDF5.

use ct_reconstruction::combine::{LoadedStack, Projection};
use ct_reconstruction::crop::{read_npy, write_npy};
use ct_reconstruction::rebin::{rebin_center, rebin_projection, rebinned_size};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// The interpreter of the pixi environment that has svmbir installed.
pub const SVMBIR_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/.pixi/envs/default/bin/python";

/// svmbir's on-disk system-matrix cache locations, first writable one wins
/// (same list as the Python configuration).
pub const SVMBIR_LIB_PATHS: [&str; 3] = ["/fastdata/", "/SNS/VENUS/shared/fastdata/", "/tmp/"];

/// Slices reconstructed around each selected line (the Python
/// `MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH`); the middle one is shown.
pub const BAND: usize = 4;

/// The n×n rebin factors the test reconstruction can run on (1 = the
/// checkpoint's own resolution).
pub const TEST_REBIN_FACTORS: [usize; 5] = [1, 2, 3, 4, 6];

/// The center of rotation (px) a center offset stands for on
/// `width`-pixel-wide projections — the inverse of the seeding rule
/// `offset = -(width/2 - cor)` (integer half width, like the notebook).
pub fn center_from_offset(offset: f64, width: usize) -> f64 {
    (width / 2) as f64 + offset
}

/// The center offset that puts the center of rotation at `cor` on
/// `width`-pixel-wide projections.
pub fn offset_from_center(cor: f64, width: usize) -> f64 {
    cor - (width / 2) as f64
}

/// The center offset for the n×n rebinned test data, given the offset on
/// the checkpoint's `width`-pixel-wide projections: the center of rotation
/// follows the pixel grid ([`rebin_center`]) and is re-expressed against
/// the rebinned half width.
pub fn rebinned_offset(offset: f64, width: usize, n: usize) -> f64 {
    if n <= 1 {
        return offset;
    }
    let cor = rebin_center(center_from_offset(offset, width), n);
    let (rw, _) = rebinned_size(width, 1, n);
    offset_from_center(cor, rw)
}

/// Largest n×n test-rebin factor that is useful for a stack of `height`
/// rows: the band must still hold [`BAND`] rebinned slices.
pub fn max_test_rebin(height: usize) -> usize {
    TEST_REBIN_FACTORS
        .iter()
        .copied()
        .filter(|n| BAND * n <= height.max(1))
        .max()
        .unwrap_or(1)
}

/// The SVMBIR parameters exposed by the marimo notebook, with its defaults
/// and ranges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvmbirParams {
    /// General parameter (-1 to 3): larger = sharper.
    pub sharpness: f64,
    /// Assumed signal-to-noise ratio in dB (25 to 40).
    pub snr_db: f64,
    /// Force reconstructed voxels >= 0.
    pub positivity: bool,
    /// Iterations of the solver (10 to 100).
    pub max_iterations: i64,
    /// Multi-resolution levels (0 to 4).
    pub max_resolutions: i64,
    /// Center of rotation as a pixel offset from the detector center.
    pub center_offset: f64,
}

impl Default for SvmbirParams {
    fn default() -> Self {
        Self {
            sharpness: 0.0,
            snr_db: 30.0,
            positivity: true,
            max_iterations: 20,
            max_resolutions: 3,
            center_offset: 0.0,
        }
    }
}

impl SvmbirParams {
    /// Defaults seeded from the stack: the saved `svmbir_config` when the
    /// checkpoint carries one, otherwise the center offset derived from the
    /// stack's center of rotation (`-(width/2 - cor)`, like the notebook).
    pub fn from_stack(stack: &LoadedStack) -> Self {
        if let Some((_, json)) = stack
            .metadata
            .iter()
            .find(|(name, _)| name == "svmbir_config")
            && let Some(params) = Self::from_json(json)
        {
            return params;
        }
        let mut params = Self::default();
        if let (Some(cor), Some(first)) = (stack.center_of_rotation, stack.sample.first()) {
            params.center_offset = -((first.width / 2) as f64 - cor);
        }
        params
    }

    pub fn to_json(&self) -> String {
        serde_json::json!({
            "sharpness": self.sharpness,
            "snr_db": self.snr_db,
            "positivity": self.positivity,
            "max_iterations": self.max_iterations,
            "max_resolutions": self.max_resolutions,
            "center_offset": self.center_offset,
        })
        .to_string()
    }

    pub fn from_json(text: &str) -> Option<Self> {
        let doc: serde_json::Value = serde_json::from_str(text).ok()?;
        let mut params = Self::default();
        if let Some(v) = doc.get("sharpness").and_then(|v| v.as_f64()) {
            params.sharpness = v;
        }
        if let Some(v) = doc.get("snr_db").and_then(|v| v.as_f64()) {
            params.snr_db = v;
        }
        if let Some(v) = doc.get("positivity").and_then(|v| v.as_bool()) {
            params.positivity = v;
        }
        if let Some(v) = doc.get("max_iterations").and_then(|v| v.as_i64()) {
            params.max_iterations = v;
        }
        if let Some(v) = doc.get("max_resolutions").and_then(|v| v.as_i64()) {
            params.max_resolutions = v;
        }
        if let Some(v) = doc.get("center_offset").and_then(|v| v.as_f64()) {
            params.center_offset = v;
        }
        Some(params)
    }

    pub fn describe(&self) -> String {
        format!(
            "sharpness {:.1}, snr {:.0} dB, {} iter, {} res, offset {:.2}{}",
            self.sharpness,
            self.snr_db,
            self.max_iterations,
            self.max_resolutions,
            self.center_offset,
            if self.positivity { ", positivity" } else { "" }
        )
    }
}

const SVMBIR_SCRIPT: &str = r#"
import json
import sys

import numpy as np
import svmbir

sino_file, spec_file, out_file = sys.argv[1:4]
with open(spec_file) as f:
    spec = json.load(f)
sino = np.load(sino_file)  # (n_angles, total_band_slices, width)
angles = np.array(spec["angles_rad"], dtype=np.float32)
p = spec["params"]
slices = []
for a, b in spec["bands"]:
    s = np.ascontiguousarray(sino[:, a:b, :])
    w = s.shape[2]
    recon = svmbir.recon(
        sino=s,
        angles=angles,
        num_rows=w,
        num_cols=w,
        center_offset=p["center_offset"],
        max_resolutions=int(p["max_resolutions"]),
        sharpness=p["sharpness"],
        snr_db=p["snr_db"],
        positivity=bool(p["positivity"]),
        max_iterations=int(p["max_iterations"]),
        num_threads=int(spec["num_threads"]),
        verbose=0,
        svmbir_lib_path=spec.get("lib_path"),
    )
    slices.append(np.array(recon[recon.shape[0] // 2], dtype=np.float32))
np.save(out_file, np.stack(slices))
"#;

/// One test reconstruction of the two bands on a background thread;
/// resolves to the two reconstructed middle slices `(width, values0,
/// values1, seconds)`.
pub struct ReconJob {
    rx: Receiver<Result<(usize, Vec<f32>, Vec<f32>, f64), String>>,
}

impl ReconJob {
    pub fn start(
        stack: Arc<LoadedStack>,
        top_slice: usize,
        bottom_slice: usize,
        params: SvmbirParams,
        test_rebin: usize,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = run_recon(&stack, top_slice, bottom_slice, params, test_rebin)
                .map(|(w, top, bottom)| (w, top, bottom, started.elapsed().as_secs_f64()));
            let _ = tx.send(result);
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<(usize, Vec<f32>, Vec<f32>, f64), String>> {
        self.rx.try_recv().ok()
    }
}

fn scratch_dir(stack: &LoadedStack) -> Result<PathBuf, String> {
    let base = stack
        .path
        .parent()
        .filter(|p| p.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!(".svmbir_optimizer_{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_ok() {
        return Ok(dir);
    }
    let dir = std::env::temp_dir().join(format!("svmbir_optimizer_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// The rows of one projection feeding a test band around `line`: `BAND`
/// rows at the checkpoint's resolution, `BAND × rebin` rows when the band
/// is rebinned first (so it still holds `BAND` slices afterwards).
fn band_rows(line: usize, height: usize, rebin: usize) -> (usize, usize) {
    let rows = BAND * rebin.max(1);
    let start = line
        .saturating_sub(rows / 2)
        .min(height.saturating_sub(rows));
    (start, start + rows)
}

/// The two test bands of one projection, stacked (top band first), at the
/// test resolution: cut out of the full projection and, for a rebin factor
/// above 1, block-averaged n×n like the pre-processing rebin step.
fn extract_bands(p: &Projection, bands: [(usize, usize); 2], rebin: usize) -> Vec<f32> {
    let w = p.width;
    let mut out = Vec::with_capacity(2 * BAND * w / rebin.max(1));
    for (a, b) in bands {
        let rows = &p.mean[a * w..b * w];
        if rebin <= 1 {
            out.extend_from_slice(rows);
        } else {
            let band = Projection {
                name: String::new(),
                run_number: None,
                angle_deg: None,
                n_images_used: 1,
                height: b - a,
                width: w,
                mean: rows.to_vec(),
                total_counts: 0.0,
            };
            out.extend_from_slice(&rebin_projection(&band, rebin).mean);
        }
    }
    out
}

fn run_recon(
    stack: &LoadedStack,
    top_slice: usize,
    bottom_slice: usize,
    params: SvmbirParams,
    test_rebin: usize,
) -> Result<(usize, Vec<f32>, Vec<f32>), String> {
    let first = stack
        .sample
        .first()
        .ok_or("no projections in the stack")?;
    let (w, h, n) = (first.width, first.height, stack.sample.len());
    let angles: Vec<f64> = stack
        .sample
        .iter()
        .map(|p| p.angle_deg.map(|a| a.to_radians()))
        .collect::<Option<Vec<f64>>>()
        .ok_or("some projections carry no angle — the reconstruction needs all of them")?;

    // The test runs on n×n rebinned data when asked: smaller sinograms
    // reconstruct much faster on the CPU. The parameters keep the
    // checkpoint's pixel units; only the center offset is re-expressed for
    // the rebinned width.
    let rebin = test_rebin.clamp(1, max_test_rebin(h));
    let (rw, _) = rebinned_size(w, h, rebin);
    let mut test_params = params;
    test_params.center_offset = rebinned_offset(params.center_offset, w, rebin);

    // One BAND-slice band around each selected line; the middle slice is shown.
    let bands = [
        band_rows(top_slice, h, rebin),
        band_rows(bottom_slice, h, rebin),
    ];

    let dir = scratch_dir(stack)?;
    let sino_npy = dir.join("sino.npy");
    let spec_file = dir.join("spec.json");
    let out_npy = dir.join("recon.npy");
    let script = dir.join("svmbir_run.py");
    let cleanup = || {
        for f in [&sino_npy, &spec_file, &out_npy, &script] {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&dir);
    };
    let run = || -> Result<(usize, Vec<f32>, Vec<f32>), String> {
        // The two bands stacked into one (n, 2*BAND, rw) volume.
        let mut volume = Vec::with_capacity(n * 2 * BAND * rw);
        for p in &stack.sample {
            volume.extend_from_slice(&extract_bands(p, bands, rebin));
        }
        write_npy(&sino_npy, &[n, 2 * BAND, rw], volume.chunks(2 * BAND * rw))?;
        let lib_path = SVMBIR_LIB_PATHS
            .iter()
            .find(|p| {
                let path = Path::new(p);
                path.is_dir()
                    && std::fs::metadata(path)
                        .map(|m| !m.permissions().readonly())
                        .unwrap_or(false)
            })
            .copied();
        let spec = serde_json::json!({
            "angles_rad": angles,
            "bands": [[0, BAND], [BAND, 2 * BAND]],
            "num_threads": std::thread::available_parallelism().map(|n| n.get().min(60)).unwrap_or(8),
            "lib_path": lib_path,
            "params": serde_json::from_str::<serde_json::Value>(&test_params.to_json()).expect("params json"),
        });
        std::fs::write(&spec_file, spec.to_string())
            .map_err(|e| format!("write {}: {e}", spec_file.display()))?;
        std::fs::write(&script, SVMBIR_SCRIPT)
            .map_err(|e| format!("write {}: {e}", script.display()))?;
        let output = std::process::Command::new(SVMBIR_PYTHON)
            .arg(&script)
            .arg(&sino_npy)
            .arg(&spec_file)
            .arg(&out_npy)
            .output()
            .map_err(|e| format!("cannot launch {SVMBIR_PYTHON}: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: Vec<&str> = stderr.trim().lines().rev().take(4).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!(
                "svmbir failed ({}): {}",
                output.status,
                tail.join(" | ")
            ));
        }
        let (shape, values) = read_npy(&out_npy)?;
        if shape != [2, rw, rw] {
            return Err(format!("svmbir returned shape {shape:?}, expected (2, {rw}, {rw})"));
        }
        let (top, bottom) = values.split_at(rw * rw);
        Ok((rw, top.to_vec(), bottom.to_vec()))
    };
    let result = run();
    cleanup();
    result
}

/// The standalone tilt & center-of-rotation tool's correction records in a
/// stack's metadata: one JSON object per applied correction, oldest first.
pub fn tilt_tool_records(metadata: &[(String, String)]) -> Vec<String> {
    metadata
        .iter()
        .find(|(name, _)| name == "tilt_center_of_rotation")
        .map(|(_, value)| {
            value
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// The checkpoint's geometry without loading the projections: its
/// `/center_of_rotation` and the tilt tool's correction records. Used to
/// tell whether the tool changed the file before reloading gigabytes.
pub fn checkpoint_geometry(path: &Path) -> Result<(Option<f64>, Vec<String>), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let cor = file
        .dataset("center_of_rotation")
        .and_then(|ds| ds.read_scalar::<f64>())
        .ok();
    let records = file
        .group("metadata")
        .and_then(|g| g.dataset("tilt_center_of_rotation"))
        .and_then(|ds| ds.read_scalar::<VarLenUnicode>())
        .map(|v| {
            v.as_str()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    Ok((cor, records))
}

/// Write (or replace) the `svmbir_config` JSON in the checkpoint's
/// `/metadata` group, where the main application reads it back.
pub fn save_params(path: &Path, params: &SvmbirParams) -> Result<(), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open_rw(path)
        .map_err(|e| format!("cannot open {} for writing: {e}", path.display()))?;
    let metadata = match file.group("metadata") {
        Ok(group) => group,
        Err(_) => file
            .create_group("metadata")
            .map_err(|e| format!("create metadata group: {e}"))?,
    };
    if metadata.dataset("svmbir_config").is_ok() {
        metadata
            .unlink("svmbir_config")
            .map_err(|e| format!("replace svmbir_config: {e}"))?;
    }
    let value: VarLenUnicode = params.to_json().parse().unwrap_or_default();
    metadata
        .new_dataset::<VarLenUnicode>()
        .create("svmbir_config")
        .and_then(|ds| ds.write_scalar(&value))
        .map_err(|e| format!("write svmbir_config: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_json_roundtrip() {
        let params = SvmbirParams {
            sharpness: 1.2,
            snr_db: 32.0,
            positivity: false,
            max_iterations: 42,
            max_resolutions: 2,
            center_offset: -3.5,
        };
        let back = SvmbirParams::from_json(&params.to_json()).unwrap();
        assert_eq!(back, params);
        assert!(SvmbirParams::from_json("not json").is_none());
        // Partial json keeps defaults for missing fields.
        let partial = SvmbirParams::from_json(r#"{"sharpness": 2.0}"#).unwrap();
        assert_eq!(partial.sharpness, 2.0);
        assert_eq!(partial.max_iterations, 20);
    }

    #[test]
    fn offset_and_center_are_inverse() {
        // The seeding rule: offset = -(width/2 - cor).
        assert_eq!(offset_from_center(979.96, 2054), -(1027.0 - 979.96));
        assert!((center_from_offset(offset_from_center(979.96, 2054), 2054) - 979.96).abs() < 1e-9);
        // Odd widths use the integer half width, like the notebook.
        assert_eq!(center_from_offset(0.0, 2055), 1027.0);
    }

    #[test]
    fn rebinned_offset_follows_the_pixel_grid() {
        // No rebin: unchanged.
        assert_eq!(rebinned_offset(-3.5, 2054, 1), -3.5);
        // The detector center stays the detector center of the rebinned
        // image (4096 wide: cor 2047.5 -> 1023.5 at 2x2, offset -0.5 both).
        let c = offset_from_center(2047.5, 4096);
        assert!((rebinned_offset(c, 4096, 2) - offset_from_center(1023.5, 2048)).abs() < 1e-9);
        // A general center: cor 979.96 on 2054 px -> (979.96+0.5)/2-0.5 on 1027 px.
        let cor = 979.96;
        let expected = offset_from_center((cor + 0.5) / 2.0 - 0.5, 1027);
        assert!((rebinned_offset(offset_from_center(cor, 2054), 2054, 2) - expected).abs() < 1e-9);
    }

    #[test]
    fn band_rows_hold_band_slices_after_rebin() {
        // 2x2: 8 rows around row 100 -> rebinned to 4 slices.
        assert_eq!(band_rows(100, 2048, 2), (96, 104));
        // Clamped at the edges.
        assert_eq!(band_rows(0, 2048, 3), (0, 12));
        assert_eq!(band_rows(2047, 2048, 1), (2044, 2048));
        // Tiny stacks limit the useful factor.
        assert_eq!(max_test_rebin(10), 2);
        assert_eq!(max_test_rebin(3), 1);
        assert_eq!(max_test_rebin(2048), 6);
    }

    #[test]
    fn extract_bands_rebins_each_band() {
        // 4 px wide, 8 rows: row r holds the value r everywhere.
        let mut mean = Vec::new();
        for r in 0..8 {
            mean.extend(std::iter::repeat_n(r as f32, 4));
        }
        let p = Projection {
            name: "p".into(),
            run_number: None,
            angle_deg: Some(0.0),
            n_images_used: 1,
            height: 8,
            width: 4,
            mean,
            total_counts: 0.0,
        };
        // Full resolution: the rows verbatim.
        let full = extract_bands(&p, [(0, 2), (6, 8)], 1);
        assert_eq!(full, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0]);
        // 2x2: rows (0,1) -> 0.5, rows (2,3) -> 2.5; 2 px wide.
        let reb = extract_bands(&p, [(0, 4), (4, 8)], 2);
        assert_eq!(reb, vec![0.5, 0.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5]);
    }
}
