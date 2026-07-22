//! SVMBIR parameters, the test reconstruction on two slice bands (through
//! the real svmbir of the `all_ct_reconstruction_development` pixi
//! environment), and saving the parameters back into the checkpoint HDF5.

use ct_reconstruction::combine::LoadedStack;
use ct_reconstruction::crop::{read_npy, write_npy};
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
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = run_recon(&stack, top_slice, bottom_slice, params)
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

fn run_recon(
    stack: &LoadedStack,
    top_slice: usize,
    bottom_slice: usize,
    params: SvmbirParams,
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

    // One 4-slice band around each selected line; the middle slice is shown.
    let band_range = |line: usize| -> (usize, usize) {
        let half = BAND / 2;
        let start = line.saturating_sub(half).min(h.saturating_sub(BAND));
        (start, start + BAND)
    };
    let (top_a, top_b) = band_range(top_slice);
    let (bottom_a, bottom_b) = band_range(bottom_slice);

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
        // The two bands stacked into one (n, 2*BAND, w) volume.
        let mut volume = Vec::with_capacity(n * 2 * BAND * w);
        for p in &stack.sample {
            for range in [(top_a, top_b), (bottom_a, bottom_b)] {
                volume.extend_from_slice(&p.mean[range.0 * w..range.1 * w]);
            }
        }
        write_npy(&sino_npy, &[n, 2 * BAND, w], volume.chunks(2 * BAND * w))?;
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
            "params": serde_json::from_str::<serde_json::Value>(&params.to_json()).expect("params json"),
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
        if shape != [2, w, w] {
            return Err(format!("svmbir returned shape {shape:?}, expected (2, {w}, {w})"));
        }
        let (top, bottom) = values.split_at(w * w);
        Ok((w, top.to_vec(), bottom.to_vec()))
    };
    let result = run();
    cleanup();
    result
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
}
