//! Trained policies for live matches: actors exported to ONNX by
//! `python/arena/export_model.py` (`<id>.onnx` + `<id>.json` in `<ZAI_DATA_DIR>/models`),
//! run on the CPU with `tract`.
//!
//! The graph is one decision, `(obs [B, D], h [B, H]) -> (move logits, [interact logits,]
//! h_new)`; observations come from the same `zai_sim::obs` writers the training env uses,
//! and the action mask and sampling are applied here.

use std::path::PathBuf;
use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;
use zai_sim::env::{INTERACT_ACTIONS, MOVE_ACTIONS};
use zai_sim::obs::{write_human_obs, write_zombie_obs};
use zai_sim::{Env, HumanAction, HUMAN_OBS_DIM, ZOMBIE_OBS_DIM};

/// Model ids are file names, with the same rules as replay ids.
pub use crate::replay::valid_id;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Human,
    Zombie,
}

impl Side {
    fn obs_dim(self) -> usize {
        match self {
            Side::Human => HUMAN_OBS_DIM,
            Side::Zombie => ZOMBIE_OBS_DIM,
        }
    }
}

/// Contents of `<id>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub side: Side,
    pub run_id: String,
    pub iteration: u64,
    pub steps: u64,
    /// Name of the env config file the model was trained on (`custom` if none matched).
    pub stage: String,
    pub hidden: usize,
    pub obs_dim: usize,
    pub created_at: u64,
    /// Full env config of the training run (the map matches with this model should use).
    pub config: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ModelStore {
    dir: PathBuf,
}

impl ModelStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Models with both files present, newest first. Unreadable metadata is logged and skipped.
    pub fn list(&self) -> Vec<ModelInfo> {
        let mut out: Vec<ModelInfo> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .filter(|p| p.with_extension("onnx").is_file())
            .filter_map(|p| match Self::read_info(&p) {
                Ok(info) => Some(info),
                Err(e) => {
                    tracing::warn!("skipping model {}: {e}", p.display());
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(a.id.cmp(&b.id)));
        out
    }

    fn read_info(path: &std::path::Path) -> Result<ModelInfo, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())
    }

    pub fn info(&self, id: &str) -> Result<ModelInfo, String> {
        if !valid_id(id) {
            return Err(format!("invalid model id {id:?}"));
        }
        let path = self.dir.join(format!("{id}.json"));
        if !path.is_file() || !path.with_extension("onnx").is_file() {
            return Err(format!("model {id:?} not found"));
        }
        Self::read_info(&path).map_err(|e| format!("model {id:?}: {e}"))
    }

    /// Loads model `id` for `side`, optimized for `batch` agents per decision (blocking).
    pub fn load(&self, id: &str, side: Side, batch: usize) -> Result<LoadedModel, String> {
        let info = self.info(id)?;
        if info.side != side {
            return Err(format!("model {id:?} plays the {:?} side", info.side));
        }
        LoadedModel::load(self.dir.join(format!("{id}.onnx")), info, batch)
    }
}

type Plan = TypedRunnableModel<TypedModel>;

/// An optimized plan for a fixed batch size; cheap to clone, shared between episodes.
#[derive(Clone)]
pub struct LoadedModel {
    pub info: ModelInfo,
    plan: Arc<Plan>,
    batch: usize,
}

impl LoadedModel {
    pub fn load(path: PathBuf, info: ModelInfo, batch: usize) -> Result<Self, String> {
        let side = info.side;
        if info.obs_dim != side.obs_dim() {
            return Err(format!(
                "model {:?} expects {} observation features, the simulation writes {}",
                info.id,
                info.obs_dim,
                side.obs_dim()
            ));
        }
        let plan = tract_onnx::onnx()
            .model_for_path(&path)
            .and_then(|m| m.with_input_fact(0, f32::fact([batch, info.obs_dim]).into()))
            .and_then(|m| m.with_input_fact(1, f32::fact([batch, info.hidden]).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| format!("cannot load {}: {e:#}", path.display()))?;
        let outputs = plan.model().outputs.len();
        let expected = match side {
            Side::Human => 3,
            Side::Zombie => 2,
        };
        if outputs != expected {
            return Err(format!(
                "model {:?} has {outputs} outputs, expected {expected}",
                info.id
            ));
        }
        Ok(Self {
            info,
            plan: Arc::new(plan),
            batch,
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    /// One decision for `obs [batch * obs_dim]` and `h [batch * hidden]`: returns the logits
    /// outputs and replaces `h` with the new hidden state.
    pub fn run(&self, obs: &[f32], h: &mut [f32]) -> Result<Vec<Vec<f32>>, String> {
        let (b, d, hd) = (self.batch, self.info.obs_dim, self.info.hidden);
        let to_err = |e: TractError| format!("{e:#}");
        let obs = Tensor::from_shape(&[b, d], obs).map_err(to_err)?;
        let h_in = Tensor::from_shape(&[b, hd], h).map_err(to_err)?;
        let out = self
            .plan
            .run(tvec!(obs.into_tvalue(), h_in.into_tvalue()))
            .map_err(to_err)?;
        let (h_out, logits) = out.split_last().expect("validated output count");
        h.copy_from_slice(h_out.as_slice::<f32>().map_err(to_err)?);
        logits
            .iter()
            .map(|t| t.as_slice::<f32>().map(<[f32]>::to_vec).map_err(to_err))
            .collect()
    }
}

/// Index of the action drawn from `softmax(logits)` restricted to `allowed`, or the most
/// likely allowed action when `deterministic`. Falls back to the last action if none is
/// allowed (the env always allows at least `STAY` / `NONE`).
pub fn choose(logits: &[f32], allowed: &[bool], deterministic: bool, rng: &mut ChaCha8Rng) -> u8 {
    let max = logits
        .iter()
        .zip(allowed)
        .filter(|(_, &a)| a)
        .map(|(&l, _)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return (logits.len() - 1) as u8;
    }
    if deterministic {
        return logits
            .iter()
            .zip(allowed)
            .position(|(&l, &a)| a && l == max)
            .expect("max is attained") as u8;
    }
    let weights: Vec<f32> = logits
        .iter()
        .zip(allowed)
        .map(|(&l, &a)| if a { (l - max).exp() } else { 0.0 })
        .collect();
    let mut r = rng.random::<f32>() * weights.iter().sum::<f32>();
    let mut last = 0;
    for (i, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            last = i;
            if r < w {
                return i as u8;
            }
            r -= w;
        }
    }
    last as u8
}

/// Recurrent policy state for one episode.
pub struct PolicyAgent {
    model: LoadedModel,
    h: Vec<f32>,
    obs: Vec<f32>,
    rng: ChaCha8Rng,
    deterministic: bool,
}

impl PolicyAgent {
    pub fn new(model: LoadedModel, seed: u64, deterministic: bool) -> Self {
        let h = vec![0.0; model.batch * model.info.hidden];
        let obs = vec![0.0; model.batch * model.info.obs_dim];
        Self {
            model,
            h,
            obs,
            rng: ChaCha8Rng::seed_from_u64(seed),
            deterministic,
        }
    }

    pub fn act_human(&mut self, env: &Env) -> Result<HumanAction, String> {
        write_human_obs(env, &mut self.obs);
        let out = self.model.run(&self.obs, &mut self.h)?;
        let mask = env.human_action_mask();
        let (move_mask, interact_mask) = mask.split_at(MOVE_ACTIONS);
        debug_assert_eq!(interact_mask.len(), INTERACT_ACTIONS);
        Ok(HumanAction {
            movement: choose(&out[0], move_mask, self.deterministic, &mut self.rng),
            interact: choose(&out[1], interact_mask, self.deterministic, &mut self.rng),
        })
    }

    pub fn act_zombies(&mut self, env: &Env) -> Result<Vec<u8>, String> {
        let n = env.zombies().len();
        if n != self.model.batch {
            return Err(format!(
                "zombie model loaded for {} zombies, the match has {n}",
                self.model.batch
            ));
        }
        let d = self.model.info.obs_dim;
        for (i, chunk) in self.obs.chunks_mut(d).enumerate() {
            write_zombie_obs(env, i, chunk);
        }
        let out = self.model.run(&self.obs, &mut self.h)?;
        let allowed = [true; MOVE_ACTIONS];
        Ok(out[0]
            .chunks(MOVE_ACTIONS)
            .map(|l| choose(l, &allowed, self.deterministic, &mut self.rng))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_respects_the_mask_and_argmax() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let logits = [5.0, 1.0, 0.0];
        assert_eq!(choose(&logits, &[true, true, true], true, &mut rng), 0);
        assert_eq!(choose(&logits, &[false, true, true], true, &mut rng), 1);
        for _ in 0..100 {
            assert_ne!(choose(&logits, &[false, true, true], false, &mut rng), 0);
        }
        // Sampling follows the softmax: action 0 has ~98% of the mass.
        let hits = (0..1000)
            .filter(|_| choose(&logits, &[true; 3], false, &mut rng) == 0)
            .count();
        assert!(hits > 950, "{hits}");
    }

    #[test]
    fn model_ids_are_file_names() {
        assert!(valid_id("zombie-s0small-gpu1"));
        assert!(!valid_id("../etc"));
        assert!(!valid_id(""));
        assert!(!valid_id(&"a".repeat(65)));
    }
}
