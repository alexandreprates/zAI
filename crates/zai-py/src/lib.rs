//! Python module `zai_sim`: bindings over the `zai-sim` crate.
//!
//! - `Env`: one match, for tests, debugging and the PettingZoo wrapper.
//! - `VecEnv`: many matches stepped in parallel (rayon, GIL released) that write into
//!   preallocated NumPy arrays. The returned arrays are the same objects on every call and
//!   are overwritten by the next `reset()` / `step()`; copy them to keep a value.
//!
//! Configurations cross the boundary as JSON strings validated by the Rust `Config` schema.

use ::zai_sim::env::{INTERACT_ACTIONS, MOVE_ACTIONS};
use ::zai_sim::obs::{
    write_human_obs, write_state, write_zombie_obs, Block, HUMAN_LAYOUT, STATE_LAYOUT,
    ZOMBIE_LAYOUT,
};
use ::zai_sim::{
    Config, Env as SimEnv, HumanAction, Outcome, HUMAN_OBS_DIM, STATE_DIM, ZOMBIE_OBS_DIM,
};
use numpy::{Element, PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

const MASK_DIM: usize = MOVE_ACTIONS + INTERACT_ACTIONS;

fn parse_config(json: &str) -> PyResult<Config> {
    let cfg: Config = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("invalid config: {e}")))?;
    cfg.validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(cfg)
}

fn to_json(cfg: &Config) -> String {
    serde_json::to_string(cfg).expect("Config serializes to JSON")
}

/// Outcome codes: 0 running, 1 caught, 2 starved, 3 truncated.
fn outcome_code(o: Outcome) -> u8 {
    match o {
        Outcome::Running => 0,
        Outcome::Caught => 1,
        Outcome::Starved => 2,
        Outcome::Truncated => 3,
    }
}

/// Seed of episode `episode` of environment `env_index` in a `VecEnv` seeded with `base`.
#[pyfunction]
fn episode_seed(base: u64, env_index: u64, episode: u64) -> u64 {
    let mut x = base
        ^ env_index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ episode.wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    // splitmix64 finalizer
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Default configuration (PRD values) as JSON.
#[pyfunction]
fn default_config() -> String {
    to_json(&Config::default())
}

/// Validates a (possibly partial) JSON config and returns it complete, as JSON.
#[pyfunction]
fn validate_config(json: &str) -> PyResult<String> {
    Ok(to_json(&parse_config(json)?))
}

/// Applies the `ZAI_ZOMBIE_COUNT` / `ZAI_FOOD_COUNT` environment variables to a JSON config.
#[pyfunction]
fn apply_env_overrides(json: &str) -> PyResult<String> {
    let cfg = parse_config(json)?
        .with_env_overrides()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    cfg.validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(to_json(&cfg))
}

fn layout_dict<'py>(py: Python<'py>, layout: &[Block]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for b in layout {
        let e = PyDict::new(py);
        e.set_item("start", b.start)?;
        e.set_item("slots", b.slots)?;
        e.set_item("fields", PyList::new(py, b.fields)?)?;
        d.set_item(b.name, e)?;
    }
    Ok(d)
}

/// Observation layouts: `{"human"|"zombie"|"state": {block: {start, slots, fields}}}`.
#[pyfunction]
fn obs_layout(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("human", layout_dict(py, &HUMAN_LAYOUT)?)?;
    d.set_item("zombie", layout_dict(py, &ZOMBIE_LAYOUT)?)?;
    d.set_item("state", layout_dict(py, &STATE_LAYOUT)?)?;
    Ok(d)
}

fn check_human_action(movement: i64, interact: i64) -> PyResult<HumanAction> {
    if !(0..MOVE_ACTIONS as i64).contains(&movement)
        || !(0..INTERACT_ACTIONS as i64).contains(&interact)
    {
        return Err(PyValueError::new_err(format!(
            "human action ({movement}, {interact}) out of range ({MOVE_ACTIONS}, {INTERACT_ACTIONS})"
        )));
    }
    Ok(HumanAction {
        movement: movement as u8,
        interact: interact as u8,
    })
}

fn check_zombie_action(a: i64) -> PyResult<u8> {
    if !(0..MOVE_ACTIONS as i64).contains(&a) {
        return Err(PyValueError::new_err(format!(
            "zombie action {a} out of range {MOVE_ACTIONS}"
        )));
    }
    Ok(a as u8)
}

fn write_mask(env: &SimEnv, out: &mut [bool]) {
    out.copy_from_slice(&env.human_action_mask());
}

/// One match.
#[pyclass(module = "zai_sim")]
struct Env {
    inner: SimEnv,
}

#[pymethods]
impl Env {
    #[new]
    fn new(config_json: &str, seed: u64) -> PyResult<Self> {
        let cfg = parse_config(config_json)?;
        let inner = SimEnv::new(cfg, seed).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
    }

    /// Returns `(human_reward, zombie_reward, terminated, truncated, outcome_code)`.
    fn step(
        &mut self,
        movement: i64,
        interact: i64,
        zombie_actions: Vec<i64>,
    ) -> PyResult<(f32, f32, bool, bool, u8)> {
        if self.inner.outcome() != Outcome::Running {
            return Err(PyValueError::new_err(
                "step() called on a finished episode; call reset()",
            ));
        }
        if zombie_actions.len() != self.inner.zombies().len() {
            return Err(PyValueError::new_err(format!(
                "expected {} zombie actions, got {}",
                self.inner.zombies().len(),
                zombie_actions.len()
            )));
        }
        let h = check_human_action(movement, interact)?;
        let z = zombie_actions
            .into_iter()
            .map(check_zombie_action)
            .collect::<PyResult<Vec<u8>>>()?;
        let r = self.inner.step(h, &z);
        Ok((
            r.human_reward,
            r.zombie_reward,
            r.terminated,
            r.truncated,
            outcome_code(r.outcome),
        ))
    }

    fn human_obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut v = vec![0.0; HUMAN_OBS_DIM];
        write_human_obs(&self.inner, &mut v);
        PyArray1::from_vec(py, v)
    }

    fn human_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &self.inner.human_action_mask())
    }

    /// Observations of all zombies, shape `[N, ZOMBIE_OBS_DIM]`.
    fn zombie_obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.inner.zombies().len();
        let arr = PyArray2::<f32>::zeros(py, [n, ZOMBIE_OBS_DIM], false);
        // SAFETY: freshly allocated, contiguous and not shared yet.
        let buf = unsafe { arr.as_slice_mut().expect("contiguous") };
        for (i, chunk) in buf.chunks_mut(ZOMBIE_OBS_DIM).enumerate() {
            write_zombie_obs(&self.inner, i, chunk);
        }
        arr
    }

    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut v = vec![0.0; STATE_DIM];
        write_state(&self.inner, &mut v);
        PyArray1::from_vec(py, v)
    }

    fn state_hash(&self) -> u64 {
        self.inner.state_hash()
    }

    #[getter]
    fn num_zombies(&self) -> usize {
        self.inner.zombies().len()
    }

    #[getter]
    fn time(&self) -> f32 {
        self.inner.time()
    }

    #[getter]
    fn decisions(&self) -> u32 {
        self.inner.decisions()
    }

    #[getter]
    fn outcome(&self) -> u8 {
        outcome_code(self.inner.outcome())
    }

    #[getter]
    fn config_json(&self) -> String {
        to_json(self.inner.config())
    }
}

/// Raw view of a NumPy buffer that worker threads write into disjoint rows of.
#[derive(Clone, Copy)]
struct RawBuf<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: every worker only touches the rows of its own environment index.
unsafe impl<T> Send for RawBuf<T> {}
unsafe impl<T> Sync for RawBuf<T> {}

impl<T> RawBuf<T> {
    fn of<D: numpy::ndarray::Dimension>(arr: &Bound<'_, PyArray<T, D>>) -> Self
    where
        T: Element,
    {
        // SAFETY: the arrays are allocated contiguous by VecEnv and owned by it.
        let s = unsafe { arr.as_slice_mut().expect("contiguous buffer") };
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }

    /// Row `i` of width `w`.
    ///
    /// # Safety
    /// Rows must not be accessed concurrently by more than one thread.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, i: usize, w: usize) -> &mut [T] {
        assert!((i + 1) * w <= self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(i * w), w)
    }
}

struct Buffers {
    human_obs: Py<PyArray2<f32>>,
    human_mask: Py<PyArray2<bool>>,
    zombie_obs: Py<PyArray3<f32>>,
    zombie_valid: Py<PyArray2<bool>>,
    state: Py<PyArray2<f32>>,
    human_reward: Py<PyArray1<f32>>,
    zombie_reward: Py<PyArray1<f32>>,
    terminated: Py<PyArray1<bool>>,
    truncated: Py<PyArray1<bool>>,
    outcome: Py<PyArray1<u8>>,
    episode_length: Py<PyArray1<u32>>,
    final_human_obs: Py<PyArray2<f32>>,
    final_zombie_obs: Py<PyArray3<f32>>,
    final_state: Py<PyArray2<f32>>,
}

#[derive(Clone, Copy)]
struct Raw {
    human_obs: RawBuf<f32>,
    human_mask: RawBuf<bool>,
    zombie_obs: RawBuf<f32>,
    state: RawBuf<f32>,
    human_reward: RawBuf<f32>,
    zombie_reward: RawBuf<f32>,
    terminated: RawBuf<bool>,
    truncated: RawBuf<bool>,
    outcome: RawBuf<u8>,
    episode_length: RawBuf<u32>,
    final_human_obs: RawBuf<f32>,
    final_zombie_obs: RawBuf<f32>,
    final_state: RawBuf<f32>,
}

impl Buffers {
    fn new(py: Python<'_>, e: usize, n: usize) -> Self {
        let zombie_valid = PyArray2::<bool>::zeros(py, [e, n], false);
        // SAFETY: freshly allocated and contiguous.
        unsafe { zombie_valid.as_slice_mut().expect("contiguous") }.fill(true);
        Self {
            human_obs: PyArray2::zeros(py, [e, HUMAN_OBS_DIM], false).unbind(),
            human_mask: PyArray2::zeros(py, [e, MASK_DIM], false).unbind(),
            zombie_obs: PyArray3::zeros(py, [e, n, ZOMBIE_OBS_DIM], false).unbind(),
            zombie_valid: zombie_valid.unbind(),
            state: PyArray2::zeros(py, [e, STATE_DIM], false).unbind(),
            human_reward: PyArray1::zeros(py, [e], false).unbind(),
            zombie_reward: PyArray1::zeros(py, [e], false).unbind(),
            terminated: PyArray1::zeros(py, [e], false).unbind(),
            truncated: PyArray1::zeros(py, [e], false).unbind(),
            outcome: PyArray1::zeros(py, [e], false).unbind(),
            episode_length: PyArray1::zeros(py, [e], false).unbind(),
            final_human_obs: PyArray2::zeros(py, [e, HUMAN_OBS_DIM], false).unbind(),
            final_zombie_obs: PyArray3::zeros(py, [e, n, ZOMBIE_OBS_DIM], false).unbind(),
            final_state: PyArray2::zeros(py, [e, STATE_DIM], false).unbind(),
        }
    }

    fn raw(&self, py: Python<'_>) -> Raw {
        Raw {
            human_obs: RawBuf::of(self.human_obs.bind(py)),
            human_mask: RawBuf::of(self.human_mask.bind(py)),
            zombie_obs: RawBuf::of(self.zombie_obs.bind(py)),
            state: RawBuf::of(self.state.bind(py)),
            human_reward: RawBuf::of(self.human_reward.bind(py)),
            zombie_reward: RawBuf::of(self.zombie_reward.bind(py)),
            terminated: RawBuf::of(self.terminated.bind(py)),
            truncated: RawBuf::of(self.truncated.bind(py)),
            outcome: RawBuf::of(self.outcome.bind(py)),
            episode_length: RawBuf::of(self.episode_length.bind(py)),
            final_human_obs: RawBuf::of(self.final_human_obs.bind(py)),
            final_zombie_obs: RawBuf::of(self.final_zombie_obs.bind(py)),
            final_state: RawBuf::of(self.final_state.bind(py)),
        }
    }

    fn obs_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("human_obs", self.human_obs.bind(py))?;
        d.set_item("human_mask", self.human_mask.bind(py))?;
        d.set_item("zombie_obs", self.zombie_obs.bind(py))?;
        d.set_item("zombie_valid", self.zombie_valid.bind(py))?;
        d.set_item("state", self.state.bind(py))?;
        Ok(d)
    }
}

/// Writes the observations of env `i` into rows `i` of the given buffers.
///
/// # Safety
/// Row `i` must not be accessed concurrently.
unsafe fn write_obs(
    env: &SimEnv,
    i: usize,
    n: usize,
    h: RawBuf<f32>,
    z: RawBuf<f32>,
    s: RawBuf<f32>,
) {
    write_human_obs(env, h.row(i, HUMAN_OBS_DIM));
    for (k, chunk) in z
        .row(i, n * ZOMBIE_OBS_DIM)
        .chunks_mut(ZOMBIE_OBS_DIM)
        .enumerate()
    {
        write_zombie_obs(env, k, chunk);
    }
    write_state(env, s.row(i, STATE_DIM));
}

/// Many matches with the same config, stepped in parallel with automatic reset.
#[pyclass(module = "zai_sim")]
struct VecEnv {
    envs: Vec<SimEnv>,
    episodes: Vec<u64>,
    base_seed: u64,
    num_zombies: usize,
    pool: rayon::ThreadPool,
    bufs: Buffers,
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (config_json, num_envs, seed=0, num_threads=None))]
    fn new(
        py: Python<'_>,
        config_json: &str,
        num_envs: usize,
        seed: u64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be >= 1"));
        }
        let cfg = parse_config(config_json)?;
        let n = cfg.zombie_count;
        let envs = (0..num_envs)
            .map(|i| SimEnv::new(cfg.clone(), episode_seed(seed, i as u64, 0)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(0))
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            envs,
            episodes: vec![0; num_envs],
            base_seed: seed,
            num_zombies: n,
            pool,
            bufs: Buffers::new(py, num_envs, n),
        })
    }

    /// Resets every environment (optionally with a new base seed) and returns the observations.
    #[pyo3(signature = (seed=None))]
    fn reset<'py>(&mut self, py: Python<'py>, seed: Option<u64>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(s) = seed {
            self.base_seed = s;
        }
        let raw = self.bufs.raw(py);
        let base = self.base_seed;
        let n = self.num_zombies;
        let Self {
            envs,
            episodes,
            pool,
            ..
        } = self;
        py.allow_threads(|| {
            pool.install(|| {
                envs.par_iter_mut()
                    .zip(episodes.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, (env, ep))| {
                        *ep = 0;
                        env.reset(episode_seed(base, i as u64, 0));
                        // SAFETY: each index is handled by exactly one worker.
                        unsafe {
                            write_obs(env, i, n, raw.human_obs, raw.zombie_obs, raw.state);
                            write_mask(env, raw.human_mask.row(i, MASK_DIM));
                        }
                    })
            })
        });
        self.bufs.obs_dict(py)
    }

    /// Steps every environment.
    ///
    /// `human_actions`: int64 `[E, 2]` (movement, interact); `zombie_actions`: int64 `[E, N]`.
    /// Returns `(obs, human_reward, zombie_reward, terminated, truncated, info)`. Finished
    /// environments are reset automatically: `obs` holds the first observation of the new
    /// episode and `info["final_*"]` the last observation of the finished one (only rows where
    /// `terminated | truncated` are meaningful).
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        human_actions: PyReadonlyArray2<'py, i64>,
        zombie_actions: PyReadonlyArray2<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyDict>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyDict>,
    )> {
        let e = self.envs.len();
        let n = self.num_zombies;
        let ha = human_actions.as_array();
        let za = zombie_actions.as_array();
        if ha.shape() != [e, 2] {
            return Err(PyValueError::new_err(format!(
                "human_actions must have shape [{e}, 2], got {:?}",
                ha.shape()
            )));
        }
        if za.shape() != [e, n] {
            return Err(PyValueError::new_err(format!(
                "zombie_actions must have shape [{e}, {n}], got {:?}",
                za.shape()
            )));
        }
        let human: Vec<HumanAction> = (0..e)
            .map(|i| check_human_action(ha[[i, 0]], ha[[i, 1]]))
            .collect::<PyResult<_>>()?;
        let zombies: Vec<u8> = za
            .iter()
            .map(|&a| check_zombie_action(a))
            .collect::<PyResult<_>>()?;

        let raw = self.bufs.raw(py);
        let base = self.base_seed;
        let Self {
            envs,
            episodes,
            pool,
            ..
        } = self;
        py.allow_threads(|| {
            pool.install(|| {
                envs.par_iter_mut()
                    .zip(episodes.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, (env, ep))| {
                        let r = env.step(human[i], &zombies[i * n..(i + 1) * n]);
                        // SAFETY: each index is handled by exactly one worker.
                        unsafe {
                            raw.human_reward.row(i, 1)[0] = r.human_reward;
                            raw.zombie_reward.row(i, 1)[0] = r.zombie_reward;
                            raw.terminated.row(i, 1)[0] = r.terminated;
                            raw.truncated.row(i, 1)[0] = r.truncated;
                            raw.outcome.row(i, 1)[0] = outcome_code(r.outcome);
                            raw.episode_length.row(i, 1)[0] = env.decisions();
                            if r.outcome != Outcome::Running {
                                write_obs(
                                    env,
                                    i,
                                    n,
                                    raw.final_human_obs,
                                    raw.final_zombie_obs,
                                    raw.final_state,
                                );
                                *ep += 1;
                                env.reset(episode_seed(base, i as u64, *ep));
                            }
                            write_obs(env, i, n, raw.human_obs, raw.zombie_obs, raw.state);
                            write_mask(env, raw.human_mask.row(i, MASK_DIM));
                        }
                    })
            })
        });

        let b = &self.bufs;
        let info = PyDict::new(py);
        info.set_item("outcome", b.outcome.bind(py))?;
        info.set_item("episode_length", b.episode_length.bind(py))?;
        info.set_item("final_human_obs", b.final_human_obs.bind(py))?;
        info.set_item("final_zombie_obs", b.final_zombie_obs.bind(py))?;
        info.set_item("final_state", b.final_state.bind(py))?;
        Ok((
            b.obs_dict(py)?,
            b.human_reward.bind(py).clone(),
            b.zombie_reward.bind(py).clone(),
            b.terminated.bind(py).clone(),
            b.truncated.bind(py).clone(),
            info,
        ))
    }

    /// Seed of the episode currently running in each environment.
    fn current_seeds(&self) -> Vec<u64> {
        self.episodes
            .iter()
            .enumerate()
            .map(|(i, &ep)| episode_seed(self.base_seed, i as u64, ep))
            .collect()
    }

    fn state_hashes(&self) -> Vec<u64> {
        self.envs.iter().map(SimEnv::state_hash).collect()
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    #[getter]
    fn num_zombies(&self) -> usize {
        self.num_zombies
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    #[getter]
    fn config_json(&self) -> String {
        to_json(self.envs[0].config())
    }
}

#[pymodule]
fn zai_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", ::zai_sim::VERSION)?;
    m.add("HUMAN_OBS_DIM", HUMAN_OBS_DIM)?;
    m.add("ZOMBIE_OBS_DIM", ZOMBIE_OBS_DIM)?;
    m.add("STATE_DIM", STATE_DIM)?;
    m.add("MOVE_ACTIONS", MOVE_ACTIONS)?;
    m.add("INTERACT_ACTIONS", INTERACT_ACTIONS)?;
    m.add("MAX_ZOMBIES", ::zai_sim::MAX_ZOMBIES)?;
    m.add_function(wrap_pyfunction!(default_config, m)?)?;
    m.add_function(wrap_pyfunction!(validate_config, m)?)?;
    m.add_function(wrap_pyfunction!(apply_env_overrides, m)?)?;
    m.add_function(wrap_pyfunction!(episode_seed, m)?)?;
    m.add_function(wrap_pyfunction!(obs_layout, m)?)?;
    m.add_class::<Env>()?;
    m.add_class::<VecEnv>()?;
    Ok(())
}
