//! The ONNX export (`python -m arena.export_model --parity-fixtures crates/zai-server/tests/fixtures`)
//! run with tract matches PyTorch's outputs for two chained decisions.

use std::path::PathBuf;

use serde::Deserialize;
use zai_server::policy::{ModelStore, Side};

#[derive(Deserialize)]
struct Fixture {
    batch: usize,
    steps: Vec<serde_json::Map<String, serde_json::Value>>,
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    serde_json::from_value(v.clone()).expect("float array")
}

fn check(side: Side, name: &str, outputs: &[&str]) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fixture: Fixture = serde_json::from_slice(
        &std::fs::read(dir.join(format!("{name}.parity.json"))).expect("fixture exists"),
    )
    .expect("fixture parses");
    let store = ModelStore::new(dir);
    assert!(store.list().iter().any(|m| m.id == name && m.side == side));
    let model = store.load(name, side, fixture.batch).expect("model loads");
    let mut h = vec![0.0; fixture.batch * model.info.hidden];
    for step in &fixture.steps {
        let out = model
            .run(&floats(&step["obs"]), &mut h)
            .expect("model runs");
        for (got, key) in out.iter().zip(outputs) {
            let want = floats(&step[*key]);
            assert_eq!(got.len(), want.len(), "{name} {key}");
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-4, "{name} {key}: {g} vs {w}");
            }
        }
        let want_h = floats(&step["h_out"]);
        for (g, w) in h.iter().zip(&want_h) {
            assert!((g - w).abs() < 1e-4, "{name} h_out: {g} vs {w}");
        }
    }
}

#[test]
fn human_model_matches_pytorch() {
    check(Side::Human, "human", &["move", "interact"]);
}

#[test]
fn zombie_model_matches_pytorch() {
    check(Side::Zombie, "zombie", &["move"]);
}

#[test]
fn a_model_cannot_play_the_other_side() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let err = ModelStore::new(dir)
        .load("human", Side::Zombie, 1)
        .err()
        .expect("side mismatch is an error");
    assert!(err.contains("human"), "{err}");
}
