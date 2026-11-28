import json

from arena.league import Arena, ArenaConfig
from zai_env.config import config_dir
from zai_rl.league.pool import Entry, Pool
from zai_rl.models import HumanActor, ZombieActor
from zai_rl.train_league import save_snapshot


class Recorder:
    def __init__(self):
        self.logged = []

    def log(self, step, scalars):
        self.logged.append((step, scalars))


def test_arena_plays_the_elo_set_once_and_reports(tmp_path):
    pool = Pool()
    for side, actor in (("human", HumanActor(hidden=8)), ("zombie", ZombieActor(hidden=8))):
        for k in range(2):
            path = tmp_path / f"{side}{k}.pt"
            save_snapshot(path, actor, side, 50_000_000 * k, "s0_small")
            entry_id = pool.next_id(side)
            pool.add(Entry(entry_id, side, str(path), 50_000_000 * k, "s0_small", "league"))
    summary = pool.summary()
    cfg = ArenaConfig(
        run_id="lg",
        league_dir=tmp_path / "league",
        env_config=str(config_dir() / "s0_small.yaml"),
        zombie_counts=[2],
        episodes=2,
        bootstrap_samples=5,
        num_threads=1,
        data_dir=tmp_path / "data",
    )
    cfg.league_dir.mkdir()
    rec = Recorder()
    arena = Arena(cfg, rec)
    # 4 humans (scripted, random, 2 snapshots) x 4 zombie teams.
    assert arena.round(summary) == 16
    assert arena.round(summary) == 0, "results are never recomputed"
    report = arena.publish(summary)
    assert {e["id"] for e in report["entries"]} >= {"zombie-scripted", "h0002", "z0002"}
    assert report["latest"] == {"human": "h0002", "zombie": "z0002"}
    assert json.loads((tmp_path / "data" / "arena.json").read_text())["run_id"] == "lg"
    step, scalars = rec.logged[0]
    assert step == 50_000_000
    assert {"elo/human", "elo/zombie_lo", "anchor/caught", "anchor/baseline_survived"} <= set(
        scalars
    )
    assert "xplay/human_score_vs_old" in scalars
    assert any(k.startswith("tactics/human/baseline_") for k in scalars)
    # A restarted arena reads the table back from crossplay.jsonl.
    assert Arena(cfg).round(summary) == 0


def test_probe_configs_load_in_the_single_side_trainers(tmp_path):
    from dataclasses import asdict

    import yaml

    from arena.probe import probe_config
    from zai_rl.train import TrainConfig
    from zai_rl.train_league import LeagueConfig
    from zai_rl.train_zombie import ZombieTrainConfig

    league = asdict(
        LeagueConfig.load(config_dir().parent / "train" / "league_s0small_gpu.yaml", {})
    )
    for side, cls in (("human", TrainConfig), ("zombie", ZombieTrainConfig)):
        cfg = probe_config(league, side, tmp_path / "opp.pt", tmp_path / "me.pt", 1000, 4)
        path = tmp_path / f"{side}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        loaded = cls.load(path, {})
        assert loaded.opponent == "policy" and loaded.init_checkpoint.endswith("me.pt")


def test_showcase_follows_the_newest_league_snapshots():
    from arena.showcase import latest

    pool = {
        "stage": {"starters": {"human": "human-starter", "zombie": "zombie-starter"}},
        "sides": {
            "human": [
                {"id": "human-scripted", "source": "scripted", "path": None, "steps": 0},
                {"id": "human-starter", "source": "distilled", "path": "s.pt", "steps": 10},
            ],
            "zombie": [
                {"id": "zombie-starter", "source": "distilled", "path": "zs.pt", "steps": 10},
                {"id": "z0001", "source": "league", "path": "z1.pt", "steps": 100},
                {"id": "z0002", "source": "league", "path": "z2.pt", "steps": 200},
            ],
        },
    }
    assert latest(pool, "human")["id"] == "human-starter"
    assert latest(pool, "zombie")["id"] == "z0002"
