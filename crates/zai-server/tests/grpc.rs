//! gRPC agent API against a real listener (docs/ARCHITECTURE.md section 4.9).

use std::sync::Arc;
use std::time::Duration;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{Code, Streaming};
use zai_server::grpc::pb::agent_service_client::AgentServiceClient;
use zai_server::grpc::pb::server_message::Msg;
use zai_server::grpc::pb::{
    client_message, Actions, ClientMessage, CreateMatchRequest, GetInfoRequest, Join,
    ServerMessage, Side, SideSpec,
};
use zai_server::grpc::PROTOCOL_VERSION;
use zai_server::matches::MatchStatus;
use zai_server::remote::RemoteSettings;
use zai_server::{grpc_service, AppState};
use zai_sim::env::{INTERACT_ACTIONS, MOVE_ACTIONS};
use zai_sim::obs::{write_human_obs, write_zombie_obs, Block, HUMAN_LAYOUT, ZOMBIE_LAYOUT};
use zai_sim::{Config, Env, HumanAction, Outcome, HUMAN_OBS_DIM, OBS_VERSION, ZOMBIE_OBS_DIM};

type Client = AgentServiceClient<tonic::transport::Channel>;

/// Serves the agent API on a free port; returns a connected client, the state and the temp
/// dir (kept alive by the caller).
async fn serve_with(remote: RemoteSettings) -> (Client, Arc<AppState>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut state = AppState::new(Config::default(), dir.path().join("replays"), None);
    state.remote = remote;
    let state = Arc::new(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(grpc_service(state.clone()))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let client = AgentServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    (client, state, dir)
}

async fn serve() -> (Client, tempfile::TempDir) {
    let (client, _, dir) = serve_with(RemoteSettings::default()).await;
    (client, dir)
}

/// Small, short episodes (75 m map, 3 zombies, 30 s cap = 60 decisions).
const SMALL: &str = r#"{"map_size_m": 75.0, "house_count": 3, "zombie_count": 3,
    "food_count": 20, "episode_cap_s": 30.0, "min_spawn_distance_m": 20.0}"#;

async fn create(client: &mut Client, human: &str, zombies: &str, auto_restart: bool) -> String {
    client
        .create_match(CreateMatchRequest {
            config_json: SMALL.into(),
            seed: Some(7),
            human: human.into(),
            zombies: zombies.into(),
            speed: 0.0,
            auto_restart,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner()
        .match_id
}

/// An open `Play` stream: outgoing messages and the server's replies.
struct Stream {
    tx: mpsc::Sender<ClientMessage>,
    rx: Streaming<ServerMessage>,
}

fn join_msg(match_id: &str, side: Side, obs_version: u32, obs_dim: usize) -> ClientMessage {
    ClientMessage {
        msg: Some(client_message::Msg::Join(Join {
            match_id: match_id.into(),
            side: side as i32,
            protocol: PROTOCOL_VERSION,
            obs_version,
            obs_dim: obs_dim as u32,
            agent_name: format!("test-{side:?}"),
        })),
    }
}

async fn play(client: &mut Client, first: ClientMessage) -> Result<Stream, tonic::Status> {
    let (tx, rx) = mpsc::channel(16);
    tx.send(first).await.unwrap();
    let rx = client.play(ReceiverStream::new(rx)).await?.into_inner();
    Ok(Stream { tx, rx })
}

async fn join(client: &mut Client, match_id: &str, side: Side) -> Stream {
    let dim = if side == Side::Human {
        HUMAN_OBS_DIM
    } else {
        ZOMBIE_OBS_DIM
    };
    play(client, join_msg(match_id, side, OBS_VERSION, dim))
        .await
        .unwrap()
}

impl Stream {
    async fn next(&mut self) -> Msg {
        tokio::time::timeout(Duration::from_secs(10), self.rx.message())
            .await
            .expect("server message within 10 s")
            .expect("stream ok")
            .expect("stream open")
            .msg
            .expect("message set")
    }

    async fn act(&self, episode: u32, tick: u32, actions: Vec<u8>) {
        let msg = ClientMessage {
            msg: Some(client_message::Msg::Actions(Actions {
                episode,
                tick,
                actions,
            })),
        };
        self.tx.send(msg).await.unwrap();
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

async fn wait_status(state: &AppState, id: &str, want: MatchStatus) -> String {
    for _ in 0..200 {
        let info = state.get_match(id).unwrap().info.lock().unwrap().clone();
        if info.status == want {
            return info.outcome;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let info = state.get_match(id).unwrap().info.lock().unwrap().clone();
    panic!(
        "match {id} is {:?} ({}), expected {want:?}",
        info.status, info.outcome
    );
}

fn assert_layout(spec: &SideSpec, obs_dim: usize, layout: &[Block], action_dims: &[usize]) {
    assert_eq!(spec.obs_dim as usize, obs_dim);
    assert_eq!(spec.blocks.len(), layout.len());
    for (got, want) in spec.blocks.iter().zip(layout) {
        assert_eq!(got.name, want.name);
        assert_eq!(got.start as usize, want.start);
        assert_eq!(got.len as usize, want.len());
        assert_eq!(got.slots as usize, want.slots);
        assert_eq!(got.fields, want.fields);
    }
    let end = layout.last().map_or(0, |b| b.start + b.len());
    assert_eq!(end, obs_dim, "the blocks cover the observation");
    let dims: Vec<usize> = spec.action_dims.iter().map(|&d| d as usize).collect();
    assert_eq!(dims, action_dims);
}

#[tokio::test]
async fn get_info_reports_versions_layouts_and_the_base_config() {
    let (mut client, _dir) = serve().await;
    let info = client
        .get_info(GetInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.version, zai_sim::VERSION);
    assert_eq!(info.protocol, PROTOCOL_VERSION);
    assert_eq!(info.obs_version, OBS_VERSION);
    assert_layout(
        info.human.as_ref().unwrap(),
        HUMAN_OBS_DIM,
        &HUMAN_LAYOUT,
        &[MOVE_ACTIONS, INTERACT_ACTIONS],
    );
    assert_layout(
        info.zombie.as_ref().unwrap(),
        ZOMBIE_OBS_DIM,
        &ZOMBIE_LAYOUT,
        &[MOVE_ACTIONS],
    );
    let base: Config = serde_json::from_str(&info.base_config_json).unwrap();
    assert_eq!(base.zombie_count, Config::default().zombie_count);
}

#[tokio::test]
async fn realtime_and_unknown_agents_are_refused() {
    let (mut client, _dir) = serve().await;
    let realtime = CreateMatchRequest {
        human: "remote".into(),
        speed: 1.0,
        realtime: true,
        ..Default::default()
    };
    let err = client.create_match(realtime).await.unwrap_err();
    assert_eq!(err.code(), Code::Unimplemented);
    let unknown = CreateMatchRequest {
        human: "keyboard".into(),
        ..Default::default()
    };
    let err = client.create_match(unknown).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[tokio::test]
async fn both_sides_remote_match_a_locally_stepped_env_byte_for_byte() {
    let (mut client, state, _dir) = serve_with(RemoteSettings::default()).await;
    let id = create(&mut client, "remote", "remote", false).await;
    assert_eq!(
        state.get_match(&id).unwrap().info.lock().unwrap().status,
        MatchStatus::Waiting
    );
    let mut frames = state.get_match(&id).unwrap().frames.subscribe();
    let mut h = join(&mut client, &id, Side::Human).await;
    let mut z = join(&mut client, &id, Side::Zombies).await;
    for s in [&mut h, &mut z] {
        assert!(matches!(s.next().await, Msg::Joined(_)));
    }
    let Msg::Start(start) = h.next().await else {
        panic!("expected EpisodeStart")
    };
    assert!(matches!(z.next().await, Msg::Start(_)));
    let cfg: Config = serde_json::from_str(&start.config_json).unwrap();
    assert_eq!(cfg.zombie_count, 3);
    let mut env = Env::new(cfg, start.seed).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let (mut hobs, mut zobs) = (vec![0.0; HUMAN_OBS_DIM], vec![0.0; ZOMBIE_OBS_DIM]);
    loop {
        let (hm, zm) = (h.next().await, z.next().await);
        if let (Msg::End(he), Msg::End(ze)) = (&hm, &zm) {
            assert_ne!(env.outcome(), Outcome::Running);
            assert_eq!(he.state_hash, env.state_hash());
            assert_eq!(ze.state_hash, env.state_hash());
            assert_eq!(he.ticks, env.decisions());
            let replay = state.replays.load(&he.replay_id).expect("replay saved");
            assert!(replay.verify(), "the replay re-simulates to its hash");
            assert_eq!(replay.human_model.as_deref(), Some("test-Human"));
            break;
        }
        let (Msg::Obs(ho), Msg::Obs(zo)) = (hm, zm) else {
            panic!("expected observations")
        };
        assert_eq!(ho.tick, env.decisions());
        assert_eq!(ho.state_hash, env.state_hash());
        write_human_obs(&env, &mut hobs);
        assert_eq!(ho.obs, f32_bytes(&hobs), "human obs at tick {}", ho.tick);
        let mask: Vec<u8> = env.human_action_mask().iter().map(|&m| m as u8).collect();
        assert_eq!(ho.mask, mask);
        assert_eq!(zo.num_agents, 3);
        let mut want = Vec::new();
        for i in 0..3 {
            write_zombie_obs(&env, i, &mut zobs);
            want.extend(f32_bytes(&zobs));
        }
        assert_eq!(zo.obs, want, "zombie obs at tick {}", zo.tick);
        let ha = vec![
            rng.random_range(0..MOVE_ACTIONS as u8),
            rng.random_range(0..INTERACT_ACTIONS as u8),
        ];
        let za: Vec<u8> = (0..3)
            .map(|_| rng.random_range(0..MOVE_ACTIONS as u8))
            .collect();
        h.act(ho.episode, ho.tick, ha.clone()).await;
        z.act(zo.episode, zo.tick, za.clone()).await;
        env.step(
            HumanAction {
                movement: ha[0],
                interact: ha[1],
            },
            &za,
        );
    }
    for s in [&mut h, &mut z] {
        let Msg::MatchEnd(end) = s.next().await else {
            panic!("expected MatchEnd")
        };
        assert_eq!(end.status, "finished");
    }
    wait_status(&state, &id, MatchStatus::Finished).await;
    assert!(
        frames.try_recv().is_ok(),
        "the viewer stream still gets frames"
    );
}

#[tokio::test]
async fn remote_human_plays_scripted_zombies_with_rewards() {
    let (mut client, state, _dir) = serve_with(RemoteSettings::default()).await;
    let id = create(&mut client, "remote", "scripted", false).await;
    let mut h = join(&mut client, &id, Side::Human).await;
    assert!(matches!(h.next().await, Msg::Joined(_)));
    assert!(matches!(h.next().await, Msg::Start(_)));
    let mut ticks = 0;
    loop {
        match h.next().await {
            Msg::Obs(o) => {
                assert_eq!(o.reward.len(), 1);
                if o.tick == 0 {
                    assert_eq!(o.reward, vec![0.0]);
                }
                h.act(o.episode, o.tick, vec![0, 0]).await;
                ticks += 1;
            }
            Msg::End(e) => {
                assert_eq!(e.ticks, ticks);
                assert!(!e.replay_id.is_empty());
                break;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    wait_status(&state, &id, MatchStatus::Finished).await;
}

#[tokio::test]
async fn joins_with_another_layout_or_twice_are_refused() {
    let (mut client, _state, _dir) = serve_with(RemoteSettings::default()).await;
    let id = create(&mut client, "remote", "scripted", false).await;
    let stale = join_msg(&id, Side::Human, OBS_VERSION - 1, HUMAN_OBS_DIM);
    let err = play(&mut client, stale).await.err().unwrap();
    assert_eq!(err.code(), Code::FailedPrecondition);
    assert!(err.message().contains("obs_version"), "{}", err.message());
    let wrong_dim = join_msg(&id, Side::Human, OBS_VERSION, HUMAN_OBS_DIM - 1);
    assert_eq!(
        play(&mut client, wrong_dim).await.err().unwrap().code(),
        Code::FailedPrecondition
    );
    let not_remote = join_msg(&id, Side::Zombies, OBS_VERSION, ZOMBIE_OBS_DIM);
    assert_eq!(
        play(&mut client, not_remote).await.err().unwrap().code(),
        Code::AlreadyExists
    );
    let _h = join(&mut client, &id, Side::Human).await;
    let again = join_msg(&id, Side::Human, OBS_VERSION, HUMAN_OBS_DIM);
    assert_eq!(
        play(&mut client, again).await.err().unwrap().code(),
        Code::AlreadyExists
    );
    let missing = join_msg("nope", Side::Human, OBS_VERSION, HUMAN_OBS_DIM);
    assert_eq!(
        play(&mut client, missing).await.err().unwrap().code(),
        Code::NotFound
    );
}

#[tokio::test]
async fn a_side_that_never_joins_fails_the_match() {
    let settings = RemoteSettings {
        join_timeout: Duration::from_millis(200),
        ..RemoteSettings::default()
    };
    let (mut client, state, _dir) = serve_with(settings).await;
    let id = create(&mut client, "remote", "remote", false).await;
    let mut h = join(&mut client, &id, Side::Human).await;
    assert!(matches!(h.next().await, Msg::Joined(_)));
    let Msg::MatchEnd(end) = h.next().await else {
        panic!("expected MatchEnd")
    };
    assert_eq!(end.status, "failed");
    let reason = wait_status(&state, &id, MatchStatus::Failed).await;
    assert_eq!(reason, "remote zombies did not join");
}

#[tokio::test]
async fn a_disconnect_fails_the_match_and_discards_the_episode() {
    let (mut client, state, _dir) = serve_with(RemoteSettings::default()).await;
    let id = create(&mut client, "remote", "remote", false).await;
    let mut h = join(&mut client, &id, Side::Human).await;
    let mut z = join(&mut client, &id, Side::Zombies).await;
    for _ in 0..2 {
        h.next().await;
        z.next().await;
    }
    let Msg::Obs(o) = h.next().await else {
        panic!("expected an observation")
    };
    h.act(o.episode, o.tick, vec![0, 0]).await;
    drop(z);
    let Msg::MatchEnd(end) = h.next().await else {
        panic!("expected MatchEnd")
    };
    assert_eq!(end.status, "failed");
    let reason = wait_status(&state, &id, MatchStatus::Failed).await;
    assert!(reason.contains("zombies disconnected"), "{reason}");
    assert!(
        state.replays.list().is_empty(),
        "no replay of the unfinished episode"
    );
}

#[tokio::test]
async fn invalid_actions_fail_the_match_and_the_remote_cap_holds() {
    let settings = RemoteSettings {
        max_matches: 1,
        ..RemoteSettings::default()
    };
    let (mut client, state, _dir) = serve_with(settings).await;
    let id = create(&mut client, "remote", "scripted", false).await;
    let err = client
        .create_match(CreateMatchRequest {
            human: "remote".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::ResourceExhausted);
    let mut h = join(&mut client, &id, Side::Human).await;
    h.next().await;
    h.next().await;
    let Msg::Obs(o) = h.next().await else {
        panic!("expected an observation")
    };
    h.act(o.episode, o.tick, vec![MOVE_ACTIONS as u8, 0]).await;
    let status = tokio::time::timeout(Duration::from_secs(10), h.rx.message())
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    let reason = wait_status(&state, &id, MatchStatus::Failed).await;
    assert!(reason.contains("out of range"), "{reason}");
    // The failed match no longer counts against the cap.
    create(&mut client, "remote", "scripted", false).await;
}
