from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Side(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SIDE_UNSPECIFIED: _ClassVar[Side]
    SIDE_HUMAN: _ClassVar[Side]
    SIDE_ZOMBIES: _ClassVar[Side]
SIDE_UNSPECIFIED: Side
SIDE_HUMAN: Side
SIDE_ZOMBIES: Side

class GetInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LayoutBlock(_message.Message):
    __slots__ = ("name", "start", "len", "slots", "fields")
    NAME_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    LEN_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    name: str
    start: int
    len: int
    slots: int
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., start: _Optional[int] = ..., len: _Optional[int] = ..., slots: _Optional[int] = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class SideSpec(_message.Message):
    __slots__ = ("obs_dim", "blocks", "action_dims")
    OBS_DIM_FIELD_NUMBER: _ClassVar[int]
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    ACTION_DIMS_FIELD_NUMBER: _ClassVar[int]
    obs_dim: int
    blocks: _containers.RepeatedCompositeFieldContainer[LayoutBlock]
    action_dims: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, obs_dim: _Optional[int] = ..., blocks: _Optional[_Iterable[_Union[LayoutBlock, _Mapping]]] = ..., action_dims: _Optional[_Iterable[int]] = ...) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("version", "protocol", "obs_version", "human", "zombie", "base_config_json")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    OBS_VERSION_FIELD_NUMBER: _ClassVar[int]
    HUMAN_FIELD_NUMBER: _ClassVar[int]
    ZOMBIE_FIELD_NUMBER: _ClassVar[int]
    BASE_CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    version: str
    protocol: int
    obs_version: int
    human: SideSpec
    zombie: SideSpec
    base_config_json: str
    def __init__(self, version: _Optional[str] = ..., protocol: _Optional[int] = ..., obs_version: _Optional[int] = ..., human: _Optional[_Union[SideSpec, _Mapping]] = ..., zombie: _Optional[_Union[SideSpec, _Mapping]] = ..., base_config_json: _Optional[str] = ...) -> None: ...

class CreateMatchRequest(_message.Message):
    __slots__ = ("config_json", "seed", "human", "zombies", "speed", "auto_restart", "realtime", "deterministic")
    CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    HUMAN_FIELD_NUMBER: _ClassVar[int]
    ZOMBIES_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    AUTO_RESTART_FIELD_NUMBER: _ClassVar[int]
    REALTIME_FIELD_NUMBER: _ClassVar[int]
    DETERMINISTIC_FIELD_NUMBER: _ClassVar[int]
    config_json: str
    seed: int
    human: str
    zombies: str
    speed: float
    auto_restart: bool
    realtime: bool
    deterministic: bool
    def __init__(self, config_json: _Optional[str] = ..., seed: _Optional[int] = ..., human: _Optional[str] = ..., zombies: _Optional[str] = ..., speed: _Optional[float] = ..., auto_restart: _Optional[bool] = ..., realtime: _Optional[bool] = ..., deterministic: _Optional[bool] = ...) -> None: ...

class MatchCreated(_message.Message):
    __slots__ = ("match_id",)
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    match_id: str
    def __init__(self, match_id: _Optional[str] = ...) -> None: ...

class ClientMessage(_message.Message):
    __slots__ = ("join", "actions")
    JOIN_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    join: Join
    actions: Actions
    def __init__(self, join: _Optional[_Union[Join, _Mapping]] = ..., actions: _Optional[_Union[Actions, _Mapping]] = ...) -> None: ...

class Join(_message.Message):
    __slots__ = ("match_id", "side", "protocol", "obs_version", "obs_dim", "agent_name")
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    OBS_VERSION_FIELD_NUMBER: _ClassVar[int]
    OBS_DIM_FIELD_NUMBER: _ClassVar[int]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    match_id: str
    side: Side
    protocol: int
    obs_version: int
    obs_dim: int
    agent_name: str
    def __init__(self, match_id: _Optional[str] = ..., side: _Optional[_Union[Side, str]] = ..., protocol: _Optional[int] = ..., obs_version: _Optional[int] = ..., obs_dim: _Optional[int] = ..., agent_name: _Optional[str] = ...) -> None: ...

class Actions(_message.Message):
    __slots__ = ("episode", "tick", "actions")
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    episode: int
    tick: int
    actions: bytes
    def __init__(self, episode: _Optional[int] = ..., tick: _Optional[int] = ..., actions: _Optional[bytes] = ...) -> None: ...

class ServerMessage(_message.Message):
    __slots__ = ("joined", "start", "obs", "end", "match_end")
    JOINED_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    OBS_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    MATCH_END_FIELD_NUMBER: _ClassVar[int]
    joined: Joined
    start: EpisodeStart
    obs: Observation
    end: EpisodeEnd
    match_end: MatchEnd
    def __init__(self, joined: _Optional[_Union[Joined, _Mapping]] = ..., start: _Optional[_Union[EpisodeStart, _Mapping]] = ..., obs: _Optional[_Union[Observation, _Mapping]] = ..., end: _Optional[_Union[EpisodeEnd, _Mapping]] = ..., match_end: _Optional[_Union[MatchEnd, _Mapping]] = ...) -> None: ...

class Joined(_message.Message):
    __slots__ = ("match_id", "side", "realtime", "decision_dt")
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    REALTIME_FIELD_NUMBER: _ClassVar[int]
    DECISION_DT_FIELD_NUMBER: _ClassVar[int]
    match_id: str
    side: Side
    realtime: bool
    decision_dt: float
    def __init__(self, match_id: _Optional[str] = ..., side: _Optional[_Union[Side, str]] = ..., realtime: _Optional[bool] = ..., decision_dt: _Optional[float] = ...) -> None: ...

class EpisodeStart(_message.Message):
    __slots__ = ("episode", "seed", "num_agents", "config_json")
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    NUM_AGENTS_FIELD_NUMBER: _ClassVar[int]
    CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    episode: int
    seed: int
    num_agents: int
    config_json: str
    def __init__(self, episode: _Optional[int] = ..., seed: _Optional[int] = ..., num_agents: _Optional[int] = ..., config_json: _Optional[str] = ...) -> None: ...

class Observation(_message.Message):
    __slots__ = ("episode", "tick", "num_agents", "obs_dim", "obs", "mask", "reward", "state_hash", "time", "deadline_ms")
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    NUM_AGENTS_FIELD_NUMBER: _ClassVar[int]
    OBS_DIM_FIELD_NUMBER: _ClassVar[int]
    OBS_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    STATE_HASH_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    episode: int
    tick: int
    num_agents: int
    obs_dim: int
    obs: bytes
    mask: bytes
    reward: _containers.RepeatedScalarFieldContainer[float]
    state_hash: int
    time: float
    deadline_ms: int
    def __init__(self, episode: _Optional[int] = ..., tick: _Optional[int] = ..., num_agents: _Optional[int] = ..., obs_dim: _Optional[int] = ..., obs: _Optional[bytes] = ..., mask: _Optional[bytes] = ..., reward: _Optional[_Iterable[float]] = ..., state_hash: _Optional[int] = ..., time: _Optional[float] = ..., deadline_ms: _Optional[int] = ...) -> None: ...

class EpisodeEnd(_message.Message):
    __slots__ = ("episode", "outcome", "ticks", "reward", "state_hash", "replay_id")
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    TICKS_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    STATE_HASH_FIELD_NUMBER: _ClassVar[int]
    REPLAY_ID_FIELD_NUMBER: _ClassVar[int]
    episode: int
    outcome: str
    ticks: int
    reward: _containers.RepeatedScalarFieldContainer[float]
    state_hash: int
    replay_id: str
    def __init__(self, episode: _Optional[int] = ..., outcome: _Optional[str] = ..., ticks: _Optional[int] = ..., reward: _Optional[_Iterable[float]] = ..., state_hash: _Optional[int] = ..., replay_id: _Optional[str] = ...) -> None: ...

class MatchEnd(_message.Message):
    __slots__ = ("status", "reason")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    status: str
    reason: str
    def __init__(self, status: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...
