"""Tests for generalized multi-dimensional multiplexing.

Tests cover:
- @serve.multiplexed(name=..., size_per_replica=...) decorator
- serve.get_multiplexed_id(name) function
- handle.options(multiplex_ids={...}) parameter
- HTTP header parsing for serve_multiplexed_{name}_id and x-session-id
- Multi-dimensional routing priority in PowerOfTwoChoicesRequestRouter
- Backward compatibility of legacy API
- Batching with multi-dimensional multiplex IDs
- Per-dimension ID reporting from replicas
"""

import asyncio
import warnings
from typing import Dict, Optional, Set

import pytest

from ray.serve._private.common import (
    DeploymentHandleSource,
    DeploymentID,
    ReplicaID,
    RequestMetadata,
    RequestRoutingInfo,
    RunningReplicaInfo,
)
from ray.serve._private.constants import (
    SERVE_MULTIPLEX_HEADER_PREFIX,
    SERVE_MULTIPLEX_HEADER_SUFFIX,
    SERVE_MULTIPLEXED_MODEL_ID,
    SERVE_SESSION_ID_HEADER,
)
from ray.serve._private.handle_options import DynamicHandleOptions
from ray.serve._private.request_router import (
    PendingRequest,
    PowerOfTwoChoicesRequestRouter,
    RunningReplica,
)
from ray.serve._private.test_utils import MockTimer
from ray.serve._private.utils import generate_request_id
from ray.serve.context import _RequestContext

TIMER = MockTimer()
DEFAULT_MAX_ONGOING_REQUESTS = 10
ROUTER_NODE_ID = "router_node_id"


# ---------------------------------------------------------------------------
# Fake replica for unit tests
# ---------------------------------------------------------------------------
class FakeRunningReplica(RunningReplica):
    def __init__(
        self,
        replica_unique_id: str,
        *,
        node_id: str = "",
        availability_zone: Optional[str] = None,
        model_ids: Optional[Set[str]] = None,
        dimension_ids: Optional[Dict[str, Set[str]]] = None,
        max_ongoing_requests: int = DEFAULT_MAX_ONGOING_REQUESTS,
    ):
        self._replica_id = ReplicaID(
            unique_id=replica_unique_id,
            deployment_id=DeploymentID(name="TEST_DEPLOYMENT"),
        )
        self._node_id = node_id
        self._availability_zone = availability_zone
        self._queue_len = 0
        self._max_ongoing_requests = max_ongoing_requests
        self._dimension_ids: Dict[str, Set[str]] = dimension_ids or {}
        # Convenience: callers passing model_ids get merged into the "model"
        # dimension. multiplex_dim_to_ids is the canonical storage.
        if model_ids:
            self._dimension_ids.setdefault("model", set()).update(model_ids)
        self._has_queue_len_response = asyncio.Event()

    @property
    def replica_id(self) -> ReplicaID:
        return self._replica_id

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def availability_zone(self) -> Optional[str]:
        return self._availability_zone

    @property
    def multiplex_dim_to_ids(self) -> Dict[str, Set[str]]:
        return self._dimension_ids

    @property
    def max_ongoing_requests(self) -> int:
        return self._max_ongoing_requests

    def update_replica_info(self, replica_info: RunningReplicaInfo) -> None:
        self._dimension_ids = {
            dim: set(ids) for dim, ids in replica_info.multiplex_dim_to_ids.items()
        }

    def set_queue_len_response(self, queue_len: int):
        self._queue_len = queue_len
        self._has_queue_len_response.set()

    def push_proxy_handle(self, handle):
        pass

    async def get_queue_len(self, *, deadline_s: float) -> int:
        while not self._has_queue_len_response.is_set():
            await self._has_queue_len_response.wait()
        return self._queue_len

    def try_send_request(self, pr, with_rejection):
        raise NotImplementedError()

    def send_request_with_rejection(self, pr):
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pending_request(
    model_id: str = "",
    multiplex_ids: Optional[Dict[str, str]] = None,
) -> PendingRequest:
    # Merge legacy-style `model_id` into multiplex_ids for ergonomics in tests.
    ids = dict(multiplex_ids) if multiplex_ids else {}
    if model_id and "model" not in ids:
        ids["model"] = model_id
    return PendingRequest(
        args=[],
        kwargs={},
        metadata=RequestMetadata(
            request_id=generate_request_id(),
            internal_request_id=generate_request_id(),
            multiplex_ids=ids,
        ),
    )


def _make_router(**kwargs) -> PowerOfTwoChoicesRequestRouter:
    return PowerOfTwoChoicesRequestRouter(
        deployment_id=DeploymentID(name="TEST_DEPLOYMENT"),
        handle_source=DeploymentHandleSource.REPLICA,
        prefer_local_node_routing=kwargs.get("prefer_local_node", False),
        prefer_local_az_routing=kwargs.get("prefer_local_az", False),
        self_node_id=ROUTER_NODE_ID,
        self_actor_id="fake-actor-id",
        self_actor_handle=None,
        self_availability_zone=kwargs.get("az", None),
        get_curr_time_s=TIMER,
        create_replica_wrapper_func=lambda info: FakeRunningReplica(
            info.replica_id.unique_id,
            dimension_ids={d: set(ids) for d, ids in info.multiplex_dim_to_ids.items()},
        ),
    )


# =====================================================================
# 1. Data structure tests
# =====================================================================
class TestRequestMetadataMultiplexIds:
    def test_multiplex_ids_field_exists(self):
        md = RequestMetadata(
            request_id="r1",
            internal_request_id="ir1",
            multiplex_ids={"model": "lora_v1", "session": "user_1"},
        )
        assert md.multiplex_ids == {"model": "lora_v1", "session": "user_1"}
        # Derived property returns the "model" dimension's ID.
        assert md.multiplexed_model_id == "lora_v1"

    def test_multiplexed_model_id_property_defaults_empty(self):
        """When `multiplex_ids` has no "model" key, the derived property is ''."""
        md = RequestMetadata(
            request_id="r1",
            internal_request_id="ir1",
            multiplex_ids={"session": "user_1"},
        )
        assert md.multiplexed_model_id == ""


class TestRequestContextMultiplexIds:
    def test_multiplex_ids_field(self):
        ctx = _RequestContext(
            multiplex_ids={"model": "lora_v1", "session": "user_1"},
        )
        assert ctx.multiplex_ids == {"model": "lora_v1", "session": "user_1"}

    def test_default_empty(self):
        ctx = _RequestContext()
        assert ctx.multiplex_ids == {}


class TestDynamicHandleOptionsMultiplexIds:
    def test_multiplex_ids_field(self):
        opts = DynamicHandleOptions(multiplex_ids={"model": "m1", "session": "s1"})
        assert opts.multiplex_ids == {"model": "m1", "session": "s1"}

    def test_copy_and_update_preserves_multiplex_ids(self):
        opts = DynamicHandleOptions(multiplex_ids={"model": "m1"})
        updated = opts.copy_and_update(multiplex_ids={"model": "m2", "session": "s1"})
        assert updated.multiplex_ids == {"model": "m2", "session": "s1"}


class TestRunningReplicaInfoMultiplexDimensions:
    def test_multiplex_dim_to_ids(self):
        replica_id = ReplicaID(
            unique_id="r1",
            deployment_id=DeploymentID(name="TEST"),
        )
        info = RunningReplicaInfo(
            replica_id=replica_id,
            node_id="n1",
            node_ip="1.2.3.4",
            availability_zone=None,
            actor_name="actor_r1",
            max_ongoing_requests=10,
            multiplex_dim_to_ids={
                "model": ["lora_v1", "lora_v2"],
                "session": ["user_1"],
            },
        )
        assert info.multiplex_dim_to_ids == {
            "model": ["lora_v1", "lora_v2"],
            "session": ["user_1"],
        }

    def test_hash_includes_dimension_ids(self):
        """Two infos with different dimension_to_ids should hash differently."""
        replica_id = ReplicaID(
            unique_id="r1",
            deployment_id=DeploymentID(name="TEST"),
        )
        info1 = RunningReplicaInfo(
            replica_id=replica_id,
            node_id="n1",
            node_ip="1.2.3.4",
            availability_zone=None,
            actor_name="actor_r1",
            max_ongoing_requests=10,
            multiplex_dim_to_ids={"model": ["m1"]},
        )
        info2 = RunningReplicaInfo(
            replica_id=replica_id,
            node_id="n1",
            node_ip="1.2.3.4",
            availability_zone=None,
            actor_name="actor_r1",
            max_ongoing_requests=10,
            multiplex_dim_to_ids={"model": ["m1", "m2"]},
        )
        assert hash(info1) != hash(info2)


class TestRequestRoutingInfoDimensions:
    def test_multiplex_dim_to_ids_field(self):
        replica_id = ReplicaID(
            unique_id="r1",
            deployment_id=DeploymentID(name="TEST"),
        )
        info = RequestRoutingInfo(
            replica_id=replica_id,
            multiplex_dim_to_ids={"model": ["m1"], "session": ["s1"]},
        )
        assert info.multiplex_dim_to_ids == {
            "model": ["m1"],
            "session": ["s1"],
        }


class TestMultiplexedLoadLatencyBucketsEnvVar:
    """Verify env-var resolution for `MULTIPLEXED_LOAD_LATENCY_BUCKETS_MS`:
    prefers the new `RAY_SERVE_MULTIPLEXED_LOAD_LATENCY_BUCKETS_MS`; falls
    back to the legacy `*_MODEL_LOAD_*` names with a deprecation warning."""

    def test_prefers_new_env_var(self, monkeypatch):
        from ray.serve._private import constants

        monkeypatch.setenv("RAY_SERVE_MULTIPLEXED_LOAD_LATENCY_BUCKETS_MS", "1,2,3")
        monkeypatch.setenv("RAY_SERVE_MODEL_LOAD_LATENCY_BUCKETS_MS", "100,200")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = constants._resolve_multiplexed_load_latency_buckets_env()

        assert result == "1,2,3"
        # No deprecation warning when the new var is used.
        assert not any("deprecated" in str(x.message) for x in w)

    def test_legacy_prefixed_warns(self, monkeypatch):
        from ray.serve._private import constants

        monkeypatch.delenv(
            "RAY_SERVE_MULTIPLEXED_LOAD_LATENCY_BUCKETS_MS", raising=False
        )
        monkeypatch.setenv("RAY_SERVE_MODEL_LOAD_LATENCY_BUCKETS_MS", "5,10")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = constants._resolve_multiplexed_load_latency_buckets_env()

        assert result == "5,10"
        matching = [
            x
            for x in w
            if "RAY_SERVE_MODEL_LOAD_LATENCY_BUCKETS_MS" in str(x.message)
            and "deprecated" in str(x.message)
        ]
        assert len(matching) == 1

    def test_legacy_unprefixed_warns(self, monkeypatch):
        from ray.serve._private import constants

        monkeypatch.delenv(
            "RAY_SERVE_MULTIPLEXED_LOAD_LATENCY_BUCKETS_MS", raising=False
        )
        monkeypatch.delenv("RAY_SERVE_MODEL_LOAD_LATENCY_BUCKETS_MS", raising=False)
        monkeypatch.setenv("MODEL_LOAD_LATENCY_BUCKETS_MS", "1,2")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = constants._resolve_multiplexed_load_latency_buckets_env()

        assert result == "1,2"
        # The unprefixed var also triggers a separate pre-existing
        # FutureWarning from get_env_str about the missing RAY_SERVE_ prefix.
        # We only care that our own UserWarning also fires pointing users at
        # the new generalized name.
        matching = [
            x
            for x in w
            if "RAY_SERVE_MULTIPLEXED_LOAD_LATENCY_BUCKETS_MS" in str(x.message)
        ]
        assert len(matching) == 1

    def test_no_env_returns_empty(self, monkeypatch):
        from ray.serve._private import constants

        for name in (
            "RAY_SERVE_MULTIPLEXED_LOAD_LATENCY_BUCKETS_MS",
            "RAY_SERVE_MODEL_LOAD_LATENCY_BUCKETS_MS",
            "MODEL_LOAD_LATENCY_BUCKETS_MS",
        ):
            monkeypatch.delenv(name, raising=False)

        assert constants._resolve_multiplexed_load_latency_buckets_env() == ""


# =====================================================================
# 2. Decorator tests (api.py)
# =====================================================================
class TestMultiplexedDecorator:
    def test_new_api_name_and_size_per_replica(self):
        """@serve.multiplexed(name=..., size_per_replica=...) works."""
        import ray.serve.api as api

        @api.multiplexed(name="session", size_per_replica=10)
        async def load_session(session_id: str):
            return session_id

        # The wrapper function should be created.
        assert asyncio.iscoroutinefunction(load_session)

    def test_no_name_falls_through_to_legacy_single_dim_default(self):
        """User-facing backward compat: calls that omit both `name` and
        `max_num_models_per_replica` (bare form, empty parens, or just
        `size_per_replica`) must not raise — they silently route to the
        legacy single-dimension path (`name="model"`, legacy metric names)
        so master's `@serve.multiplexed` examples keep working.
        """
        import ray.serve.api as api

        # Bare decorator: @serve.multiplexed (no parens).
        @api.multiplexed
        async def _bare(model_id: str):
            return model_id

        # Empty parens: @serve.multiplexed().
        @api.multiplexed()
        async def _parens(model_id: str):
            return model_id

        # size_per_replica without `name` — no error.
        @api.multiplexed(size_per_replica=5)
        async def _size_only(model_id: str):
            return model_id

        # All three decorate to the legacy "model" dimension.
        assert _bare._serve_multiplex_dimension == "model"
        assert _parens._serve_multiplex_dimension == "model"
        assert _size_only._serve_multiplex_dimension == "model"

    def test_name_must_be_nonempty_string(self):
        import ray.serve.api as api

        with pytest.raises(ValueError, match="name must be a non-empty string"):
            api.multiplexed(name="", size_per_replica=3)

    def test_size_per_replica_validation(self):
        import ray.serve.api as api

        with pytest.raises(TypeError, match="size_per_replica must be an integer"):
            api.multiplexed(name="model", size_per_replica="invalid")

        with pytest.raises(ValueError, match="size_per_replica must be positive"):
            api.multiplexed(name="model", size_per_replica=0)


# =====================================================================
# 3. get_multiplexed_id tests
# =====================================================================
class TestGetMultiplexId:
    def test_get_multiplexed_id_from_request_context(self):
        """get_multiplexed_id reads from the request context's multiplex_ids."""
        import ray.serve.context as ctx

        ctx._serve_request_context.set(
            _RequestContext(multiplex_ids={"model": "lora_v1", "session": "user_1"})
        )
        try:
            from ray.serve.api import get_multiplexed_id

            assert get_multiplexed_id("model") == "lora_v1"
            assert get_multiplexed_id("session") == "user_1"
            assert get_multiplexed_id("nonexistent") == ""
        finally:
            ctx._serve_request_context.set(None)

    def test_get_multiplexed_id_from_batch_context(self):
        """get_multiplexed_id reads from batch request context when in batch."""
        import ray.serve.context as ctx

        batch_contexts = [
            _RequestContext(multiplex_ids={"model": "m1", "session": "s1"}),
        ]
        ctx._serve_batch_request_context.set(batch_contexts)
        try:
            from ray.serve.api import get_multiplexed_id

            assert get_multiplexed_id("model") == "m1"
            assert get_multiplexed_id("session") == "s1"
        finally:
            ctx._serve_batch_request_context.set(None)


class TestGetMultiplexIdDimensionValidation:
    """Verify get_multiplexed_id distinguishes two kinds of "unset":
    - typo / dimension never registered via @serve.multiplexed → raises
    - dimension registered but this request didn't carry an ID → returns ""
    """

    def setup_method(self, method):
        """Snapshot global context state so teardown can restore it and
        avoid bleeding test state into sibling test modules."""
        import ray.serve.context as ctx

        self._saved_replica_ctx = ctx._INTERNAL_REPLICA_CONTEXT
        self._saved_request_ctx = ctx._serve_request_context.get()
        self._saved_batch_ctx = ctx._serve_batch_request_context.get()

    def _install_servable_with_dimensions(self, dimensions):
        """Set up a fake replica context whose servable_object exposes
        methods tagged with the given multiplex dimensions, mirroring
        _set_internal_replica_context's eager population of
        _multiplex_dimensions."""
        import ray.serve.context as ctx
        from ray.serve._private.common import DeploymentID, ReplicaID

        class FakeServable:
            pass

        servable = FakeServable()
        # Each tagged method is just a callable with the sentinel attribute.
        for dim in dimensions:

            async def _loader(x):
                return x

            _loader._serve_multiplex_dimension = dim
            setattr(servable, f"load_{dim}", _loader)

        replica_id = ReplicaID(unique_id="r1", deployment_id=DeploymentID(name="TEST"))
        # ReplicaContext is a dataclass — construct directly, bypassing the
        # _set_internal_replica_context helper that expects other fields.
        # Populate _multiplex_dimensions the same way _set_internal_replica_context
        # would have via _collect_multiplex_dimensions.
        ctx._INTERNAL_REPLICA_CONTEXT = ctx.ReplicaContext(
            replica_id=replica_id,
            servable_object=servable,
            _deployment_config=None,
            rank=0,
            world_size=1,
            _multiplex_dimensions=ctx._collect_multiplex_dimensions(servable),
        )

    def teardown_method(self, method):
        """Restore the pre-test snapshot so other test modules see
        whatever global state they originally set up."""
        import ray.serve.context as ctx

        ctx._INTERNAL_REPLICA_CONTEXT = self._saved_replica_ctx
        ctx._serve_request_context.set(self._saved_request_ctx)
        ctx._serve_batch_request_context.set(self._saved_batch_ctx)

    def test_typo_dimension_raises_key_error(self):
        """Calling get_multiplexed_id with an unregistered name raises."""
        import ray.serve.context as ctx
        from ray.serve.api import get_multiplexed_id

        self._install_servable_with_dimensions({"model", "session"})
        ctx._serve_request_context.set(_RequestContext(multiplex_ids={"model": "m1"}))

        with pytest.raises(KeyError, match="'sesion' is not a registered"):
            get_multiplexed_id("sesion")  # typo for "session"

    def test_registered_but_unset_returns_empty_string(self):
        """A dimension registered on the deployment but missing from the
        current request's multiplex_ids returns '' — not an error."""
        import ray.serve.context as ctx
        from ray.serve.api import get_multiplexed_id

        self._install_servable_with_dimensions({"model", "session"})
        # Request carries model but not session — legitimate "first turn"
        # case.
        ctx._serve_request_context.set(_RequestContext(multiplex_ids={"model": "m1"}))

        assert get_multiplexed_id("model") == "m1"
        assert get_multiplexed_id("session") == ""

    def test_outside_replica_is_permissive(self):
        """When there's no replica context (e.g. unit tests, proxy-side
        utility usage), the dimension-registration check is skipped."""
        import ray.serve.context as ctx
        from ray.serve.api import get_multiplexed_id

        # No replica context installed.
        ctx._INTERNAL_REPLICA_CONTEXT = None
        ctx._serve_request_context.set(_RequestContext(multiplex_ids={"model": "m1"}))

        # Would raise if the check ran; instead returns "".
        assert get_multiplexed_id("anything") == ""
        assert get_multiplexed_id("model") == "m1"

    def test_dimensions_populated_at_startup_not_re_scanned(self):
        """_multiplex_dimensions is populated once when the replica context
        is set; dimensions added to the servable afterwards are not picked
        up (per-call re-scanning would be wasteful)."""
        import ray.serve.context as ctx
        from ray.serve.api import get_multiplexed_id

        self._install_servable_with_dimensions({"model"})
        ctx._serve_request_context.set(_RequestContext())

        replica_ctx = ctx._INTERNAL_REPLICA_CONTEXT
        # Populated eagerly at replica-context setup — not None, not empty.
        assert replica_ctx._multiplex_dimensions == {"model"}

        # Mutate servable after startup — the registered set does not update.
        async def sneak_loader(x):
            return x

        sneak_loader._serve_multiplex_dimension = "sneak"
        replica_ctx.servable_object.load_sneak = sneak_loader

        # "sneak" appears unregistered because startup-time snapshot is
        # authoritative.
        with pytest.raises(KeyError):
            get_multiplexed_id("sneak")


# =====================================================================
# 4. HTTP header parsing tests
# =====================================================================
class TestHTTPHeaderParsing:
    """Test that the proxy correctly parses multiplex headers."""

    def test_serve_multiplexed_model_id_header(self):
        """serve_multiplexed_model_id -> multiplex_ids['model']."""
        header_key = (
            f"{SERVE_MULTIPLEX_HEADER_PREFIX}model{SERVE_MULTIPLEX_HEADER_SUFFIX}"
        )
        assert header_key == "serve_multiplexed_model_id"

    def test_serve_multiplexed_session_id_header(self):
        """serve_multiplexed_session_id -> multiplex_ids['session']."""
        header_key = (
            f"{SERVE_MULTIPLEX_HEADER_PREFIX}session{SERVE_MULTIPLEX_HEADER_SUFFIX}"
        )
        assert header_key == "serve_multiplexed_session_id"

    def test_legacy_model_header_derived_from_pattern(self):
        """The long-standing `serve_multiplexed_model_id` header is now a
        natural case of the generalized pattern (dimension="model")."""
        assert SERVE_MULTIPLEXED_MODEL_ID == "serve_multiplexed_model_id"
        assert SERVE_MULTIPLEXED_MODEL_ID == (
            f"{SERVE_MULTIPLEX_HEADER_PREFIX}model{SERVE_MULTIPLEX_HEADER_SUFFIX}"
        )

    def test_session_header_constant(self):
        """x_session_id header is recognized."""
        assert SERVE_SESSION_ID_HEADER == "x_session_id"


class TestParseMultiplexIdsFromHeaders:
    """Shared helper used by both the proxy HTTP path and the replica's
    direct-ingress (WebSocket) path to lift multiplex headers into a dict."""

    def _parse(self, headers):
        from ray.serve._private.common import parse_multiplex_ids_from_headers

        # Callers pass (bytes, bytes) pairs (ASGI scope format).
        return parse_multiplex_ids_from_headers(
            [(k.encode(), v.encode()) for k, v in headers]
        )

    def test_generalized_pattern_multiple_dimensions(self):
        ids = self._parse(
            [
                ("serve_multiplexed_model_id", "lora_v2"),
                ("serve_multiplexed_session_id", "user_123"),
            ]
        )
        assert ids == {"model": "lora_v2", "session": "user_123"}

    def test_legacy_header_covered_by_generalized_pattern(self):
        """serve_multiplexed_model_id still maps to dimension='model'."""
        ids = self._parse([("serve_multiplexed_model_id", "v1")])
        assert ids == {"model": "v1"}

    def test_dash_form_normalized_to_underscore(self):
        """Fronting proxies (nginx, AWS API GW) sometimes turn `_` → `-`
        in header names. Normalization collapses both to the same key."""
        ids = self._parse(
            [
                ("Serve-Multiplexed-Model-Id", "m1"),
                ("X-Session-Id", "s1"),
            ]
        )
        assert ids == {"model": "m1", "session": "s1"}

    def test_conventional_session_header_shortcut(self):
        ids = self._parse([("x-session-id", "user_abc")])
        assert ids == {"session": "user_abc"}

    def test_explicit_session_dim_wins_over_x_session_id(self):
        """If both the generalized `serve_multiplexed_session_id` and the
        shortcut `x-session-id` are present, the explicit dimension wins."""
        ids = self._parse(
            [
                ("serve_multiplexed_session_id", "explicit"),
                ("x-session-id", "fallback"),
            ]
        )
        assert ids == {"session": "explicit"}

    def test_unknown_headers_ignored(self):
        ids = self._parse(
            [
                ("content-type", "application/json"),
                ("authorization", "Bearer xyz"),
            ]
        )
        assert ids == {}

    def test_empty_dimension_name_rejected(self):
        """`serve_multiplexed__id` has an empty dimension — silently dropped
        rather than adding a `""` key to the dict."""
        ids = self._parse([("serve_multiplexed__id", "v")])
        assert ids == {}

    def test_empty_headers(self):
        assert self._parse([]) == {}


# =====================================================================
# 5. Router multi-dimensional routing tests
# =====================================================================
@pytest.mark.asyncio
class TestMultiplexMixinDimensions:
    async def test_update_multiplexed_model_ids_builds_dimension_map(self):
        """_update_multiplexed_model_ids_with_replicas builds per-dim map."""
        router = _make_router()
        replica1 = FakeRunningReplica(
            "r1",
            model_ids={"m1"},
            dimension_ids={"model": {"m1"}, "session": {"s1", "s2"}},
        )
        replica2 = FakeRunningReplica(
            "r2",
            model_ids={"m2"},
            dimension_ids={"model": {"m2"}, "session": {"s3"}},
        )
        replica1.set_queue_len_response(0)
        replica2.set_queue_len_response(0)
        router.update_replicas([replica1, replica2])

        # Legacy map should be populated.
        model_map = router._multiplex_dim_id_to_replica_ids["model"]
        assert replica1.replica_id in model_map["m1"]
        assert replica2.replica_id in model_map["m2"]

        # New per-dimension map should be populated.
        dim_map = router._multiplex_dim_id_to_replica_ids
        assert replica1.replica_id in dim_map["model"]["m1"]
        assert replica2.replica_id in dim_map["model"]["m2"]
        assert replica1.replica_id in dim_map["session"]["s1"]
        assert replica1.replica_id in dim_map["session"]["s2"]
        assert replica2.replica_id in dim_map["session"]["s3"]


@pytest.mark.asyncio
class TestPow2RouterMultiDimensional:
    async def test_model_only_routing(self):
        """Requests with a "model" dim route to replicas that have it cached."""
        router = _make_router()
        replica1 = FakeRunningReplica("r1", model_ids={"m1"})
        replica2 = FakeRunningReplica("r2", model_ids={"m2"})
        replica1.set_queue_len_response(0)
        replica2.set_queue_len_response(0)
        router.update_replicas([replica1, replica2])

        # Request for m1 should prefer replica1.
        pr = _make_pending_request(model_id="m1")
        result = await router.choose_replicas([], pr)
        assert len(result) == 1
        chosen_ids = {r.replica_id for r in result[0]}
        assert replica1.replica_id in chosen_ids

    async def test_no_multiplex_ids_uses_locality(self):
        """Requests with no multiplex IDs use locality routing."""
        router = _make_router(prefer_local_node=True)
        local_replica = FakeRunningReplica("r1", node_id=ROUTER_NODE_ID)
        remote_replica = FakeRunningReplica("r2", node_id="other_node")
        local_replica.set_queue_len_response(0)
        remote_replica.set_queue_len_response(0)
        router.update_replicas([local_replica, remote_replica])

        pr = _make_pending_request()
        result = await router.choose_replicas([], pr)
        assert len(result) == 1
        chosen_ids = {r.replica_id for r in result[0]}
        # Should prefer local replica.
        assert local_replica.replica_id in chosen_ids


# =====================================================================
# 6. Batching split tests
# =====================================================================
class TestBatchSplitByMultiplexIds:
    def test_split_by_multiplex_ids(self):
        """Batches are split by full multiplex identity."""
        from ray.serve.batching import _BatchQueue

        bq = _BatchQueue.__new__(_BatchQueue)

        # Create fake requests with different multiplex identities.
        class FakeRequest:
            def __init__(self, multiplex_ids=None, model_id=""):
                ids = dict(multiplex_ids) if multiplex_ids else {}
                if model_id and "model" not in ids:
                    ids["model"] = model_id
                self.request_context = _RequestContext(multiplex_ids=ids)

        batch = [
            FakeRequest(multiplex_ids={"model": "m1", "session": "s1"}),
            FakeRequest(multiplex_ids={"model": "m1", "session": "s1"}),
            FakeRequest(multiplex_ids={"model": "m1", "session": "s2"}),
            FakeRequest(multiplex_ids={"model": "m2", "session": "s1"}),
        ]

        sub_batches = bq._split_batch_by_multiplex_ids(batch)
        assert len(sub_batches) == 3

    def test_split_legacy_model_id(self):
        """Legacy model_id-only batches still work."""
        from ray.serve.batching import _BatchQueue

        bq = _BatchQueue.__new__(_BatchQueue)

        class FakeRequest:
            def __init__(self, model_id=""):
                self.request_context = _RequestContext(
                    multiplex_ids={"model": model_id} if model_id else {},
                )

        batch = [
            FakeRequest(model_id="m1"),
            FakeRequest(model_id="m1"),
            FakeRequest(model_id="m2"),
        ]

        sub_batches = bq._split_batch_by_multiplex_ids(batch)
        assert len(sub_batches) == 2

    def test_split_mixed_dimension_cardinality(self):
        """Requests with different subsets of multiplex dimensions must not
        be merged into the same sub-batch — a batched function that reads a
        dimension present in one request but missing from another would get
        a silently-wrong value for the second.

        Given three requests sharing a replica:
          A: {model=lora1, session=2}
          B: {model=lora1}              (no session)
          C: {session=3}                (no model)

        The split must produce 3 distinct sub-batches so that
        get_multiplexed_id("session") on A's batch returns "2", on B's
        returns "", and on C's returns "3".
        """
        from ray.serve.batching import _BatchQueue

        bq = _BatchQueue.__new__(_BatchQueue)

        class FakeRequest:
            def __init__(self, multiplex_ids):
                self.request_context = _RequestContext(
                    multiplex_ids=multiplex_ids,
                )

        req_a = FakeRequest({"model": "lora1", "session": "2"})
        req_b = FakeRequest({"model": "lora1"})
        req_c = FakeRequest({"session": "3"})

        sub_batches = bq._split_batch_by_multiplex_ids([req_a, req_b, req_c])

        assert len(sub_batches) == 3
        # Each sub-batch contains exactly one request with its original IDs.
        by_ids = {
            frozenset(sb[0].request_context.multiplex_ids.items()): sb
            for sb in sub_batches
        }
        assert len(by_ids) == 3
        assert by_ids[frozenset({("model", "lora1"), ("session", "2")})] == [req_a]
        assert by_ids[frozenset({("model", "lora1")})] == [req_b]
        assert by_ids[frozenset({("session", "3")})] == [req_c]

    def test_split_mixed_cardinality_groups_duplicates(self):
        """When multiple requests share the same multiplex identity, they
        are grouped together even if other requests in the batch have
        different dimension subsets."""
        from ray.serve.batching import _BatchQueue

        bq = _BatchQueue.__new__(_BatchQueue)

        class FakeRequest:
            def __init__(self, multiplex_ids):
                self.request_context = _RequestContext(
                    multiplex_ids=multiplex_ids,
                )

        # Two requests with {model=lora1, session=2}, one with {model=lora1},
        # one with {session=3}. Expect 3 sub-batches with sizes 2, 1, 1.
        batch = [
            FakeRequest({"model": "lora1", "session": "2"}),
            FakeRequest({"model": "lora1", "session": "2"}),
            FakeRequest({"model": "lora1"}),
            FakeRequest({"session": "3"}),
        ]

        sub_batches = bq._split_batch_by_multiplex_ids(batch)

        assert len(sub_batches) == 3
        assert sorted(len(sb) for sb in sub_batches) == [1, 1, 2]


# =====================================================================
# 6b. Metric naming tests
# =====================================================================
class TestMetricNaming:
    """Verify the wrapper emits legacy metric names when constructed via
    the deprecated max_num_models_per_replica path, and the generalized
    `dimension`-tagged names otherwise."""

    def _make_wrapper(
        self, *, multiplex_model_legacy_metrics: bool, name: str = "model"
    ):
        from unittest.mock import MagicMock, patch

        from ray.serve.multiplex import _MultiplexWrapper

        fake_ctx = MagicMock()
        fake_ctx.app_name = "app"
        fake_ctx.deployment = "dep"
        fake_ctx.replica_id = MagicMock()

        with patch(
            "ray.serve.multiplex._get_internal_replica_context",
            return_value=fake_ctx,
        ), patch("ray.serve.multiplex.MetricsPusher"):

            async def load(x):
                return x

            return _MultiplexWrapper(
                load,
                None,
                size_per_replica=3,
                name=name,
                multiplex_model_legacy_metrics=multiplex_model_legacy_metrics,
            )

    def test_multiplex_model_legacy_metrics_use_pre_generalization_names(self):
        wrapper = self._make_wrapper(multiplex_model_legacy_metrics=True)
        # No `dimension` tag populated.
        assert wrapper._metric_dimension_tag == {}
        # Histograms/Gauges/Counters expose a `_name` attribute on Ray metrics.
        # Check a representative sample for the old names.
        assert "serve_multiplexed_model_load_latency_ms" in str(
            wrapper.load_latency_ms._name
        )
        assert "serve_num_multiplexed_models" in str(wrapper.num_entries_gauge._name)

    def test_new_metrics_use_dimension_tag_for_model(self):
        wrapper = self._make_wrapper(multiplex_model_legacy_metrics=False, name="model")
        assert wrapper._metric_dimension_tag == {"dimension": "model"}
        assert "serve_multiplexed_load_latency_ms" in str(wrapper.load_latency_ms._name)
        assert "serve_multiplexed_num_entries" in str(wrapper.num_entries_gauge._name)

    def test_new_metrics_use_dimension_tag_for_session(self):
        wrapper = self._make_wrapper(
            multiplex_model_legacy_metrics=False, name="session"
        )
        assert wrapper._metric_dimension_tag == {"dimension": "session"}
        # Same metric name as "model" — distinguished only by tag.
        assert "serve_multiplexed_load_latency_ms" in str(wrapper.load_latency_ms._name)


# =====================================================================
# 7. Handle options tests
# =====================================================================
class TestHandleOptionsMultiplexIds:
    def test_multiplex_ids_in_dynamic_handle_options(self):
        opts = DynamicHandleOptions(multiplex_ids={"model": "m1", "session": "s1"})
        assert opts.multiplex_ids["model"] == "m1"
        assert opts.multiplex_ids["session"] == "s1"

    def test_copy_and_update_replaces_multiplex_ids(self):
        opts = DynamicHandleOptions(multiplex_ids={"model": "m1"})
        new_opts = opts.copy_and_update(multiplex_ids={"model": "m2", "session": "s1"})
        assert new_opts.multiplex_ids == {"model": "m2", "session": "s1"}
        # Original is unchanged (frozen).
        assert opts.multiplex_ids == {"model": "m1"}

    def test_default_multiplex_ids_is_empty(self):
        opts = DynamicHandleOptions()
        assert opts.multiplex_ids == {}


# =====================================================================
# Legacy API backward-compatibility tests.
#
# These verify the pre-RFC-#62645 public surface continues to work
# during the deprecation window (Ray 2.56 - 2.57). The entire class
# below is scheduled for deletion in Ray 2.58 alongside the production
# shims it exercises. Grep for "Ray 2.58" to find all corresponding
# removal points (production warnings.warn messages and this class).
# =====================================================================
class TestLegacyMultiplexedModelIdBackwardCompat:
    """Pre-RFC #62645 API: `@serve.multiplexed(max_num_models_per_replica=N)`,
    `serve.get_multiplexed_model_id()`, and
    `handle.options(multiplexed_model_id=...)`. DELETE THIS ENTIRE CLASS
    when those surfaces are removed in Ray 2.58.
    """

    # ---- @serve.multiplexed(max_num_models_per_replica=N) -------------
    def test_max_num_models_per_replica_emits_deprecation_warning(self):
        import ray.serve.api as api

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @api.multiplexed(max_num_models_per_replica=5)
            async def load_model(model_id: str):
                return model_id

            matching = [x for x in w if "max_num_models_per_replica" in str(x.message)]
            assert len(matching) == 1

    def test_max_num_models_per_replica_defaults_name_to_model(self):
        """Legacy path auto-sets name='model' (implicit behavior preserved)."""
        import ray.serve.api as api

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            @api.multiplexed(max_num_models_per_replica=3)
            async def load_model(model_id: str):
                return model_id

        assert asyncio.iscoroutinefunction(load_model)

    def test_max_num_models_per_replica_conflicts_with_name(self):
        """Mixing the legacy kwarg with the new `name` kwarg is an error."""
        import ray.serve.api as api

        with pytest.raises(ValueError, match="cannot be combined"):

            @api.multiplexed(max_num_models_per_replica=3, name="model")
            async def load_model(model_id: str):
                return model_id

    def test_max_num_models_per_replica_validation_uses_legacy_name(self):
        """Validation errors reference the user-facing param name."""
        import ray.serve.api as api

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with pytest.raises(
                TypeError, match="max_num_models_per_replica must be an integer"
            ):
                api.multiplexed(max_num_models_per_replica="invalid")

            with pytest.raises(
                ValueError, match="max_num_models_per_replica must be positive"
            ):
                api.multiplexed(max_num_models_per_replica=0)

    # ---- serve.get_multiplexed_model_id() ------------------------------
    def test_get_multiplexed_model_id_still_works_with_warning(self):
        import ray.serve.context as ctx

        ctx._serve_request_context.set(
            _RequestContext(multiplex_ids={"model": "lora_v1", "session": "user_1"})
        )
        try:
            from ray.serve.api import get_multiplexed_model_id

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = get_multiplexed_model_id()
                assert result == "lora_v1"
                matching = [
                    x for x in w if "get_multiplexed_model_id" in str(x.message)
                ]
                assert len(matching) == 1
        finally:
            ctx._serve_request_context.set(None)

    # ---- handle.options(multiplexed_model_id=...) ----------------------
    def _make_handle(self):
        """Build a DeploymentHandle without a running Serve cluster.
        `.options()` only manipulates handle-local state; we fake out the
        router so `.is_initialized` is True and `_init()` (which would
        trigger Ray auto-init) is not called."""
        from unittest.mock import MagicMock

        from ray.serve._private.handle_options import (
            DynamicHandleOptions,
            InitHandleOptions,
        )
        from ray.serve.handle import DeploymentHandle

        handle = DeploymentHandle(
            "test_deployment",
            "test_app",
            init_options=InitHandleOptions(),
            handle_options=DynamicHandleOptions(),
            _handle_id="test_id",
        )
        handle._router = MagicMock()
        return handle

    def test_handle_options_multiplexed_model_id_bridges_with_warning(self):
        handle = self._make_handle()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            new_handle = handle.options(multiplexed_model_id="lora_v2")

        # Bridged into the new field; no scalar leftover.
        assert new_handle.handle_options.multiplex_ids == {"model": "lora_v2"}
        assert new_handle.handle_options.multiplexed_model_id == ""
        matching = [
            x
            for x in w
            if "multiplexed_model_id" in str(x.message)
            and "deprecated" in str(x.message)
        ]
        assert len(matching) == 1

    def test_handle_options_legacy_and_new_kwargs_together_raises(self):
        handle = self._make_handle()
        with pytest.raises(ValueError, match="Cannot specify both"):
            handle.options(multiplexed_model_id="a", multiplex_ids={"model": "b"})

    def test_handle_options_new_kwarg_alone_does_not_warn(self):
        handle = self._make_handle()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            new_handle = handle.options(multiplex_ids={"model": "m1"})
        assert new_handle.handle_options.multiplex_ids == {"model": "m1"}
        deprecation_warnings = [
            x
            for x in w
            if "multiplexed_model_id" in str(x.message)
            and "deprecated" in str(x.message)
        ]
        assert deprecation_warnings == []


# =====================================================================
# 8. get_request_metadata tests
# =====================================================================
class TestGetRequestMetadata:
    def test_multiplex_ids_propagated_to_request_metadata(self):
        """get_request_metadata copies multiplex_ids into RequestMetadata;
        the derived `multiplexed_model_id` property reads from it."""
        import ray.serve.context as ctx
        from ray.serve._private.default_impl import get_request_metadata

        ctx._serve_request_context.set(_RequestContext())

        class FakeInitOptions:
            _source = DeploymentHandleSource.REPLICA
            _run_router_in_separate_loop = False

        class FakeHandleOptions:
            method_name = "__call__"
            multiplexed_model_id = ""
            multiplex_ids = {"model": "lora_v1", "session": "user_1"}
            stream = False
            _by_reference = True
            request_serialization = "cloudpickle"
            response_serialization = "cloudpickle"

        try:
            md = get_request_metadata(FakeInitOptions(), FakeHandleOptions())
            assert md.multiplex_ids == {"model": "lora_v1", "session": "user_1"}
            # multiplexed_model_id should be bridged from multiplex_ids["model"].
            assert md.multiplexed_model_id == "lora_v1"
        finally:
            ctx._serve_request_context.set(None)


# =====================================================================
# 9. _update_running_replicas integration test
# =====================================================================
@pytest.mark.asyncio
class TestUpdateRunningReplicasMultiplexDimensions:
    async def test_dimension_ids_propagate_through_update(self):
        """RunningReplicaInfo dimension_to_ids propagate to the router."""
        router = _make_router()
        deploy_id = DeploymentID(name="TEST_DEPLOYMENT")
        replica_id = ReplicaID(unique_id="r1", deployment_id=deploy_id)

        info = RunningReplicaInfo(
            replica_id=replica_id,
            node_id="n1",
            node_ip="1.2.3.4",
            availability_zone=None,
            actor_name="actor_r1",
            max_ongoing_requests=10,
            multiplex_dim_to_ids={
                "model": ["m1"],
                "session": ["s1", "s2"],
            },
        )
        router._update_running_replicas([info])

        # Per-dimension map should be populated.
        dim_map = router._multiplex_dim_id_to_replica_ids
        assert replica_id in dim_map["model"]["m1"]
        assert replica_id in dim_map["session"]["s1"]
        assert replica_id in dim_map["session"]["s2"]


# =====================================================================
# 10. Per-dimension LRU isolation
# =====================================================================
@pytest.mark.asyncio
class TestMultiDimensionLRUIsolation:
    """Per-dimension LRU caches must not interfere. The decorator stores one
    `_MultiplexWrapper` instance per `name` on the deployment instance, so
    eviction/reordering in one dimension must leave the other dimension
    untouched.
    """

    def setup_method(self, method):
        """Install a fake replica context so `_MultiplexWrapper.__init__`
        can read `replica_id` / `app_name` without a running Serve."""
        import ray.serve.context as ctx
        from ray.serve._private.common import DeploymentID, ReplicaID

        self._saved = ctx._INTERNAL_REPLICA_CONTEXT
        ctx._INTERNAL_REPLICA_CONTEXT = ctx.ReplicaContext(
            replica_id=ReplicaID(
                unique_id="r1", deployment_id=DeploymentID(name="TEST")
            ),
            servable_object=None,
            _deployment_config=None,
            rank=0,
            world_size=1,
        )

    def teardown_method(self, method):
        import ray.serve.context as ctx

        ctx._INTERNAL_REPLICA_CONTEXT = self._saved

    async def _make_wrapper(self, name: str, size: int):
        """Build a wrapper with its periodic controller-push disabled so
        unit tests don't try to reach a real controller."""
        from ray.serve.multiplex import _MultiplexWrapper

        async def _load(x):
            return x

        w = _MultiplexWrapper(_load, None, size_per_replica=size, name=name)
        await w.metrics_pusher.graceful_shutdown()
        return w

    async def test_each_wrapper_has_its_own_cache(self):
        """Loading in dim A must not populate or evict in dim B."""
        model_w = await self._make_wrapper("model", size=2)
        session_w = await self._make_wrapper("session", size=2)

        await model_w.load("m1")
        await model_w.load("m2")
        assert list(model_w.entries.keys()) == ["m1", "m2"]
        # Session cache stays empty — dims share no storage.
        assert list(session_w.entries.keys()) == []

        await session_w.load("s1")
        # Loading in session didn't touch model's entries.
        assert list(model_w.entries.keys()) == ["m1", "m2"]
        assert list(session_w.entries.keys()) == ["s1"]

    async def test_lru_order_isolated_across_dimensions(self):
        """A cache hit in one dim moves that id to MRU without reordering
        the other dim's cache."""
        model_w = await self._make_wrapper("model", size=2)
        session_w = await self._make_wrapper("session", size=2)

        # Prime both caches: model=[m1, m2], session=[s1, s2].
        await model_w.load("m1")
        await model_w.load("m2")
        await session_w.load("s1")
        await session_w.load("s2")
        assert list(model_w.entries.keys()) == ["m1", "m2"]
        assert list(session_w.entries.keys()) == ["s1", "s2"]

        # Touching model m1 (now LRU) reorders only the model cache.
        await model_w.load("m1")
        assert list(model_w.entries.keys()) == ["m2", "m1"]
        assert list(session_w.entries.keys()) == ["s1", "s2"]

    async def test_eviction_isolated_across_dimensions(self):
        """Overflowing one dim's size cap must not evict from another dim."""
        model_w = await self._make_wrapper("model", size=1)
        session_w = await self._make_wrapper("session", size=2)

        await session_w.load("s1")
        await session_w.load("s2")
        await model_w.load("m1")
        await model_w.load("m2")  # evicts m1 in the model dim only

        assert list(model_w.entries.keys()) == ["m2"]
        assert list(session_w.entries.keys()) == ["s1", "s2"]


# =====================================================================
# 11. Decorator dispatches to the right wrapper per dimension
# =====================================================================
@pytest.mark.asyncio
class TestDecoratorPerDimensionDispatch:
    """The `@serve.multiplexed(name=...)` decorator stores one
    `_MultiplexWrapper` per `name` in an attribute dict on the instance,
    and awaiting a decorated method dispatches to the wrapper for that
    method's declared dimension.
    """

    def setup_method(self, method):
        import ray.serve.context as ctx
        from ray.serve._private.common import DeploymentID, ReplicaID

        self._saved = ctx._INTERNAL_REPLICA_CONTEXT
        ctx._INTERNAL_REPLICA_CONTEXT = ctx.ReplicaContext(
            replica_id=ReplicaID(
                unique_id="r1", deployment_id=DeploymentID(name="TEST")
            ),
            servable_object=None,
            _deployment_config=None,
            rank=0,
            world_size=1,
        )

    def teardown_method(self, method):
        import ray.serve.context as ctx

        ctx._INTERNAL_REPLICA_CONTEXT = self._saved

    async def test_two_decorators_create_distinct_wrappers(self):
        from ray import serve

        class TwoDim:
            @serve.multiplexed(name="model", size_per_replica=3)
            async def load_model(self, mid):
                return ("model", mid)

            @serve.multiplexed(name="session", size_per_replica=3)
            async def load_session(self, sid):
                return ("session", sid)

        obj = TwoDim()
        # No wrappers are instantiated until the first `await` — matches
        # the lazy construction in the decorator.
        assert not hasattr(obj, "__serve_multiplex_wrapper")

        await obj.load_model("m1")
        await obj.load_session("s1")

        wrappers = getattr(obj, "__serve_multiplex_wrapper")
        assert set(wrappers.keys()) == {"model", "session"}
        # Distinct instances, distinct storage.
        assert wrappers["model"] is not wrappers["session"]
        assert list(wrappers["model"].entries.keys()) == ["m1"]
        assert list(wrappers["session"].entries.keys()) == ["s1"]

        # Dim A's load() does not move/evict in dim B.
        await obj.load_model("m2")
        assert list(wrappers["model"].entries.keys()) == ["m1", "m2"]
        assert list(wrappers["session"].entries.keys()) == ["s1"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
