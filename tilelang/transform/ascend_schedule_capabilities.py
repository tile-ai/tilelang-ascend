"""Host-only A3 scheduling contracts, independent of TileLang/TVM and solvers.

The budgets below describe the existing AscendC backend, not measured hardware
capacity. A2's existing compiler path is unaffected by this explicit A3 model.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass


_AUDITED_VECTOR_ROUTES = (
    ("global", "shared.ub", "MTE2"),
    ("shared.ub", "shared.ub", "V"),
    ("shared.ub", "global", "MTE3"),
)
_POOL_OWNERS = {
    "shared.ub": "V",
    "shared.l1": "C",
    "wmma.matrix_a": "C",
    "wmma.matrix_b": "C",
    "wmma.accumulator": "C",
}


class CapabilityError(ValueError):
    """An explicit scheduling request exceeds an audited capability."""


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CapabilityError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _pipe(pipe: str) -> str:
    if not isinstance(pipe, str):
        raise CapabilityError(f"pipe must be a string, got {pipe!r}")
    return pipe.upper().removeprefix("PIPE_")


@dataclass(frozen=True)
class Evidence:
    kind: str
    source: str
    claim: str


@dataclass(frozen=True)
class MemoryPool:
    scope: str
    owner: str
    budget_bytes: int
    allocation_alignment_bytes: int = 32


@dataclass(frozen=True)
class Route:
    owner: str
    src_scope: str
    dst_scope: str
    pipe: str


@dataclass(frozen=True)
class LocalEventDomain:
    owner: str
    src_pipe: str
    dst_pipe: str
    declared_ids: tuple[int, ...] = tuple(range(8))


@dataclass(frozen=True)
class PlatformCapabilities:
    platform: str
    backend: str
    pools: tuple[MemoryPool, ...]
    routes: tuple[Route, ...]
    local_event_domains: tuple[LocalEventDomain, ...]
    evidence: tuple[Evidence, ...]
    timing_model: None = None

    def pool(self, scope: str) -> MemoryPool:
        for pool in self.pools:
            if pool.scope == scope:
                return pool
        raise CapabilityError(f"Unknown local memory pool: {scope!r}")

    def validate(self) -> None:
        """Check the internal consistency of a capability description."""
        if self.platform != "A3" or self.backend != "ascendc":
            raise CapabilityError("This capability profile describes A3 AscendC only")
        if len({pool.scope for pool in self.pools}) != len(self.pools):
            raise CapabilityError("Memory pool scopes must be unique")
        if {pool.scope for pool in self.pools} != set(_POOL_OWNERS):
            raise CapabilityError("The A3 profile requires exactly the five known local memory pools")
        for pool in self.pools:
            if pool.owner != _POOL_OWNERS[pool.scope]:
                raise CapabilityError(f"Invalid owner {pool.owner!r} for memory pool {pool.scope}")
            alignment = _integer(pool.allocation_alignment_bytes, "allocation alignment", 1)
            budget = _integer(pool.budget_bytes, "pool budget", 1)
            if alignment & (alignment - 1) or budget % alignment:
                raise CapabilityError("Pool alignment must be a power of two dividing its budget")
        if len(set(self.routes)) != len(self.routes):
            raise CapabilityError("Copy routes must be unique")
        for route in self.routes:
            if route.owner != "V" or (route.src_scope, route.dst_scope, route.pipe) not in _AUDITED_VECTOR_ROUTES:
                raise CapabilityError(f"Unsupported Vector owner/pipe: {route!r}")
            for scope in (route.src_scope, route.dst_scope):
                if scope != "global" and self.pool(scope).owner != route.owner:
                    raise CapabilityError(f"Route crosses incompatible local owners: {route!r}")
        keys = [(d.owner, d.src_pipe, d.dst_pipe) for d in self.local_event_domains]
        if len(set(keys)) != len(keys):
            raise CapabilityError("Local event domains must be unique")
        for domain in self.local_event_domains:
            if domain.owner != "V" or domain.src_pipe == domain.dst_pipe or {domain.src_pipe, domain.dst_pipe} - {"MTE2", "V", "MTE3"}:
                raise CapabilityError(f"Unsupported local event domain: {domain!r}")
            if not domain.declared_ids or len(set(domain.declared_ids)) != len(domain.declared_ids):
                raise CapabilityError("Declared event IDs must be nonempty and unique")
            for event_id in domain.declared_ids:
                if _integer(event_id, "declared event ID") >= 8:
                    raise CapabilityError("A3 dav-2201 toolchain declares local event IDs 0..7")
        if self.timing_model is not None:
            raise CapabilityError("This capability profile does not provide a latency/II timing model")

    def validate_allocation(self, scope: str, size_bytes: int, offset_bytes: int = 0, versions: int = 1) -> int:
        """Return the physical byte count or reject an out-of-budget allocation.

        ``size_bytes`` must already include each version's layout padding and
        scratch footprint. This method only adds the pool's allocation rounding;
        it does not infer layouts, buffer lifetime reuse, or hardware capacity.
        """
        pool = self.pool(scope)
        size = _integer(size_bytes, "size_bytes", 1)
        offset = _integer(offset_bytes, "offset_bytes")
        count = _integer(versions, "versions", 1)
        alignment = pool.allocation_alignment_bytes
        if offset % alignment:
            raise CapabilityError(f"{scope} allocation offset must be {alignment}-byte aligned")
        physical_bytes = ((size + alignment - 1) // alignment) * alignment * count
        if offset + physical_bytes > pool.budget_bytes:
            raise CapabilityError(
                f"{scope} allocation needs [{offset}, {offset + physical_bytes}) bytes, exceeding compiler budget {pool.budget_bytes}"
            )
        return physical_bytes

    def validate_route(self, owner: str, src_scope: str, dst_scope: str, pipe: str) -> None:
        route = Route(owner, src_scope, dst_scope, _pipe(pipe))
        if route not in self.routes:
            raise CapabilityError(f"Route is unsupported by the audited A3 Vector model: {route!r}")

    def available_event_ids(self, owner: str, src_pipe: str, dst_pipe: str, reserved_ids: Iterable[int] = ()) -> tuple[int, ...]:
        """Subtract audited reservations in one core instance and directed domain.

        This operation does not infer event liveness or prove that unreported
        hardware/helper reservations are absent.
        """
        key = (owner, _pipe(src_pipe), _pipe(dst_pipe))
        domain = next(
            (d for d in self.local_event_domains if (d.owner, d.src_pipe, d.dst_pipe) == key),
            None,
        )
        if domain is None:
            raise CapabilityError(f"Unaudited local event domain: {key!r}")
        reserved = set()
        try:
            for event_id in reserved_ids:
                _integer(event_id, "reserved event ID")
                if event_id not in domain.declared_ids:
                    raise CapabilityError(f"Reserved event ID {event_id} is outside domain {key!r}")
                reserved.add(event_id)
        except TypeError as error:
            raise CapabilityError("reserved_ids must be an iterable of integer event IDs") from error
        return tuple(event_id for event_id in domain.declared_ids if event_id not in reserved)

    def validate_event_id(
        self,
        owner: str,
        src_pipe: str,
        dst_pipe: str,
        event_id: int,
        reserved_ids: Iterable[int] = (),
    ) -> None:
        _integer(event_id, "event ID")
        if event_id not in self.available_event_ids(owner, src_pipe, dst_pipe, reserved_ids):
            raise CapabilityError(f"Event ID {event_id} is unavailable or reserved in this domain")

    def validate_cross_core_event(self) -> None:
        raise CapabilityError("Cross-core flag banks and allocatable IDs are unaudited in this profile")

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "backend": self.backend,
            "validation_basis": "compiler contracts and SDK declarations; not hardware measurement",
            "hardware_validated": False,
            "pools": {pool.scope: asdict(pool) for pool in self.pools},
            "routes": [asdict(route) for route in self.routes],
            "local_event_domains": [asdict(domain) for domain in self.local_event_domains],
            "event_domain_identity": "core instance plus directed source/destination PIPE pair",
            "cross_core_events": {
                "supported": False,
                "allocatable_ids": None,
                "reason": "Four-bit message encoding does not prove independent or available banks",
            },
            "timing_model": self.timing_model,
            "evidence": [asdict(evidence) for evidence in self.evidence],
        }


_A3 = PlatformCapabilities(
    platform="A3",
    backend="ascendc",
    pools=(
        MemoryPool("shared.ub", "V", 196352),
        MemoryPool("shared.l1", "C", 524032),
        MemoryPool("wmma.matrix_a", "C", 65536),
        MemoryPool("wmma.matrix_b", "C", 65536),
        MemoryPool("wmma.accumulator", "C", 131072),
    ),
    routes=tuple(Route("V", *route) for route in _AUDITED_VECTOR_ROUTES),
    local_event_domains=tuple(
        LocalEventDomain("V", src, dst) for src in ("MTE2", "V", "MTE3") for dst in ("MTE2", "V", "MTE3") if src != dst
    ),
    evidence=(
        Evidence(
            "compiler_contract",
            "src/target/codegen_ascend.cc; src/transform/ascend_memory_planning.cc",
            "Existing A2/A3 AscendC pool budgets and 32-byte allocation alignment",
        ),
        Evidence(
            "compiler_contract",
            "src/transform/common/operation_config.h; src/transform/ascend_combinecv.cc",
            "Audited routes use explicit Vector ownership independently of PIPE names",
        ),
        Evidence(
            "sdk_declaration",
            "CANN 9.1.0 basic_api/impl/kernel_macros.h; kernel_tquesync_impl.h",
            "QUE_MAX_EVENT=8 for __NPU_ARCH__=2201; SetFlag/WaitFlag check local IDs <8",
        ),
        Evidence(
            "compiler_contract",
            "tilelang/jit/adapter/libgen.py",
            "AscendC compilation selects --npu-arch=dav-2201",
        ),
        Evidence(
            "sdk_encoding_only",
            "CANN 9.1.0 basic_api/impl/dav_c220/kernel_operator_sync_impl.h",
            "Cross-core messages mask flag IDs to four bits; allocation/bank safety is unproved",
        ),
    ),
)


def get_capabilities(platform: str = "A3") -> PlatformCapabilities:
    """Return the explicit A3 profile without altering any legacy target path."""
    if not isinstance(platform, str) or platform.upper() != "A3":
        raise CapabilityError(f"No audited capability profile for {platform!r}; expected A3")
    _A3.validate()
    return _A3
