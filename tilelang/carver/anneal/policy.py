from dataclasses import dataclass
import random
import math
from tilelang.utils.npu_arch import AscendArch, get_ascend_device_name
from .customtemplate import (
    PolyConstraints,
    FuncConstraints,
    PolyConstraintsSet,
    CustomTemplate,
    Config,
    get_byte_per_numel,
    get_all_factors,
)
import numpy as np
import bisect


def _get_unique_configs(configs):
    _configs = []
    seen = set()
    for config in configs:
        kwargs_tuple = tuple(config.kwargs)
        if kwargs_tuple not in seen:
            _configs.append(config)
            seen.add(kwargs_tuple)
    return _configs


def _get_aivector_cores(num_ai_cores=-1):
    if num_ai_cores < 0:
        default_arch = AscendArch(get_ascend_device_name())
        return default_arch.aivector_core_num
    return num_ai_cores


def _get_aicube_cores(num_ai_cores=-1):
    if num_ai_cores < 0:
        default_arch = AscendArch(get_ascend_device_name())
        return default_arch.aicube_core_num
    return num_ai_cores


def _get_real_idx(idx, use_idx, other_idx):
    use_idx_i = 0
    _idx = []
    for i in range(len(other_idx)):
        if other_idx[i] <= 0:
            _idx.append(idx[use_idx[use_idx_i]])
            use_idx_i += 1
        else:
            _idx.append(other_idx[i])
    return _idx


def _get_mem_caps(mem_caps, arch=None):
    if arch is None:
        default_arch = AscendArch(get_ascend_device_name())

    _mem_caps = []
    for cap in mem_caps:
        if cap in default_arch.mem_cap:
            _mem_caps.append(default_arch.mem_cap[cap])
        else:
            raise ValueError("Unsupported mem_caps!")

    return _mem_caps


@dataclass
class Annealparam:
    # default anneal param
    expand_neighbour: int = 2
    onestep_chance: float = 0.7  # possibility of 1 step neighbour
    topk: int = 10
    number: int = 100  # init length
    steps: int = 60
    temperature: float = 100
    alpha: float = 0.97  # temperature reduce rate
    custom_mem_multiply: float = 0.7


class AnnealCarver:
    def __init__(
        self,
        policy: PolyConstraintsSet,
        custom_kernel: CustomTemplate,
        mem_caps,
        annealparam: Annealparam,
    ):
        self.BLOCK_BASE = 16
        self.policy = policy
        self.mem_caps = mem_caps
        self.custom_kernel = custom_kernel
        self.annealparam = annealparam

    def check_valid(self, cfg):
        for constraints, cap in zip(
            self.custom_kernel.mem_caps, self.mem_caps, strict=True
        ):
            if (
                constraints.calc_results(cfg) * self.annealparam.custom_mem_multiply
                > cap
            ):
                return False
        return True

    def _calc_value_with_constraints(self, cfg):
        value = 0
        for _policy in self.policy:
            value += _policy.calc_results(cfg)
        return value

    def _calc_delta_with_constraints(self, change_x, new_cfg, old_cfg):
        # TODO: calc delta optimize
        pass

    def _get_neighbour(self, cfg, x, step_p):
        lo, hi = self.bound[x]
        new_cfg = cfg[:]
        if self.custom_kernel.only_factor[x] is True:
            pos = bisect.bisect_left(self.tile_factors[x], cfg[x]) + 1
            new_cfg[x] = self.tile_factors[x][max(lo, min(hi, pos + step_p)) - 1]
        else:
            new_cfg[x] = max(lo, min(hi, new_cfg[x] + step_p))
        return new_cfg

    def _neighbor(self, cfg, temperature):

        times = 15
        for _i in range(times):
            x = random.randrange(self.n)
            lo, hi = self.bound[x]
            tmp = cfg[x]
            new_cfg = cfg[:]

            if random.random() < self.annealparam.onestep_chance:
                step = max(1, hi // 16) if temperature > 0.5 else 1
                step_p = step if random.random() < 0.5 else -step
                new_cfg = self._get_neighbour(cfg, x, step_p)
            else:
                v = random.randint(lo, hi)
                new_cfg[x] = (
                    self.tile_factors[x][v - 1]
                    if self.custom_kernel.only_factor[x] is True
                    else v
                )

            if self.check_valid(new_cfg) is True:
                return x, new_cfg
            else:
                new_cfg[x] = tmp

        raise ValueError(f"Unable to find a neighbor within {times} attempts!")

    def _neighbor_configs(self, configs):
        _configs = []
        for config in configs:
            _configs.append(config)

            for x in range(self.n):
                for p in range(self.annealparam.expand_neighbour):
                    cfg = config.kwargs
                    step_p = p + 1
                    new_cfg = self._get_neighbour(cfg, x, step_p)
                    if self.check_valid(new_cfg) is True:
                        _configs.append(
                            Config(new_cfg, self._calc_value_with_constraints(new_cfg))
                        )

                    new_cfg = self._get_neighbour(cfg, x, -step_p)
                    if self.check_valid(new_cfg) is True:
                        _configs.append(
                            Config(new_cfg, self._calc_value_with_constraints(new_cfg))
                        )

        return _get_unique_configs(_configs)

    def _accept(self, new_value, old_value, temperature):  # default min

        improved = new_value < old_value

        if improved:
            return True

        delta = new_value - old_value
        p = math.exp(-delta / temperature)
        return random.random() < p

    def get_configs_from_anneal(self) -> list[Config]:
        topk = self.annealparam.topk
        number = self.annealparam.number
        steps = self.annealparam.steps
        temperature = self.annealparam.temperature
        alpha = self.annealparam.alpha

        configs: list[Config] = []
        self.n = len(self.custom_kernel.init_tile)
        self.bound = []
        self.tile_factors = []

        for x, y, z in zip(
            self.custom_kernel.init_low_tile,
            self.custom_kernel.init_tile,
            self.custom_kernel.only_factor,
            strict=True,
        ):
            if z is True:
                if y & 15 != 0:
                    raise ValueError("16 must be a factor of the tile!")
                lo = max(1, x >> 4)  # // BLOCK_BASE
                factors = get_all_factors(y >> 4)
                while factors[0] < lo:
                    factors.pop(0)
                self.bound.append([1, len(factors)])
                self.tile_factors.append(factors)
            else:
                lo = max(1, x >> 4)
                hi = max(1, y >> 4)
                self.bound.append([lo, hi])
                self.tile_factors.append([1])

        number_i = 0
        for _i in range(number * 4):
            cfg = []

            for j in range(self.n):
                v = random.randint(self.bound[j][0], self.bound[j][1])
                cfg.append(
                    self.tile_factors[j][v - 1]
                    if self.custom_kernel.only_factor[j] is True
                    else v
                )

            if self.check_valid(cfg):
                configs.append(Config(cfg, self._calc_value_with_constraints(cfg)))
                number_i += 1
                if number_i == number:
                    break

        self.m = len(configs)
        for _ in range(steps):
            for i in range(self.m):
                cfg = configs[i].kwargs
                old_value = configs[i].value
                change_x, new_cfg = self._neighbor(cfg, temperature)
                new_value = self._calc_value_with_constraints(new_cfg)
                accept = self._accept(new_value, old_value, temperature)
                if accept is True:
                    configs[i] = Config(new_cfg, new_value)
            temperature *= alpha

        configs.sort(key=lambda c: c.value)
        configs = _get_unique_configs(configs)
        configs = self._neighbor_configs(configs)
        configs.sort(key=lambda c: c.value)

        result_configs = configs[:topk]
        for cfg in result_configs:
            for i in range(self.n):
                cfg.kwargs[i] = min(
                    self.custom_kernel.init_tile[i], cfg.kwargs[i] * self.BLOCK_BASE
                )

        return result_configs


def _compute_memory_traffic_matmul(idx, nbytes):
    MIN_BYTES = 32
    traffic = 0
    traffic += max(MIN_BYTES, idx[0] * idx[1] * nbytes[0])
    traffic += max(MIN_BYTES, idx[0] * idx[2] * nbytes[1])
    traffic += max(MIN_BYTES, idx[1] * idx[2] * nbytes[2])
    return traffic


def _compute_memory_traffic_vector(idx, nbytes):
    MIN_BYTES = 32
    traffic = 0
    traffic += max(MIN_BYTES, idx[0] * idx[1] * nbytes[0])
    return traffic


def _compute_num_waves_matmul(idx, init_idx, num_ai_cores):
    w = [(y + x - 1) // x for x, y in zip(idx, init_idx, strict=True)]
    num_grids = w[0] * w[2] * w[1] * w[2]
    num_waves = (num_grids + num_ai_cores - 1) // num_ai_cores
    return num_waves


def _compute_num_waves_vector(idx, init_idx, num_ai_cores):
    w = [(y + x - 1) // x for x, y in zip(idx, init_idx, strict=True)]
    num_grids = np.prod(w)
    num_waves = (num_grids + num_ai_cores - 1) // num_ai_cores
    return num_waves


def _compute_carver_policy_matmul(idx, nbytes, init_idx, num_ai_cores=-1):
    return _compute_memory_traffic_matmul(idx, nbytes) * _compute_num_waves_matmul(
        idx, init_idx, _get_aicube_cores(num_ai_cores)
    )


def _compute_carver_policy_vector(idx, nbytes, init_idx, num_ai_cores=-1):
    return _compute_memory_traffic_vector(idx, nbytes) * _compute_num_waves_vector(
        idx, init_idx, _get_aivector_cores(num_ai_cores)
    )


def _compute_carver_policy_matmul_custom_idx(
    idx, use_idx, other_idx, nbytes, init_idx, num_ai_cores=-1
):
    _idx = _get_real_idx(idx, use_idx, other_idx)
    return _compute_memory_traffic_matmul(_idx, nbytes) * _compute_num_waves_matmul(
        _idx, init_idx, _get_aicube_cores(num_ai_cores)
    )


def _compute_carver_policy_vector_custom_idx(
    idx, use_idx, other_idx, nbytes, init_idx, num_ai_cores=-1
):
    _idx = _get_real_idx(idx, use_idx, other_idx)
    return _compute_memory_traffic_vector(_idx, nbytes) * _compute_num_waves_vector(
        _idx, init_idx, _get_aivector_cores(num_ai_cores)
    )


class AnnealTemplate:
    def __init__(
        self,
        shape,
        init_low_tile=None,
        init_tile=None,
        only_factor=None,
        annealparam=None,
        use_template="Elementwise",
        dtype="float16",
        accum_dtype="float32",
        custom_kernel=None,
        policy=None,
        mem_caps=None,
        arch=None,
    ):
        BLOCK_BASE = 16
        template_map = ["Elementwise", "Matmul", "FlashAttention", "Custom"]
        if use_template not in template_map:
            raise ValueError("use template do not support!")

        if annealparam is None:
            annealparam = Annealparam()

        if arch is None:
            self.arch = AscendArch(get_ascend_device_name())

        self.shape = shape
        self.n = len(shape)
        self.annealparam = annealparam
        self.use_template = use_template
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.custom_kernel = custom_kernel
        self.policy = policy

        if mem_caps is None:
            if self.use_template == "Custom":
                raise ValueError("custom mem_caps is none!")
            elif self.use_template == "Elementwise":
                self.mem_caps = ["UB"]
            elif self.use_template == "Matmul":
                self.mem_caps = ["L1"]
            elif self.use_template == "FlashAttention":
                self.mem_caps = ["L1", "L1", "UB"]
        else:
            self.mem_caps = mem_caps

        if init_low_tile is None:
            self.init_low_tile = [BLOCK_BASE] * self.n
            if self.use_template == "FlashAttention":
                self.init_low_tile[2] = shape[2]
        else:
            self.init_low_tile = init_low_tile

        if init_tile is None:
            self.init_tile = shape[:]
            if self.use_template == "FlashAttention":
                self.init_tile[1] = min(shape[0], 256)  # max tile
                self.init_tile[1] = min(shape[1], 256)
        else:
            self.init_tile = init_tile

        if only_factor is None:
            self.only_factor = [True] * self.n
            if self.use_template == "FlashAttention":
                self.only_factor[0] = False
        else:
            self.only_factor = only_factor

    def custom_template(self):
        annel_carver = AnnealCarver(
            self.policy,
            self.custom_kernel,
            _get_mem_caps(self.mem_caps),
            self.annealparam,
        )

        return annel_carver.get_configs_from_anneal()

    def matmul_template(self):
        numel_dtype = get_byte_per_numel(self.dtype)
        numel_accum_dtype = get_byte_per_numel(self.accum_dtype)

        arch = self.arch

        custom_kernel = CustomTemplate(
            init_low_tile=self.init_low_tile,
            init_tile=self.init_tile,
            only_factor=self.only_factor,
            mem_caps=[
                PolyConstraintsSet(
                    [
                        PolyConstraints(
                            polynomial=[1, 0, 1], other_value=[numel_dtype]
                        ),
                        PolyConstraints(
                            polynomial=[0, 1, 1], other_value=[numel_dtype]
                        ),
                    ],
                    [],
                    [],
                    [],
                ),
            ],  # l1_cap
        )

        policy = [
            PolyConstraintsSet(
                [],
                [],
                [
                    FuncConstraints(
                        [
                            1,
                            2,
                            3,
                        ],
                        [],
                        _compute_carver_policy_matmul,
                        nbytes=[numel_dtype, numel_dtype, numel_accum_dtype],
                        init_idx=self.shape,
                        num_ai_cores=arch.aicube_core_num,
                    ),
                ],
                [],
            ),  # matmul 1
        ]

        annel_carver = AnnealCarver(
            policy, custom_kernel, _get_mem_caps(self.mem_caps), self.annealparam
        )

        return annel_carver.get_configs_from_anneal()

    def elementwise_template(self):
        numel_dtype = get_byte_per_numel(self.dtype)
        numel_accum_dtype = get_byte_per_numel(self.accum_dtype)

        arch = self.arch

        custom_kernel = CustomTemplate(
            init_low_tile=self.init_low_tile,
            init_tile=self.init_tile,
            only_factor=self.only_factor,
            mem_caps=[
                PolyConstraintsSet(
                    [
                        PolyConstraints(
                            polynomial=[1] * self.n, other_value=[2, numel_accum_dtype]
                        ),
                    ],
                    [],
                    [],
                    [],
                ),
            ],  # ub_cap
        )

        policy = [
            PolyConstraintsSet(
                [],
                [],
                [
                    FuncConstraints(
                        [x + 1 for x in range(self.n)],
                        [],
                        _compute_carver_policy_vector,
                        nbytes=[numel_dtype] * self.n,
                        init_idx=self.shape,
                        num_ai_cores=arch.aivector_core_num,
                    ),
                ],
                [],
            ),  # matmul 1
        ]
        annel_carver = AnnealCarver(
            policy, custom_kernel, _get_mem_caps(self.mem_caps), self.annealparam
        )

        return annel_carver.get_configs_from_anneal()

    def flashattention_template(self):
        numel_dtype = get_byte_per_numel(self.dtype)
        numel_accum_dtype = get_byte_per_numel(self.accum_dtype)

        arch = self.arch

        custom_kernel = CustomTemplate(
            init_low_tile=self.init_low_tile,
            init_tile=self.init_tile,
            only_factor=self.only_factor,
            mem_caps=[
                PolyConstraintsSet(
                    [
                        PolyConstraints(
                            polynomial=[1, 0, 1], other_value=[numel_dtype]
                        ),
                        PolyConstraints(
                            polynomial=[0, 1, 1], other_value=[numel_dtype]
                        ),
                    ],
                    [],
                    [],
                    [],
                ),  # l1_cap1
                PolyConstraintsSet(
                    [
                        PolyConstraints(
                            polynomial=[1, 1, 0], other_value=[numel_accum_dtype]
                        ),
                        PolyConstraints(
                            polynomial=[0, 1, 1], other_value=[numel_dtype]
                        ),
                    ],
                    [],
                    [],
                    [],
                ),  # l1_cap2
                PolyConstraintsSet(
                    [
                        PolyConstraints(
                            polynomial=[1, 0, 1],
                            other_value=[numel_dtype + 2 * numel_accum_dtype],
                        ),
                        PolyConstraints(
                            polynomial=[1, 1, 0],
                            other_value=[numel_dtype + numel_accum_dtype],
                        ),
                    ],
                    [
                        PolyConstraints(polynomial=[0, 0, 0], other_value=[0.5]),
                    ],
                    [],
                    [],
                ),  # ub_cap
            ],
        )

        policy = [
            PolyConstraintsSet(
                [],
                [],
                [
                    FuncConstraints(
                        [
                            1,
                            2,
                            3,
                        ],
                        [],
                        _compute_carver_policy_matmul,
                        nbytes=[numel_dtype, numel_dtype, numel_accum_dtype],
                        init_idx=self.shape,
                        num_ai_cores=arch.aicube_core_num,
                    ),
                    FuncConstraints(
                        [
                            1,
                            3,
                            2,
                        ],
                        [],
                        _compute_carver_policy_matmul,
                        nbytes=[numel_dtype, numel_dtype, numel_accum_dtype],
                        init_idx=[self.shape[0], self.shape[2], self.shape[1]],
                        num_ai_cores=arch.aicube_core_num,
                    ),
                    FuncConstraints(
                        [
                            1,
                            3,
                        ],
                        [512],
                        _compute_carver_policy_vector,
                        nbytes=[numel_accum_dtype, numel_accum_dtype],
                        init_idx=[self.shape[0], self.shape[2]],
                        num_ai_cores=arch.aivector_core_num,
                    ),
                ],
                [],
            ),
        ]

        annel_carver = AnnealCarver(
            policy, custom_kernel, _get_mem_caps(self.mem_caps), self.annealparam
        )

        return annel_carver.get_configs_from_anneal()

    def get_configs(self):
        if self.use_template == "Custom":
            return self.custom_template()
        elif self.use_template == "Elementwise":
            return self.elementwise_template()
        elif self.use_template == "Matmul":
            return self.matmul_template()
        elif self.use_template == "FlashAttention":
            return self.flashattention_template()
