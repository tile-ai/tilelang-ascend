# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import torch
import functools
from dataclasses import dataclass
from typing import (
    Dict,
    List,
)

from .utils import (
    get_byte_per_numel,
    next_power_of_2,
    num_vector_core,
    ub_size_in_kbytes,
    rf_size_in_kbytes,
)


class Config:
    def __init__(self, cfg):
        self.kwargs = cfg


@dataclass
class AxisInfo:
    name: str
    index: int
    length: int

    prefix: str = ""
    split_name: str = ""
    tiling_name: str = ""
    is_split_axis: bool = False
    is_tiling_axis: bool = False

    @property
    def is_reduction(self):
        return self.prefix == "r"


class KernelMeta:
    def __init__(
        self,
        axis_sizes: Dict[str, int],
        split_params: Dict[str, str],
        tiling_params: Dict[str, str],
        low_dims: List[str],
        dtype: torch.dtype,
        persistent_reduction: bool,
        dual_reduction: bool,
        num_buffers: int,
        is_simt_mode: bool,
    ):
        """
        :param split_params: a dict of axis name: argument name, the argument is an adjustable parameter in a split axis, such as 'XBLOCK'.
            The axis name must be in key's axis names. Do not add prefix 'r' before the axis name.
            This param can be empty. Note that the auto tiling feature will be disabled when the split_params and tiling_params are both empty.
            The split axis can usually be identified according to `tl.program_id()` expression.
        :type split_params: Dict[str, str]
        :param tiling_params: a dict of axis name: argument name, the argument is an adjustable parameter in a tiling axis, such as 'XBLOCK_SUB'.
            The axis name must be in key's axis names. Do not add prefix 'r' before the axis name.
            This param can be empty. Note that the auto tiling feature will be disabled when the split_params and tiling_params are both empty.
            The tiling axis can usually be identified according to `tl.arange()` expression.
        :type tiling_params: Dict[str, str]
        :param low_dims: a list of axis name in which the corresponding axis is low dim axis.
            The axis name must be in key's axis names. Do not add prefix 'r' before the axis name.
        :type low_dims: List[str]
        :param dual_reduction: performing reduction on more than one axis.
        :param persistent_reduction: there is no splitting in reduction axis.
        """
        self._validate_axis(axis_sizes, split_params, tiling_params, low_dims)

        axis_dict = {}
        for index, (name, length) in enumerate(axis_sizes.items()):
            prefix = ""
            if name.startswith("r"):
                prefix = "r"

            is_split_axis = name in split_params
            is_tiling_axis = name in tiling_params
            split_name = split_params.get(name, "")
            tiling_name = tiling_params.get(name, "")

            axis_dict[name] = AxisInfo(
                name=name,
                index=index,
                length=length,
                prefix=prefix,
                split_name=split_name,
                tiling_name=tiling_name,
                is_split_axis=is_split_axis,
                is_tiling_axis=is_tiling_axis,
            )

        self.axis_info = list(axis_dict.values())
        self.split_axis = [x for x in axis_dict.values() if x.is_split_axis]
        self.tiling_axis = [x for x in axis_dict.values() if x.is_tiling_axis]
        self.low_dims_axis = [x for x in axis_dict.values() if x.name in low_dims]
        self.dtype = dtype
        self.persistent_reduction = persistent_reduction
        self.dual_reduction = dual_reduction
        self.num_buffers = num_buffers
        self.is_simt_mode = is_simt_mode

    @classmethod
    def _validate_axis(
        cls,
        axis_sizes: Dict[str, int],
        split_params: Dict[str, str],
        tiling_params: Dict[str, str],
        low_dims: List[str],
    ) -> None:
        for axis_name in axis_sizes.keys():
            if axis_name.startswith("r") and len(axis_name) == 1:
                raise ValueError("The name of a reduction axis is empty!")

        def check_keys(params: List[str], context="parameter"):
            for k in params:
                if k not in axis_sizes and ("r" + k) not in axis_sizes:
                    raise KeyError(
                        f"{context} '{k}' not found in known axes: {axis_sizes.keys()}"
                    )

        check_keys(split_params.keys(), "split axis")
        check_keys(tiling_params.keys(), "tiling axis")
        check_keys(low_dims, "low dim axis")


@dataclass
class BlockInfo:
    block_name: str  # e.g., XBLOCK
    sub_block_name: str  # e.g., XBLOCK_SUB
    block_size: int
    sub_block_size: int


"""
Generate possible candidate tiling configs for benchmarking
"""


class TileGenerator:
    num_warps = 1
    num_stages = 1

    def __init__(self, kernel_meta: KernelMeta):
        self.kernel_meta = kernel_meta
        self.persistent_reduction = self.kernel_meta.persistent_reduction
        self.dual_reduction = self.kernel_meta.dual_reduction

        self.blocks = self.init_blocks_info(kernel_meta)
        self.numels = [axis.length for axis in kernel_meta.axis_info]
        self.candidate_blocks = []
        self.configs = []
        self.dtype_bytes = get_byte_per_numel(kernel_meta.dtype)

        self.num_buffers = (
            3 if kernel_meta.num_buffers == 0 else min(kernel_meta.num_buffers, 3)
        )  # Triple buffer limit
        self.is_simt_mode = kernel_meta.is_simt_mode
        local_mem_size = rf_size_in_kbytes if self.is_simt_mode else ub_size_in_kbytes
        self.max_numel_threshold = (
            local_mem_size * 1024 // self.dtype_bytes // self.num_buffers
        )
        self.max_total_numel = (
            functools.reduce(lambda x, y: x * y, [x.block_size for x in self.blocks])
            if self.blocks
            else 1
        )
        self.tiny_kernel = (
            self.max_total_numel < 128 * 1024
        )  # Threshold of tiny kernel, block shape product  < 128 * 1024
        self.stop_numel = (
            min(1024 // self.dtype_bytes, self.max_total_numel // (num_vector_core * 2))
            if self.tiny_kernel
            else 1024 // self.dtype_bytes
        )
        self.max_programs_num = 65535

    @classmethod
    def init_blocks_info(cls, kernel_meta: KernelMeta) -> List[BlockInfo]:
        blocks = []
        for axis in kernel_meta.axis_info:
            block_name = axis.split_name
            sub_block_name = axis.tiling_name
            block_size = axis.length
            sub_block_size = block_size
            blocks.append(
                BlockInfo(block_name, sub_block_name, block_size, sub_block_size)
            )

        return blocks

    @classmethod
    def get_key_from_dict(cls, kwargs: Dict[str, int]):
        return tuple(sorted(kwargs.items()))

    def calcu_last_split_blocks(self, axis_idx):
        splits = 1
        for x in self.kernel_meta.split_axis:
            if x.index != axis_idx:
                splits = splits * (
                    (self.numels[x.index] + self.blocks[x.index].block_size - 1)
                    // self.blocks[x.index].block_size
                )
            else:
                break

        last_splits = num_vector_core // splits
        last_splits = max(1, last_splits)
        last_blocks = (self.numels[axis_idx] + last_splits - 1) // last_splits
        return last_blocks

    def aligned_numel(self, numel, align_bytes=32):
        if self.is_simt_mode:
            return next_power_of_2(numel)

        align_numel = align_bytes // self.dtype_bytes
        if numel <= align_numel:
            return numel
        return ((numel + align_numel - 1) // align_numel) * align_numel

    def calculate_tile_numel(self):
        tile_numel = 1
        for axis in self.kernel_meta.axis_info:
            if axis.is_tiling_axis:
                tile_numel *= self.blocks[axis.index].sub_block_size
            else:
                tile_numel *= self.blocks[axis.index].block_size

        return tile_numel

    def fill_config(self, cfg, candi_block):
        for axis in self.kernel_meta.axis_info:
            if not (axis.is_split_axis or axis.is_tiling_axis):
                continue
            block_info = self.blocks[axis.index]
            if axis.is_split_axis:
                curr_numel = candi_block[axis.index]
                if not axis.is_tiling_axis:
                    curr_numel = self.aligned_numel(curr_numel)
                cfg[block_info.block_name] = curr_numel
            if axis.is_tiling_axis:
                tiling_numel = self.aligned_numel(block_info.sub_block_size)
                cfg[block_info.sub_block_name] = min(
                    tiling_numel, candi_block[axis.index]
                )

    def find_config(self, cfg):
        return any(config_var.kwargs == cfg for config_var in self.configs)

    def add_to_configs(self, candi_block):
        newcfg = {}
        self.fill_config(newcfg, candi_block)
        tile_numel = self.calculate_tile_numel()
        stop_numel_threshold = (
            0 if len(self.configs) < 10 or self.tiny_kernel else self.stop_numel + 128
        )  # Stop heuristic, no threshold when less configs and +128 for margin configs
        if (
            tile_numel <= self.max_numel_threshold
            and tile_numel >= stop_numel_threshold
            and not self.find_config(newcfg)
        ):
            self.configs.append(Config(newcfg))
            return True
        return False

    def desecnd_all_low_dims_with_all_blocks(self):
        restore_sub_blocks = {}
        for axis in self.kernel_meta.low_dims_axis:
            restore_sub_blocks[axis.index] = self.blocks[axis.index].sub_block_size
        self.descend_all_low_dims()
        for axis in self.kernel_meta.low_dims_axis:
            self.blocks[axis.index].sub_block_size = restore_sub_blocks[axis.index]

    def descend_one_axis(self, axis_idx: int, is_split=False):
        def calc_total_programs():
            grids = []
            for axis in self.kernel_meta.split_axis:
                numel = self.numels[axis.index]
                block_size = self.blocks[axis.index].block_size
                programs = (numel + block_size - 1) // block_size
                grids.append(programs)

            total_programs = functools.reduce(lambda x, y: x * y, grids) if grids else 1
            return total_programs

        reached_stop_numel = False
        slow_decend_split = False
        num_vector_core_tile = num_vector_core
        max_programs_num = (
            num_vector_core_tile
            if self.kernel_meta.tiling_axis
            else self.max_programs_num
        )
        if not is_split and len(self.candidate_blocks) == 0:
            self.candidate_blocks.append(tuple([x.block_size for x in self.blocks]))

        axis = self.kernel_meta.axis_info[axis_idx]
        while True:
            for candi_block in self.candidate_blocks:
                if self.add_to_configs(candi_block):
                    self.desecnd_all_low_dims_with_all_blocks()

            # tile numel reached threshold
            tile_numel = self.calculate_tile_numel()
            if tile_numel <= self.stop_numel:
                if self.add_to_configs([x.block_size for x in self.blocks]):
                    self.desecnd_all_low_dims_with_all_blocks()
                reached_stop_numel = True
                break

            numel = (
                self.blocks[axis_idx].block_size
                if is_split
                else self.blocks[axis_idx].sub_block_size
            )
            if numel == 1:
                if self.add_to_configs([x.block_size for x in self.blocks]):
                    self.desecnd_all_low_dims_with_all_blocks()
                break

            if is_split:
                if self.persistent_reduction and axis.is_reduction:
                    reached_stop_numel = True
                    break
                total_programs = calc_total_programs()
                if total_programs > num_vector_core_tile:
                    if len(self.configs) == 0:
                        num_vector_core_tile = max_programs_num
                        slow_decend_split = total_programs > num_vector_core_tile // 2
                    if total_programs > num_vector_core_tile:
                        last_blocks = self.calcu_last_split_blocks(axis_idx)
                        if last_blocks != self.blocks[axis_idx].block_size:
                            self.blocks[axis_idx].block_size = last_blocks
                            self.candidate_blocks.append(
                                tuple([x.block_size for x in self.blocks])
                            )
                        break

                program_threshold = (
                    num_vector_core // 8 if self.tiny_kernel else num_vector_core // 2
                )
                if total_programs > program_threshold or self.dual_reduction:
                    if len(self.candidate_blocks) > 2:
                        self.candidate_blocks.pop(0)
                    self.candidate_blocks.append(
                        tuple([x.block_size for x in self.blocks])
                    )
                    if self.tiny_kernel:
                        self.add_to_configs(
                            list(tuple([x.block_size for x in self.blocks]))
                        )
                    slow_decend_split = total_programs > num_vector_core_tile // 2

                if not slow_decend_split:
                    self.blocks[axis_idx].block_size = numel // 2
                else:
                    step = numel // 4 if numel // 4 > 1 else 1
                    self.blocks[axis_idx].block_size = numel - step
                self.blocks[axis_idx].sub_block_size = self.blocks[axis_idx].block_size
                total_programs = calc_total_programs()
                if self.blocks[axis_idx].block_size == 1 and (
                    total_programs > program_threshold or self.dual_reduction
                ):
                    self.candidate_blocks.append(
                        tuple([x.block_size for x in self.blocks])
                    )
            else:
                if numel >= 32:
                    self.blocks[axis_idx].sub_block_size = next_power_of_2(numel // 2)
                else:
                    self.blocks[axis_idx].sub_block_size = numel - 1
        return reached_stop_numel

    def descend_all_low_dims(self):
        low_dim_numels = [
            self.blocks[x.index].sub_block_size for x in self.kernel_meta.low_dims_axis
        ]
        if not low_dim_numels:
            return False

        def descend_all_axis(min_numel):

            for axis in self.kernel_meta.low_dims_axis:
                if axis.is_reduction and self.persistent_reduction:
                    continue

                numel = self.blocks[axis.index].sub_block_size
                if numel == 1:
                    continue
                if min_numel > 1 and abs(numel - min_numel) / min_numel < 0.2:
                    continue
                if numel >= 128:
                    self.blocks[axis.index].sub_block_size = next_power_of_2(numel // 2)
                else:
                    numel = self.blocks[axis.index].sub_block_size
                    numel = numel // 2
                    self.blocks[axis.index].sub_block_size = min(
                        self.aligned_numel(numel), next_power_of_2(numel)
                    )

        if len(self.candidate_blocks) == 0:
            # means there is no split axis and tiling_not_low_dim axis
            # so we need to init the candidates_blk_sizes
            self.candidate_blocks.append(tuple([x.block_size for x in self.blocks]))

        count = 0
        tile_numel = self.calculate_tile_numel()
        while tile_numel > self.stop_numel and count < 100:
            count += 1
            tile_numel = self.calculate_tile_numel()
            for candi_block in self.candidate_blocks:
                self.add_to_configs(candi_block)
            min_numel = min(low_dim_numels)
            descend_all_axis(min_numel)
            new_tile_numel = self.calculate_tile_numel()
            if tile_numel == new_tile_numel:
                descend_all_axis(0)

        return tile_numel < self.stop_numel

    def descend_split_tiling(self):

        tiling_not_low_dims = [
            x
            for x in self.kernel_meta.tiling_axis
            if x not in self.kernel_meta.low_dims_axis
        ]

        def descend_split_axis():
            for axis in self.kernel_meta.split_axis:
                if self.descend_one_axis(axis.index, is_split=True):
                    return True

            return self.calculate_tile_numel() <= self.stop_numel

        def descend_tiling_not_low_dims():
            for axis in tiling_not_low_dims:
                if axis.is_reduction and self.persistent_reduction:
                    continue

                if self.descend_one_axis(axis.index):
                    return True
            return self.calculate_tile_numel() <= self.stop_numel

        while True:
            # descend split axis
            if descend_split_axis():
                break
            if len(self.candidate_blocks) > 0:
                candi_block = self.candidate_blocks[0]
                for i, blk_size in enumerate(candi_block):
                    self.blocks[i].sub_block_size = blk_size
            # descend tiling but not low dims
            if descend_tiling_not_low_dims():
                break
            # descend low dims, need to descend all axis at the same time
            self.descend_all_low_dims()
            break
