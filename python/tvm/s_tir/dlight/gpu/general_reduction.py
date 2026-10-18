# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

import tvm_ffi

from tvm import s_tir
from tvm_tirx import sym, tirx
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.target import Target

from ..analysis import SBlockInfo, get_root_block, normalize_function
from ..base import get_extent, try_inline_contiguous_spatial
from .base import GPUScheduleRule

_SCOPE_GLOBAL = MemoryScope("global")

# Reductions with at most this many elements are folded serially inside one thread
# instead of being split across `len_tx` threads. A cross-thread reduction over
# a small extent leaves most threads idle while still paying for a block per
# output, shared memory, and warp shuffles.
#
# The crossover was measured on an RTX A6000(see #20401):
# at extent 289, the serial schedule was 32.90 us vs 403.01 us for the
# cross-thread schedule. 256 is `len_tx`, the point where one thread per
# reduction element stops being applicable; the measurement confirms it is
# still on the serial side of the crossover.
_SMALL_REDUCTION_MAX_EXTENT = 256


def _static_extent(block_info: SBlockInfo, kind: str) -> int | None:
    """Product of the extents of the iterators of `kind` ("S" or "R"), or None if dynamic."""
    extent = 1
    for iter_info in block_info.iters:
        if iter_info.kind != kind:
            continue
        if not isinstance(iter_info.dom, int):
            return None
        extent *= iter_info.dom
    return extent


class GeneralReduction(GPUScheduleRule):
    """General Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None

        if target.kind.name == "cuda":
            len_tx = 256
            unroll_depth = 256
        elif target.kind.name == "opencl":
            len_tx = 256
            unroll_depth = 64
        else:
            len_tx = 64
            unroll_depth = 64

        sch = s_tir.Schedule(func)
        block_infos = normalize_function(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        dom_kind = block_infos[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        # Align the number of block iters of the last block.
        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter < len(dom_kind):
            # If the last block is a scalar value, there is nothing left to
            # tile/parallelise, and  `iters` is an empty tuple.
            # Add a unit thread loop so the final write happens inside a valid
            # GPU thread environment.
            if num_last_block_iter == 0:
                # Allocation planning can move a reduction tensor inside its
                # producer kernel when all surrounding loops are trivial. Give
                # tensors private to that kernel an explicit local scope, while
                # preserving global scope for tensors accessed by another block.
                blocks = [sch.get(info.block_rv) for info in block_infos]
                alloc_tensors = list(sch.get(get_root_block(sch)).alloc_tensors)
                analyzer = sym.Analyzer()
                for block_index, (info, block) in enumerate(zip(block_infos[:-1], blocks[:-1])):
                    loops = sch.get_loops(info.block_rv)
                    if not all(analyzer.can_prove_equal(sch.get(loop).extent, 1) for loop in loops):
                        continue

                    other_block_tensors = [
                        region.source
                        for other_index, other_block in enumerate(blocks)
                        if other_index != block_index
                        for region in (*other_block.reads, *other_block.writes)
                    ]
                    for tensor_index, write in enumerate(block.writes):
                        tensor = write.source
                        is_allocated = any(tensor.same_as(other) for other in alloc_tensors)
                        is_cross_block = any(tensor.same_as(other) for other in other_block_tensors)
                        if tensor.scope() == _SCOPE_GLOBAL and is_allocated and not is_cross_block:
                            sch.set_scope(block_infos[block_index].block_rv, tensor_index, "local")

                # Put every block (both the running reductions and the final
                # scalar write) inside a trivial GPU thread. The very first block
                # gets a `blockIdx.x` wrapper so that kernels still have a unique
                # block scope.
                for i, info in enumerate(block_infos):
                    loop_rv = sch.add_unit_loop(info.block_rv)
                    if i == 0:
                        sch.bind(loop_rv, "blockIdx.x")
                    else:
                        sch.bind(loop_rv, "threadIdx.x")

                return sch

            def f_layout_mapping(*iters):
                analyzer = sym.Analyzer()
                # Try to match the iters of last block to the iters of the first block.
                # For matched positions, use the iter from the input `iters`.
                # For unmatched positions, use a new iter which is constant 0.
                num_matched = 0
                target_layout_iters = []
                for block_iter in block_infos[0].iters:
                    if num_matched < len(iters) and analyzer.can_prove_equal(
                        block_iter.dom, block_infos[-1].iters[num_matched].dom
                    ):
                        target_layout_iters.append(iters[num_matched])
                        num_matched += 1
                    else:
                        target_layout_iters.append(tirx.const(0, iters[0].ty))

                # If all the iters of the last block can match, return the new layout.
                if num_matched == len(iters):
                    return target_layout_iters
                # Otherwise, fallback to appending zeros in the beginning.
                return [tirx.const(0, iters[0].ty)] * (len(dom_kind) - num_last_block_iter) + list(
                    iters
                )

            index_map = tirx.IndexMap.from_func(f_layout_mapping, ndim=num_last_block_iter)
            sch.transform_block_layout(block_infos[-1].block_rv, index_map)

        try:
            # TODO: fix num_leading_s = 0 case
            assert num_trailing_r > 0
            for block in block_infos[1:-1]:
                assert block.dom_kind() == dom_kind
            assert block_infos[-1].is_injective()
            assert len(block_infos[-1].dom_kind()) <= len(dom_kind)
        except AssertionError:
            return None

        if "R" not in block_infos[-1].dom_kind():
            # The final block is a spatial block.
            # It is possible that the loop order of the last block is not the same as
            # previous blocks.
            # Thus we reorder spatial loops to align with reduction loops for followup schedule.
            # We first collect all the tensors written by reduction blocks,
            # then in the final block, any index of those tensors are spatial.
            reduced_tensors = []
            for block_info in block_infos[:-1]:
                for tensor_write in sch.get(block_info.block_rv).writes:
                    reduced_tensors.append(tensor_write.source)

            spatial_block = sch.get(block_infos[-1].block_rv)
            spatial_loops = set()
            block_var_to_loop_var = {}
            loops = sch.get_loops(block_infos[-1].block_rv)
            for block_iter, loop_rv in zip(spatial_block.iter_vars, loops):
                block_var_to_loop_var[block_iter.var] = sch.get(loop_rv).loop_var

            def _visit_expr(e: tirx.Var):
                if e in block_var_to_loop_var:
                    spatial_loops.add(block_var_to_loop_var[e])

            for tensor_read in spatial_block.reads:
                tensor = tensor_read.source
                if tensor in reduced_tensors:
                    for read_range in tensor_read.region:
                        tvm_ffi.structural_walk(
                            read_range.min, (tirx.Var, _visit_expr), order="post"
                        )
                        tvm_ffi.structural_walk(
                            read_range.extent, (tirx.Var, _visit_expr), order="post"
                        )

            s_loops = []
            other_loops = []
            for loop_rv in loops:
                loop = sch.get(loop_rv)
                if loop.loop_var in spatial_loops or loop.extent == 1:
                    s_loops.append(loop_rv)
                else:
                    other_loops.append(loop_rv)
            sch.reorder(*s_loops, *other_loops)

        small_reduction = self._schedule_small_reduction(sch, block_infos, len_tx, unroll_depth)
        if small_reduction is not None:
            return small_reduction

        loops = sch.get_loops(block_infos[-1].block_rv)
        bx = sch.fuse(*loops[:num_leading_s])
        r_loop, tx = sch.split(loops[-1], [None, len_tx])
        sch.reorder(tx, r_loop)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.annotate(r_loop, ann_key="tirx.auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(r_loop, ann_key="tirx.unroll_explicit", ann_val=1)

        for block in reversed(block_infos[:-1]):
            block = block.block_rv
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, tensor_index=i, storage_scope="shared")
            sch.compute_at(block, bx, preserve_unit_loops=True)
            r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
            r_loop, tx = sch.split(r_loop, [None, len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(tx, "threadIdx.x")
            sch.annotate(r_loop, ann_key="tirx.auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(r_loop, ann_key="tirx.unroll_explicit", ann_val=1)

        # TODO: It's just a workaround to avoid unroll spatial loops, because of the bug of
        # the pass lower-thread-allreduce. We should fix it in the future.
        # sch.annotate(bx, ann_key="tirx.auto_unroll_max_step", ann_val=unroll_depth)
        # sch.annotate(bx, ann_key="tirx.unroll_explicit", ann_val=1)
        return sch

    def _schedule_small_reduction(  # pylint: disable=too-many-locals
        self,
        sch: s_tir.Schedule,
        block_infos: list[SBlockInfo],
        len_tx: int,
        unroll_depth: int,
    ) -> s_tir.Schedule | None:
        """Fold a small reduction serially inside each thread.

        Splitting a reduction of a handful of elements across `len_tx` threads
        leaves nearly every thread idle and launches one block per output. When the
        reduction is small, one output per thread is better: each thread reduces
        its own elements in a register and applies the epilogue.

        Following Kennedy and McKinley (LCPC 1993), a fused schedule must not give up
        parallelism, so this only applies when all of the following hold. Otherwise
        it returns None, without touching `sch`, and the default path runs.

        * every producer's reduction is static and at most `_SMALL_REDUCTION_MAX_EXTENT`;
        * the outputs are static and fill at least one block of `len_tx` threads;
        * every producer has exactly as many spatial elements as the final block, and
          each later block reads a producer value at a single point. A producer
          broadcast over a reduced axis (softmax, layer norm statistics) is therefore
          left alone, since one-thread-per-output would recompute it per element.
        """
        producers = block_infos[:-1]
        for info in producers:
            reduction_extent = _static_extent(info, "R")
            if reduction_extent is None or reduction_extent > _SMALL_REDUCTION_MAX_EXTENT:
                return None

        spatial_loops = sch.get_loops(block_infos[-1].block_rv)
        num_outputs = 1
        for loop_rv in spatial_loops:
            extent = get_extent(sch, loop_rv)
            if not isinstance(extent, int):
                return None
            num_outputs *= extent
        if num_outputs < len_tx:
            return None

        analyzer = sym.Analyzer()
        produced = []
        for info in block_infos:
            block = sch.get(info.block_rv)
            for read in block.reads:
                if any(read.source.same_as(t) for t in produced) and not all(
                    analyzer.can_prove_equal(rng.extent, 1) for rng in read.region
                ):
                    return None
            if info is not block_infos[-1]:
                if _static_extent(info, "S") != num_outputs:
                    return None
                produced.extend(write.source for write in block.writes)

        fused = sch.fuse(*spatial_loops)
        bx, tx = sch.split(fused, [None, len_tx])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        for info in reversed(producers):
            block = info.block_rv
            # Each thread owns its element, so keep it in a register, not shared memory.
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, tensor_index=i, storage_scope="local")
            # Compute at the thread loop, not the block loop: computing at `bx` would
            # make every thread redo the whole block's reductions.
            sch.compute_at(block, tx, preserve_unit_loops=True)
            inner_loops = sch.get_loops(block)
            sch.annotate(inner_loops[-1], ann_key="tirx.auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(inner_loops[-1], ann_key="tirx.unroll_explicit", ann_val=1)
        return sch
