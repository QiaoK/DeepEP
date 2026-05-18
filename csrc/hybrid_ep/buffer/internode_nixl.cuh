// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once

#ifdef USE_NIXL

#include "buffer/nixl_connector.h"
#include "coordinator.cuh"
#include <cuda_runtime.h>
#include <memory>

// Pack user-side token + prob (+ optional FP8 scaling factor) tensors into
// the NIXL send-staging buffer with destination-major, per-token-strided
// layout:
//     [num_of_nodes][max_tokens][packed_per_token_stride]
// where the per-token entry is
//     [token_bytes | prob_bytes | sf_bytes_if_FP8]
// All entries are populated for every dest so the N2N warp can issue one
// packed nixlPut covering a contiguous token run regardless of routing;
// the receiver gates consumption by its own rdma_to_attn_map. This collapses
// the dispatch send path from 2-3 puts per run (token + prob + [sf]) down
// to a single packed put per run.
//
// `src_sf` may be nullptr for BF16. `token_bytes_per_token` must equal
// `hidden_dim * sizeof(TOKEN_DATA_TYPE)`. The kernel performs the per-byte
// copy via vectorized 16-byte transfers when source/dest are 16B-aligned.
void pack_dispatch_for_nixl(
    const void* src_token,
    const float* src_prob,
    const float* src_sf,
    void* dst_packed,
    int num_tokens,
    int max_tokens_per_rank,
    int num_of_nodes,
    size_t token_bytes_per_token,
    size_t prob_bytes_per_token,
    size_t sf_bytes_per_token,
    size_t packed_stride,
    cudaStream_t stream);

class NIXLCoordinator : public InterNodeCoordinator {
public:
    NIXLCoordinator() = default;
    ~NIXLCoordinator() override;

    void init(pybind11::object process_group, int node_rank, int local_rank, BufferConfig config) override;
    bool grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) override;
    void update_config(BufferConfig config) override;
    void allocate_buffers() override;
    void destroy() override;

    InterNodeDispatchBuffers& get_dispatch_buffers() override { return dispatch_buffers; }
    InterNodeCombineBuffers& get_combine_buffers() override { return combine_buffers; }

    InterNodeDispatchBuffers dispatch_buffers;
    InterNodeCombineBuffers combine_buffers;

private:
    void allocate_dispatch_buffers();
    void allocate_combine_buffers();
    void free_buffers();

    pybind11::object process_group;
    int node_rank = -1;
    int local_rank = -1;
    BufferConfig buffer_config;
    std::unique_ptr<hybrid_ep::HybridEP_NIXLConnector> nixl_connector;
    bool buffer_allocated = false;
};

#endif  // USE_NIXL
