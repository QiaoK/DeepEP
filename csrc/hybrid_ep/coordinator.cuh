// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once
#include "config.cuh"
#include <pybind11/pybind11.h>

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#include "utils.cuh"
#ifdef USE_NIXL
namespace hybrid_ep { struct dispatch_gpu_nixl_ctx; struct combine_gpu_nixl_ctx; }
#else
struct doca_gpu_dev_verbs_qp;
#endif
#endif

class HybridEPCoordinator {
public:
    virtual ~HybridEPCoordinator() = default;
    virtual bool grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) = 0;
    virtual void update_config(BufferConfig config) = 0;
    virtual void allocate_buffers() = 0;
    virtual void destroy() = 0;
};

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE

struct InterNodeDispatchBuffers {
    APP_TOKEN_DATA_TYPE data_type;
    // Input buffers from attn, only used in inter-node case
    void *        attn_input_token = nullptr;
    size_t       attn_input_token_sz = 0;
    void *        attn_input_prob = nullptr;
    size_t       attn_input_prob_sz = 0;
    void *        attn_input_flags = nullptr;
    void *        attn_input_scaling_factor = nullptr;
    size_t       attn_input_scaling_factor_sz = 0;
    // RDMA buffers for dispatch kernel.
    void *        rdma_inter_node_group_token = nullptr;
    size_t       rdma_inter_node_group_token_sz = 0;
    float *       rdma_inter_node_group_prob = nullptr;
    size_t       rdma_inter_node_group_prob_sz = 0;
    float *       rdma_inter_node_group_scaling_factor = nullptr;
    size_t       rdma_inter_node_group_scaling_factor_sz = 0;
    uint64_t *    rdma_inter_node_group_flags = nullptr;
    size_t       rdma_inter_node_group_flags_sz = 0;
    uint64_t *    expected_rdma_flag_value = nullptr;
#ifdef USE_NIXL
    // Packed dispatch staging (NIXL only). Replaces the 3 separate
    // attn_input_{token,prob,scaling_factor} and
    // rdma_inter_node_group_{token,prob,scaling_factor} buffers with one
    // per-token-strided packed buffer per side. Per-token layout:
    //     [token_bytes | prob_bytes | sf_bytes_if_FP8]
    // packed_per_token_stride = HIDDEN*sizeof(TOKEN) + prob_per_token*4
    //                         + (FP8 ? HIDDEN/128*4 : 0).
    // attn_input_packed shape:           [NUM_OF_NODES][max_tokens][stride]
    // rdma_inter_node_group_packed shape:[NUM_OF_NODES-1][max_tokens][stride]
    // Collapses 2-3 nixlPuts per dispatch run down to 1 packed put.
    void *        attn_input_packed = nullptr;
    size_t       attn_input_packed_sz = 0;
    void *        rdma_inter_node_group_packed = nullptr;
    size_t       rdma_inter_node_group_packed_sz = 0;
    size_t       packed_per_token_stride = 0;
    size_t       packed_token_offset = 0;
    size_t       packed_prob_offset = 0;
    size_t       packed_sf_offset = 0;
#endif
    // Backend-specific
#ifndef USE_NIXL
    struct doca_gpu_dev_verbs_qp ** d_qps_gpu = nullptr;
    struct dispatch_memory_region_info_t * mr_info = nullptr;
#else
    hybrid_ep::dispatch_gpu_nixl_ctx * nixl_gpu_ctx = nullptr;
#endif
};

struct InterNodeCombineBuffers {
    // Output buffers to attn, only used in inter-node case
    void *        attn_output_flags = nullptr;
    // RDMA buffers for combine kernel.
    uint16_t *    rdma_intra_node_red_token = nullptr;
    size_t       rdma_intra_node_red_token_sz = 0;
    float *       rdma_intra_node_red_prob = nullptr;
    size_t       rdma_intra_node_red_prob_sz = 0;
    uint16_t *    rdma_inter_node_group_token = nullptr;
    size_t       rdma_inter_node_group_token_sz = 0;
    float *       rdma_inter_node_group_prob = nullptr;
    size_t       rdma_inter_node_group_prob_sz = 0;
    uint64_t *    rdma_inter_node_group_flags = nullptr;
    size_t       rdma_inter_node_group_flags_sz = 0;
    uint64_t *    expected_rdma_flag_value = nullptr;
    // Backend-specific
#ifndef USE_NIXL
    struct doca_gpu_dev_verbs_qp ** d_qps_gpu = nullptr;
    struct combine_memory_region_info_t * mr_info = nullptr;
#else
    hybrid_ep::combine_gpu_nixl_ctx * nixl_gpu_ctx = nullptr;
#endif
};

class InterNodeCoordinator : public HybridEPCoordinator {
public:
    virtual ~InterNodeCoordinator() = default;
    virtual void init(pybind11::object process_group, int node_rank, int local_rank, BufferConfig config) = 0;
    virtual InterNodeDispatchBuffers& get_dispatch_buffers() = 0;
    virtual InterNodeCombineBuffers& get_combine_buffers() = 0;
};

#endif  // HYBRID_EP_BUILD_MULTINODE_ENABLE
