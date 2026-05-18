// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#ifdef USE_NIXL

#include "buffer/internode.cuh"
#include "buffer/internode_nixl.cuh"
#include <pybind11/pybind11.h>

// Pack the user-side token + prob (+ optional FP8 SF) tensors into a
// destination-major, per-token-strided NIXL send-staging layout:
//     [num_of_nodes][max_tokens][packed_stride]
// where each per-token entry is byte-laid-out as
//     [token_bytes | prob_bytes | sf_bytes_if_FP8].
// Every (dest, token) slot is populated regardless of routing; the
// dispatch N2N warp issues a single packed put per contiguous token
// run, and the receiver gates consumption via its own rdma_to_attn_map.
// One block handles one (dest, token) pair; threads cooperatively copy
// all three sub-regions with 16-byte vector transfers.
namespace {

constexpr int PACK_BLOCK_THREADS = 128;
constexpr size_t PACK_VEC_BYTES = 16;

// Bulk copy `n_bytes` from `src` to `dst` using uint4 (16-byte) vector loads
// when possible; falls back to byte copy for any tail. Assumes the bulk of
// the region is 16-byte aligned (which our packed layout guarantees by
// construction).
__device__ __forceinline__ void packed_memcpy_warp(
    void* __restrict__ dst, const void* __restrict__ src, size_t n_bytes, int tid, int nthreads)
{
    const size_t n_vec = n_bytes / PACK_VEC_BYTES;
    const size_t tail = n_bytes - n_vec * PACK_VEC_BYTES;
    const uint4* sv = reinterpret_cast<const uint4*>(src);
    uint4* dv = reinterpret_cast<uint4*>(dst);
    for (size_t i = tid; i < n_vec; i += nthreads) {
        dv[i] = sv[i];
    }
    if (tail) {
        const uint8_t* sb = reinterpret_cast<const uint8_t*>(src) + n_vec * PACK_VEC_BYTES;
        uint8_t* db = reinterpret_cast<uint8_t*>(dst) + n_vec * PACK_VEC_BYTES;
        for (size_t i = tid; i < tail; i += nthreads) {
            db[i] = sb[i];
        }
    }
}

// NB: kernel name intentionally avoids the substring "dispatch_kernel" so
// that test harness profiling-table grep (bench_kineto) keeps matching the
// real `dispatch_kernel` uniquely.
__global__ void nixl_pack_kernel(
    const uint8_t* __restrict__ src_token,
    const float* __restrict__ src_prob,
    const float* __restrict__ src_sf,
    uint8_t* __restrict__ dst_packed,
    int num_tokens,
    int max_tokens_per_rank,
    int num_of_nodes,
    size_t token_bytes,
    size_t prob_bytes,
    size_t sf_bytes,
    size_t packed_stride)
{
    const int dest  = blockIdx.y;
    const int token = blockIdx.x;
    if (token >= num_tokens) return;

    const size_t token_row_off = static_cast<size_t>(token) * token_bytes;
    const size_t prob_row_off  = (static_cast<size_t>(token) * num_of_nodes + dest) * prob_bytes;
    const size_t dst_entry_off = (static_cast<size_t>(dest) * max_tokens_per_rank + token) * packed_stride;

    uint8_t* dst_token = dst_packed + dst_entry_off;
    uint8_t* dst_prob  = dst_token  + token_bytes;
    uint8_t* dst_sf    = dst_prob   + prob_bytes;

    packed_memcpy_warp(dst_token, src_token + token_row_off,
                       token_bytes, threadIdx.x, blockDim.x);
    // src_prob may be null for backward dispatch; the prob sub-region in
    // the packed buffer is not consumed in that case so we leave it
    // untouched (saves HBM bandwidth on the pack path).
    if (src_prob) {
        packed_memcpy_warp(dst_prob,
                           reinterpret_cast<const uint8_t*>(src_prob) + prob_row_off,
                           prob_bytes, threadIdx.x, blockDim.x);
    }
    if (sf_bytes && src_sf) {
        const size_t sf_row_off = static_cast<size_t>(token) * sf_bytes;
        packed_memcpy_warp(dst_sf,
                           reinterpret_cast<const uint8_t*>(src_sf) + sf_row_off,
                           sf_bytes, threadIdx.x, blockDim.x);
    }
}

}  // anonymous

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
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_of_nodes <= 0 || packed_stride == 0) return;
    assert(token_bytes_per_token + prob_bytes_per_token + sf_bytes_per_token == packed_stride);
    dim3 grid(num_tokens, num_of_nodes);
    nixl_pack_kernel<<<grid, PACK_BLOCK_THREADS, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src_token),
        src_prob,
        src_sf,
        reinterpret_cast<uint8_t*>(dst_packed),
        num_tokens,
        max_tokens_per_rank,
        num_of_nodes,
        token_bytes_per_token,
        prob_bytes_per_token,
        sf_bytes_per_token,
        packed_stride);
}

NIXLCoordinator::~NIXLCoordinator() {
    destroy();
}

void NIXLCoordinator::init(
    pybind11::object process_group,
    int node_rank,
    int local_rank,
    BufferConfig config
) {
    this->process_group = process_group;
    this->node_rank = node_rank;
    this->local_rank = local_rank;
    this->buffer_config = config;
    assert(buffer_config.num_of_nodes > 1);
}

bool NIXLCoordinator::grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) {
    bool changed = false;
    changed |= grow_to(buf_config.max_num_of_tokens_per_rank, config.max_num_of_tokens_per_rank);
    changed |= grow_to(buf_config.hidden_dim, config.hidden_dim);
    changed |= grow_to(buf_config.num_of_experts_per_rank, config.num_of_experts_per_rank);
    changed |= grow_to(buf_config.num_of_ranks_per_node, config.num_of_ranks_per_node);
    changed |= grow_to(buf_config.num_of_nodes, config.num_of_nodes);
    changed |= grow_to(buf_config.num_of_blocks_dispatch_api, config.num_of_blocks_dispatch_api);
    changed |= grow_to(buf_config.num_of_blocks_combine_api, config.num_of_blocks_combine_api);
    if (buf_config.num_of_tokens_per_chunk_dispatch_api != config.num_of_tokens_per_chunk_dispatch_api) {
        changed = true;
        buf_config.num_of_tokens_per_chunk_dispatch_api = config.num_of_tokens_per_chunk_dispatch_api;
    }
    if (buf_config.num_of_tokens_per_chunk_combine_api != config.num_of_tokens_per_chunk_combine_api) {
        changed = true;
        buf_config.num_of_tokens_per_chunk_combine_api = config.num_of_tokens_per_chunk_combine_api;
    }
    return changed;
}

void NIXLCoordinator::update_config(BufferConfig config) {
    this->buffer_config = config;
}

void NIXLCoordinator::destroy() {
    if (!buffer_allocated) return;

    CUDA_CHECK(cudaDeviceSynchronize());

    nixl_connector.reset();

    free_buffers();
    buffer_allocated = false;

    CUDA_CHECK(cudaDeviceSynchronize());
}

void NIXLCoordinator::free_buffers() {
    auto free_ptr = [](auto*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };

    free_ptr(dispatch_buffers.attn_input_token);
    free_ptr(dispatch_buffers.attn_input_prob);
    free_ptr(dispatch_buffers.attn_input_flags);
    free_ptr(dispatch_buffers.attn_input_scaling_factor);
    free_ptr(dispatch_buffers.rdma_inter_node_group_token);
    free_ptr(dispatch_buffers.rdma_inter_node_group_prob);
    free_ptr(dispatch_buffers.rdma_inter_node_group_scaling_factor);
    free_ptr(dispatch_buffers.rdma_inter_node_group_flags);
    free_ptr(dispatch_buffers.expected_rdma_flag_value);
    free_ptr(dispatch_buffers.attn_input_packed);
    free_ptr(dispatch_buffers.rdma_inter_node_group_packed);

    free_ptr(combine_buffers.attn_output_flags);
    free_ptr(combine_buffers.rdma_intra_node_red_token);
    free_ptr(combine_buffers.rdma_intra_node_red_prob);
    free_ptr(combine_buffers.rdma_inter_node_group_token);
    free_ptr(combine_buffers.rdma_inter_node_group_prob);
    free_ptr(combine_buffers.rdma_inter_node_group_flags);
    free_ptr(combine_buffers.expected_rdma_flag_value);
}

void NIXLCoordinator::allocate_dispatch_buffers() {
    dispatch_buffers.data_type = buffer_config.token_data_type;
    const size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);
    const bool use_fp8 = (dispatch_buffers.data_type == APP_TOKEN_DATA_TYPE::UINT8);

    // Packed per-token layout: [token_bytes | prob_bytes | sf_bytes_if_FP8].
    // Each sub-region is 16B-aligned by construction for HIDDEN_DIM and
    // prob counts the kernel already validates.
    const size_t token_bytes = static_cast<size_t>(buffer_config.hidden_dim) * sizeof_token_data_type;
    const size_t prob_bytes  = static_cast<size_t>(buffer_config.num_of_experts_per_rank)
                             * buffer_config.num_of_ranks_per_node * sizeof(float);
    const size_t sf_bytes    = use_fp8
                             ? static_cast<size_t>(buffer_config.hidden_dim / 128) * sizeof(float)
                             : 0;
    const size_t packed_stride = token_bytes + prob_bytes + sf_bytes;

    dispatch_buffers.packed_per_token_stride = packed_stride;
    dispatch_buffers.packed_token_offset = 0;
    dispatch_buffers.packed_prob_offset  = token_bytes;
    dispatch_buffers.packed_sf_offset    = token_bytes + prob_bytes;

    // Send-side packed staging holds all NUM_OF_NODES slots (including the
    // local-node slot so the G2S local-read path uses the same packed
    // layout, removing the need for a separate attn_input_* buffer set).
    dispatch_buffers.attn_input_packed_sz =
        static_cast<size_t>(buffer_config.num_of_nodes)
        * buffer_config.max_num_of_tokens_per_rank * packed_stride;
    dispatch_buffers.rdma_inter_node_group_packed_sz =
        static_cast<size_t>(buffer_config.num_of_nodes - 1)
        * buffer_config.max_num_of_tokens_per_rank * packed_stride;

    auto rdma_inter_node_group_flags_elts = ((buffer_config.max_num_of_tokens_per_rank - 1) /
        buffer_config.num_of_tokens_per_chunk_dispatch_api + 1) * (buffer_config.num_of_nodes - 1);
    dispatch_buffers.rdma_inter_node_group_flags_sz = rdma_inter_node_group_flags_elts * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_packed,
                          dispatch_buffers.attn_input_packed_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_packed,
                          dispatch_buffers.rdma_inter_node_group_packed_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_flags,
                          dispatch_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMemset(dispatch_buffers.rdma_inter_node_group_flags, 0,
                          dispatch_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_flags,
                          dispatch_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMemset(dispatch_buffers.attn_input_flags, 0,
                          dispatch_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(dispatch_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
}

void NIXLCoordinator::allocate_combine_buffers() {
    auto rdma_intra_node_red_token_elts = buffer_config.max_num_of_tokens_per_rank *
        (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
    auto rdma_intra_node_red_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
        (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
    auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank *
        (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
    auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
        (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
    auto rdma_inter_node_group_flags_elts = ((buffer_config.max_num_of_tokens_per_rank - 1) /
        buffer_config.num_of_tokens_per_chunk_combine_api + 1) * (buffer_config.num_of_nodes - 1);

    combine_buffers.rdma_intra_node_red_token_sz = rdma_intra_node_red_token_elts * sizeof(uint16_t);
    combine_buffers.rdma_intra_node_red_prob_sz = rdma_intra_node_red_prob_elts * sizeof(float);
    combine_buffers.rdma_inter_node_group_token_sz = rdma_inter_node_group_token_elts * sizeof(uint16_t);
    combine_buffers.rdma_inter_node_group_prob_sz = rdma_inter_node_group_prob_elts * sizeof(float);
    combine_buffers.rdma_inter_node_group_flags_sz = rdma_inter_node_group_flags_elts * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_token, combine_buffers.rdma_intra_node_red_token_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_prob, combine_buffers.rdma_intra_node_red_prob_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_token, combine_buffers.rdma_inter_node_group_token_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_prob, combine_buffers.rdma_inter_node_group_prob_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_flags, combine_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMemset(combine_buffers.rdma_inter_node_group_flags, 0, combine_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.attn_output_flags, combine_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMemset(combine_buffers.attn_output_flags, 0, combine_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(combine_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
}

void NIXLCoordinator::allocate_buffers() {
    allocate_combine_buffers();
    allocate_dispatch_buffers();

    int rank_uuid = node_rank * buffer_config.num_of_ranks_per_node + local_rank;
    int num_ranks = buffer_config.num_of_ranks_per_node * buffer_config.num_of_nodes;

    nixl_connector = std::make_unique<hybrid_ep::HybridEP_NIXLConnector>(rank_uuid, local_rank);

    nixl_connector->updateMemoryBuffers(
        num_ranks,
        buffer_config.num_of_experts_per_rank,
        buffer_config.num_of_nodes,
        buffer_config.num_of_ranks_per_node,
        buffer_config.num_of_blocks_dispatch_api,
        buffer_config.num_of_blocks_combine_api,
        dispatch_buffers,
        combine_buffers);

    auto torch_distributed = pybind11::module_::import("torch.distributed");
    torch_distributed.attr("barrier")(this->process_group);

    // sendLocalMD is async — metadata publication to etcd happens in a background
    // thread.  The barrier above ensures all ranks have *called* sendLocalMD, but
    // the etcd writes may not yet be visible.  A brief sleep reduces spurious
    // invalidate+refetch cycles in _nixl_agents_connect.
    {
        const char* env = std::getenv("DEEPEP_NIXL_POST_BARRIER_MS");
        int delay_ms = env ? std::atoi(env) : 2000;
        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }

    std::vector<int> remote_rank_uuids;
    for (int node_idx = 0; node_idx < buffer_config.num_of_nodes; ++node_idx) {
        if (node_idx != node_rank) {
            remote_rank_uuids.push_back(node_idx * buffer_config.num_of_ranks_per_node + local_rank);
        }
    }
    nixl_connector->connectRanks(remote_rank_uuids);

    dispatch_buffers.nixl_gpu_ctx = nixl_connector->get_dispatch_gpu_ctx();
    combine_buffers.nixl_gpu_ctx = nixl_connector->get_combine_gpu_ctx();

    buffer_allocated = true;
}

#endif  // USE_NIXL
