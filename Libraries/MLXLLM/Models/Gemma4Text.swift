//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Compiled fusion fragments
//
// Gemma 4 uses a single rms_norm_eps (1e-6) for every RMSNorm in the model
// (see Gemma4TextConfiguration.rmsNormEps default — all upstream Gemma 4
// weights ship with this value). Hardcoding lets one compiled graph serve
// every layer without per-layer specialization.
//
// Mirrors the upstream mlx-lm Python optimization
// (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py)
// which fuses (residual + RMSNorm(x) * weight) and gelu(g) * other into a
// single compiled graph. Measured ~+2.4% decode tps on M4 Max for
// gemma-4-e2b-it-4bit at batch=1.

private let kRMSEps: Float = 1e-6

private let _addRMSNorm: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { residual, x, weight in
    residual + MLXFast.rmsNorm(x, weight: weight, eps: kRMSEps)
}

private let _geluMul: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, other in
    geluApproximate(gate) * other
}

// MARK: - Fused q4 gate+up+geglu Metal kernel
//
// On batch=1 / seq=1 decode, the MLP `gate_proj + up_proj + gelu * up`
// dispatches three GPU kernels (two q4 matmuls + an elementwise activation),
// touches `intermediate * 2` bf16 of intermediate tensors, and pays per-launch
// overhead for each. This custom kernel folds them into one launch:
//
//   - load each x lane once into thread storage (shared across both rows of
//     the MLP), reusing the q4 dot-product layout from mlx core's qmv_fast_impl
//     (mlx/backend/metal/kernels/quantized.h): num_simdgroups=2,
//     results_per_simdgroup=4, packs_per_thread=2, block_size=512.
//   - accumulate gate and up dot products in parallel registers.
//   - after simd_sum, lane 0 of each row writes
//     `gated[row] = gelu_approx(g) * u` directly, skipping the two
//     intermediate activation tensors entirely.
//
// Eligibility is narrow on purpose: batch=1, seq=1, bf16, 4-bit/group=64
// quantized linears with both biases (the only layout shipped by every
// Gemma 4 mlx-community/unsloth weight at this time), and `hidden % 512 == 0`,
// `intermediate % 8 == 0`. Anything outside these bounds falls back to the
// reference path.

private let _fusedGateUpGegluHeader = """
constant constexpr int SIMD_SIZE = 32;
constant constexpr int PACK_FACTOR = 8;
constant constexpr int BYTES_PER_PACK = 4;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int VALUES_PER_THREAD = 16;
constant constexpr int BLOCK_SIZE = 512;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int ROWS_PER_TG = 8;
constant constexpr int GROUP_SIZE = 64;
constant constexpr int SCALE_STEP_PER_THREAD = 4;

inline float fused_load_vec(const device bfloat16_t* x, thread float* x_thread) {
    float sum = 0.0f;
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        float a = (float)x[i];
        float b = (float)x[i + 1];
        float c = (float)x[i + 2];
        float d = (float)x[i + 3];
        sum += a + b + c + d;
        x_thread[i]     = a;
        x_thread[i + 1] = b / 16.0f;
        x_thread[i + 2] = c / 256.0f;
        x_thread[i + 3] = d / 4096.0f;
    }
    return sum;
}

inline float fused_qdot4(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale, float bias, float sum)
{
    const device uint16_t* ws = (const device uint16_t*)w;
    float accum = 0.0f;
    for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
        accum +=
            (x_thread[4 * i]     * (ws[i] & 0x000f) +
             x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
             x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
             x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
    return scale * accum + sum * bias;
}
"""

private let _fusedGateUpGegluBody = """
    const uint tid_y = threadgroup_position_in_grid.y;
    const uint local_id = thread_position_in_threadgroup.x;
    const uint simd_gid = local_id / SIMD_SIZE;
    const uint simd_lid = local_id % SIMD_SIZE;

    const uint in_vec_size_w = HIDDEN * BYTES_PER_PACK / PACK_FACTOR;
    const uint in_vec_size_g = HIDDEN / GROUP_SIZE;
    const uint out_row = tid_y * ROWS_PER_TG + simd_gid * RESULTS_PER_SIMDGROUP;

    const device uint8_t* gw = (const device uint8_t*)gate_w
        + out_row * in_vec_size_w + simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device uint8_t* uw = (const device uint8_t*)up_w
        + out_row * in_vec_size_w + simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device bfloat16_t* gs = gate_s + out_row * in_vec_size_g + simd_lid / SCALE_STEP_PER_THREAD;
    const device bfloat16_t* gb = gate_b + out_row * in_vec_size_g + simd_lid / SCALE_STEP_PER_THREAD;
    const device bfloat16_t* us = up_s   + out_row * in_vec_size_g + simd_lid / SCALE_STEP_PER_THREAD;
    const device bfloat16_t* ub = up_b   + out_row * in_vec_size_g + simd_lid / SCALE_STEP_PER_THREAD;
    const device bfloat16_t* xp = x + simd_lid * VALUES_PER_THREAD;

    thread float x_thread[VALUES_PER_THREAD];
    thread float g_acc[RESULTS_PER_SIMDGROUP] = {0};
    thread float u_acc[RESULTS_PER_SIMDGROUP] = {0};

    for (uint k = 0; k < HIDDEN; k += BLOCK_SIZE) {
        float xsum = fused_load_vec(xp, x_thread);
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
            const device uint8_t* gwl = gw + row * in_vec_size_w;
            const device uint8_t* uwl = uw + row * in_vec_size_w;
            float gscale = (float)gs[row * in_vec_size_g];
            float gbias  = (float)gb[row * in_vec_size_g];
            float uscale = (float)us[row * in_vec_size_g];
            float ubias  = (float)ub[row * in_vec_size_g];
            g_acc[row] += fused_qdot4(gwl, x_thread, gscale, gbias, xsum);
            u_acc[row] += fused_qdot4(uwl, x_thread, uscale, ubias, xsum);
        }
        gw += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        uw += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        gs += BLOCK_SIZE / GROUP_SIZE;
        gb += BLOCK_SIZE / GROUP_SIZE;
        us += BLOCK_SIZE / GROUP_SIZE;
        ub += BLOCK_SIZE / GROUP_SIZE;
        xp += BLOCK_SIZE;
    }

    for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        float g = simd_sum(g_acc[row]);
        float u = simd_sum(u_acc[row]);
        if (simd_lid == 0) {
            float t = 0.7978845608f * (g + 0.044715f * g * g * g);
            float gelu_g = 0.5f * g * (1.0f + metal::tanh(t));
            gated[out_row + row] = (bfloat16_t)(gelu_g * u);
        }
    }
"""

// Cache compiled kernels per (hidden, intermediate) shape. Inference is
// single-threaded from the harness's perspective, so simple unsynchronized
// access is fine; if the model is ever evaluated from multiple threads
// concurrently we would race here.
private nonisolated(unsafe) var _fusedKernelCache: [String: MLXFast.MLXFastKernel] = [:]

private func _getFusedGateUpKernel(hidden: Int, intermediate: Int) -> MLXFast.MLXFastKernel {
    let key = "h\(hidden)_i\(intermediate)"
    if let k = _fusedKernelCache[key] { return k }
    let header = _fusedGateUpGegluHeader
        + "\nconstant constexpr int HIDDEN = \(hidden);"
        + "\nconstant constexpr int INTERMEDIATE = \(intermediate);\n"
    let k = MLXFast.metalKernel(
        name: "fused_gateup_geglu_q4_\(key)",
        inputNames: ["x", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"],
        outputNames: ["gated"],
        source: _fusedGateUpGegluBody,
        header: header
    )
    _fusedKernelCache[key] = k
    return k
}

/// Run the fused q4 gate+up+geglu kernel if the inputs are eligible
/// (batch=1, seq=1, bf16, 4-bit/group=64 quantized, expected divisibility).
/// Returns nil if the path isn't supported (caller should fall back).
private func _fusedGateUpGeglu(
    x: MLXArray, gate: QuantizedLinear, up: QuantizedLinear, intermediate: Int
) -> MLXArray? {
    guard
        x.dtype == .bfloat16,
        x.shape.count == 3, x.shape[0] == 1, x.shape[1] == 1,
        gate.bits == 4, gate.groupSize == 64,
        up.bits == 4, up.groupSize == 64,
        let gateBiases = gate.biases, let upBiases = up.biases
    else { return nil }
    let hidden = x.shape[2]
    if hidden % 512 != 0 || intermediate % 8 != 0 { return nil }
    let kernel = _getFusedGateUpKernel(hidden: hidden, intermediate: intermediate)
    let out = kernel(
        [x, gate.weight, gate.scales, gateBiases, up.weight, up.scales, upBiases],
        grid: (64, intermediate / 8, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, intermediate]],
        outputDTypes: [x.dtype]
    )
    return out[0]
}

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    var modelType: String = "gemma4_text"
    var hiddenSize: Int = 1536
    var numHiddenLayers: Int = 35
    var intermediateSize: Int = 6144
    var numAttentionHeads: Int = 8
    var headDim: Int = 256
    var globalHeadDim: Int = 512
    var globalPartialRotaryFactor: Float = 0.25
    var rmsNormEps: Float = 1e-6
    var vocabSize: Int = 262144
    var vocabSizePerLayerInput: Int = 262144
    var numKeyValueHeads: Int = 1
    var numGlobalKeyValueHeads: Int?
    var numKvSharedLayers: Int = 20
    var hiddenSizePerLayerInput: Int = 256
    var slidingWindow: Int = 512
    var slidingWindowPattern: Int = 5
    var maxPositionEmbeddings: Int = 131072
    var attentionKeqV: Bool = false
    var finalLogitSoftcapping: Float = 30.0
    var useDoubleWideMlp: Bool = true
    var layerTypes: [String] = []
    var tieWordEmbeddings: Bool = true

    // RoPE parameters (nested dict with full_attention/sliding_attention sub-configs)
    var ropeParameters: [String: [String: StringOrNumber]]?

    // Derived properties
    var slidingRopeTheta: Float = 10000.0
    var fullRopeTheta: Float = 1_000_000.0
    var fullPartialRotaryFactor: Float = 1.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case globalPartialRotaryFactor = "global_partial_rotary_factor"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKeqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 35
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        self.globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        self.globalPartialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .globalPartialRotaryFactor) ?? 0.25
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        self.vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 262144
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        self.numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        self.numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 20
        self.hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 256
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.attentionKeqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKeqV) ?? false
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        self.useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? true
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = decoded
        } else {
            // Derive layer types from sliding window pattern
            var pattern = [String]()
            for i in 0 ..< slidingWindowPattern {
                pattern.append(
                    i == slidingWindowPattern - 1 ? "full_attention" : "sliding_attention")
            }
            var types = [String]()
            while types.count < numHiddenLayers {
                types.append(contentsOf: pattern)
            }
            self.layerTypes = Array(types.prefix(numHiddenLayers))
        }
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        // Extract RoPE parameters from nested config
        if let ropeParams = ropeParameters {
            if let sliding = ropeParams["sliding_attention"] {
                self.slidingRopeTheta = sliding["rope_theta"]?.asFloat() ?? 10000.0
            }
            if let full = ropeParams["full_attention"] {
                self.fullRopeTheta = full["rope_theta"]?.asFloat() ?? 1_000_000.0
                self.fullPartialRotaryFactor =
                    full["partial_rotary_factor"]?.asFloat() ?? 1.0
            }
        }
    }
}

// MARK: - Helper Modules

private class RMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private class ScaledLinear: Module {
    let weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outFeatures, inFeatures])
        self.scalar = scalar
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T) * scalar
    }
}

private enum Gemma4PositionOffset {
    case scalar(Int)
    case batch(MLXArray)
}

private func gemma4CapturePositionOffset(from cache: KVCache?) -> Gemma4PositionOffset {
    if let batchCache = cache as? BatchPositionedKVCache {
        // Snapshot the per-sequence offsets before cache.update(...) advances them.
        .batch(batchCache.batchOffset + 0)
    } else {
        .scalar(cache?.offset ?? 0)
    }
}

private func gemma4ApplyRotaryPosition<R: RoPELayer>(
    _ rope: R,
    to x: MLXArray,
    offset: Gemma4PositionOffset
) -> MLXArray {
    switch offset {
    case .scalar(let value):
        rope(x, offset: value)
    case .batch(let values):
        rope(x, offset: values)
    }
}

// MARK: - Attention

private class Gemma4Attention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let nKvHeads: Int
    let useKeqV: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale

    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"

        // Full attention uses globalHeadDim, sliding uses headDim
        self.effectiveHeadDim =
            isSliding ? config.headDim : config.globalHeadDim

        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads

        // K-eq-V for full attention layers
        self.useKeqV = config.attentionKeqV && !isSliding
        if useKeqV, let globalKvHeads = config.numGlobalKeyValueHeads {
            self.nKvHeads = globalKvHeads
        } else {
            self.nKvHeads = config.numKeyValueHeads
        }

        self.scale = 1.0

        self._qProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
        if !useKeqV {
            self._vProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
        }
        self._oProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)

        // RoPE: sliding uses default, full uses proportional with partial rotation
        if isSliding {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.fullRopeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        positionOffset: Gemma4PositionOffset? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4PositionOffset) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = qNorm(queries)

        let keys: MLXArray
        let values: MLXArray
        let activePositionOffset = positionOffset ?? gemma4CapturePositionOffset(from: cache)

        if let (sharedK, sharedV) = sharedKV {
            // KV-shared layers use pre-computed KV from an earlier layer
            keys = sharedK
            values = sharedV
        } else {
            var k = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            k = kNorm(k)
            k = k.transposed(0, 2, 1, 3)
            k = gemma4ApplyRotaryPosition(rope, to: k, offset: activePositionOffset)

            var v: MLXArray
            if let vProj {
                v = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            } else {
                v = k
            }
            v = vNorm(v)
            v = v.transposed(0, 2, 1, 3)

            if let cache {
                let (updatedK, updatedV) = cache.update(keys: k, values: v)
                keys = updatedK
                values = updatedV
            } else {
                keys = k
                values = v
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: activePositionOffset)

        // Adjust mask if cache size differs from mask size
        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.dim(2)
            if maskArray.dim(-1) != keysSeqLen {
                adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: adjustedMask ?? .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return (oProj(output), (keys, values), activePositionOffset)
    }
}

// MARK: - MLP

private class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let intermediateSize: Int

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        let isKvSharedLayer = layerIdx >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0
        let useDoubleWide = config.useDoubleWideMlp && isKvSharedLayer
        self.intermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Fast path: fused q4 gate+up+geglu kernel for single-token decode.
        if let g = gateProj as? QuantizedLinear, let u = upProj as? QuantizedLinear,
            let gated = _fusedGateUpGeglu(x: x, gate: g, up: u, intermediate: intermediateSize)
        {
            return downProj(gated)
        }
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

private class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    // Per-layer input (PLE) gating
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    // Per-layer scalar
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)

        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1], dtype: .float16)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        positionOffset: Gemma4PositionOffset? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4PositionOffset) {
        let residual = x

        let h = inputLayernorm(x)
        let (attnOut, kvPair, attnPositionOffset) = selfAttn(
            h, mask: mask, cache: cache, sharedKV: sharedKV, positionOffset: positionOffset)
        // Fused: residual + RMSNorm(attnOut) * weight
        var out = _addRMSNorm(residual, attnOut, postAttentionLayernorm.weight)

        let residual2 = out
        out = preFeedforwardLayernorm(out)
        out = mlp(out)
        // Fused: residual + RMSNorm(out) * weight
        out = _addRMSNorm(residual2, out, postFeedforwardLayernorm.weight)

        // PLE gating
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let perLayerInput
        {
            let residual3 = out
            var g = gate(out)
            // Fused: gelu_approx(g) * perLayerInput
            g = _geluMul(g, perLayerInput)
            g = proj(g)
            // Fused: residual + RMSNorm(g) * weight
            out = _addRMSNorm(residual3, g, norm.weight)
        }

        out = out * layerScalar

        return (out, kvPair, attnPositionOffset)
    }
}

// MARK: - Text Model

private class Gemma4TextModelInner: Module {
    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    // Per-layer embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    // KV sharing mapping: for each layer, which earlier layer provides KVs
    let previousKvs: [Int]
    let firstKvSharedLayerIdx: Int

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = Float(config.hiddenSize).squareRoot()
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // PLE
        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput)
            self._perLayerModelProjection.wrappedValue = ScaledLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.numHiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5))
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        // Build KV-sharing map
        self.firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        var kvMap = Array(0 ..< config.numHiddenLayers)
        if config.numKvSharedLayers > 0 {
            // Find the last non-shared layer of each type
            var lastByType = [String: Int]()
            for i in 0 ..< firstKvSharedLayerIdx {
                lastByType[config.layerTypes[i]] = i
            }
            // Shared layers reference the last non-shared layer of the same type
            for j in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                if let prev = lastByType[config.layerTypes[j]] {
                    kvMap[j] = prev
                }
            }
        }
        self.previousKvs = kvMap

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let inputEmbeddings = embedTokens(inputs)
        var h = inputEmbeddings * embedScale

        // Compute per-layer inputs (PLE)
        var perLayerInputs: [MLXArray?]
        if hiddenSizePerLayerInput > 0,
            let embedPerLayer = embedTokensPerLayer,
            let modelProj = perLayerModelProjection,
            let projNorm = perLayerProjectionNorm
        {
            // Token-based PLE
            let tokenPLE =
                embedPerLayer(inputs)
                * Float(config.hiddenSizePerLayerInput).squareRoot()

            // [B, L, numLayers * hiddenSizePerLayerInput] -> [B, L, numLayers, hiddenSizePerLayerInput]
            let reshapedTokenPLE = tokenPLE.reshaped(
                tokenPLE.dim(0), tokenPLE.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)

            // Model projection PLE
            let modelPLE = modelProj(h).reshaped(
                h.dim(0), h.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)
            let normedModelPLE = projNorm(modelPLE)

            // Combine: (model_proj + token_embed) * 2^{-0.5}
            let perLayerInputScale = pow(Float(2.0), -0.5)
            let combined = (normedModelPLE + reshapedTokenPLE) * perLayerInputScale

            perLayerInputs = (0 ..< config.numHiddenLayers).map { i in
                combined[.ellipsis, i, 0...]
            }
        } else {
            perLayerInputs = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // Extend cache array for shared layers (which get nil caches)
        var fullCache: [KVCache?]
        if let cache {
            fullCache = cache.map { Optional($0) }
            while fullCache.count < config.numHiddenLayers {
                fullCache.append(nil)
            }
        } else {
            fullCache = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // Build masks: one per attention type
        var maskByType = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
        for (i, layer) in layers.enumerated() {
            let lt = layer.layerType
            if maskByType[lt] == nil {
                if lt == "sliding_attention" {
                    maskByType[lt] = createAttentionMask(
                        h: h, cache: fullCache[i], windowSize: config.slidingWindow)
                } else {
                    maskByType[lt] = createAttentionMask(h: h, cache: fullCache[i])
                }
            }
        }

        // Forward through layers, tracking intermediate KV pairs for sharing
        var intermediates = [(kv: (MLXArray, MLXArray)?, positionOffset: Gemma4PositionOffset?)](
            repeating: (nil, nil), count: config.numHiddenLayers)

        for (idx, layer) in layers.enumerated() {
            let prevIdx = previousKvs[idx]
            let sharedKV = intermediates[prevIdx].kv
            let sharedPositionOffset = intermediates[prevIdx].positionOffset

            let mask = maskByType[layer.layerType]
            let (out, kvPair, positionOffset) = layer(
                h,
                mask: mask,
                cache: fullCache[idx],
                perLayerInput: perLayerInputs[idx],
                sharedKV: sharedKV,
                positionOffset: sharedPositionOffset
            )
            h = out
            intermediates[idx] = (kvPair, positionOffset)
        }

        return norm(h)
    }
}

// MARK: - Public Model

public class Gemma4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let config: Gemma4TextConfiguration
    fileprivate let model: Gemma4TextModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = Gemma4TextModelInner(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        out = tanh(out / config.finalLogitSoftcapping) * config.finalLogitSoftcapping
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (k, v) in weights {
            // Skip vision/audio/rotary weights
            if k.contains("self_attn.rotary_emb")
                || k.contains("input_max")
                || k.contains("input_min")
                || k.contains("output_max")
                || k.contains("output_min")
            {
                continue
            }
            sanitized[k] = v
        }
        return sanitized
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers

        var caches = [any KVCache]()
        for i in 0 ..< firstKvShared {
            if config.layerTypes[i] == "full_attention" {
                caches.append(StandardKVCache())
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0.selfAttn }
    }
}
