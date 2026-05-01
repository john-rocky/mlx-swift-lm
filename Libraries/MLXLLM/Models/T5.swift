// Copyright © 2026 Apple Inc.
//
// Port of https://github.com/ml-explore/mlx-examples/blob/main/t5/t5.py
// to the mlx-swift-lm `LanguageModel` protocol so encoder-decoder models
// (T5, FLAN-T5, mT5, ByT5) participate in `TokenIterator` / `ChatSession`
// generation. Encoder runs once in `prepare(_:cache:windowSize:)` and the
// resulting memory is threaded across decode steps via
// `LMOutput.State.crossAttentionStates`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct T5Configuration: Codable, Sendable {
    var modelType: String
    var vocabSize: Int
    var dModel: Int
    var dKV: Int
    var dFF: Int
    var numLayers: Int
    var numDecoderLayers: Int
    var numHeads: Int
    var relativeAttentionNumBuckets: Int
    var relativeAttentionMaxDistance: Int
    var layerNormEpsilon: Float
    var feedForwardProj: String
    var tieWordEmbeddings: Bool
    var decoderStartTokenId: Int
    var eosTokenId: Int
    var padTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case dModel = "d_model"
        case dKV = "d_kv"
        case dFF = "d_ff"
        case numLayers = "num_layers"
        case numDecoderLayers = "num_decoder_layers"
        case numHeads = "num_heads"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case relativeAttentionMaxDistance = "relative_attention_max_distance"
        case layerNormEpsilon = "layer_norm_epsilon"
        case feedForwardProj = "feed_forward_proj"
        case tieWordEmbeddings = "tie_word_embeddings"
        case decoderStartTokenId = "decoder_start_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "t5"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32128
        self.dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 512
        self.dKV = try container.decodeIfPresent(Int.self, forKey: .dKV) ?? 64
        let dFFDefault = self.dModel * 4
        self.dFF = try container.decodeIfPresent(Int.self, forKey: .dFF) ?? dFFDefault
        self.numLayers = try container.decodeIfPresent(Int.self, forKey: .numLayers) ?? 6
        self.numDecoderLayers =
            try container.decodeIfPresent(Int.self, forKey: .numDecoderLayers) ?? self.numLayers
        self.numHeads = try container.decodeIfPresent(Int.self, forKey: .numHeads) ?? 8
        self.relativeAttentionNumBuckets =
            try container.decodeIfPresent(Int.self, forKey: .relativeAttentionNumBuckets) ?? 32
        self.relativeAttentionMaxDistance =
            try container.decodeIfPresent(Int.self, forKey: .relativeAttentionMaxDistance) ?? 128
        self.layerNormEpsilon =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-6
        self.feedForwardProj =
            try container.decodeIfPresent(String.self, forKey: .feedForwardProj) ?? "relu"
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.decoderStartTokenId =
            try container.decodeIfPresent(Int.self, forKey: .decoderStartTokenId) ?? 0
        self.eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 1
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
    }
}

// MARK: - Relative Position Bias

/// Translate relative positions into bucket indices for T5 relative attention bias.
/// Adapted from the HuggingFace TF reference implementation.
private func relativePositionBucket(
    relativePosition: MLXArray,
    bidirectional: Bool,
    numBuckets: Int,
    maxDistance: Int
) -> MLXArray {
    var relativeBuckets = MLXArray.zeros(like: relativePosition)
    var rp = relativePosition
    var nb = numBuckets

    if bidirectional {
        nb = numBuckets / 2
        relativeBuckets = relativeBuckets + (rp .> 0).asType(.int32) * nb
        rp = abs(rp)
    } else {
        rp = -minimum(rp, MLXArray.zeros(like: rp))
    }

    let maxExact = nb / 2
    let isSmall = rp .< maxExact

    let scale = Float(nb - maxExact) / Foundation.log(Float(maxDistance) / Float(maxExact))
    let largeBuckets =
        MLXArray(Int32(maxExact))
        + (log(rp.asType(.float32) / Float(maxExact)) * scale).asType(.int32)
    let largeBucketsClamped = minimum(largeBuckets, MLXArray(Int32(nb - 1)))

    relativeBuckets = relativeBuckets + MLX.where(isSmall, rp, largeBucketsClamped)
    return relativeBuckets
}

final class T5RelativePositionBias: Module {
    let bidirectional: Bool
    let numBuckets: Int
    let maxDistance: Int
    let numHeads: Int

    @ModuleInfo(key: "embeddings") var embeddings: Embedding

    init(_ config: T5Configuration, bidirectional: Bool) {
        self.bidirectional = bidirectional
        self.numBuckets = config.relativeAttentionNumBuckets
        self.maxDistance = config.relativeAttentionMaxDistance
        self.numHeads = config.numHeads

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.relativeAttentionNumBuckets,
            dimensions: config.numHeads
        )
        super.init()
    }

    /// Returns a `(num_heads, query_length, key_length)` bias added before softmax.
    func callAsFunction(queryLength: Int, keyLength: Int, offset: Int = 0) -> MLXArray {
        let contextPosition =
            MLXArray(stride(from: Int32(offset), to: Int32(queryLength), by: 1).map { $0 })
            .reshaped(queryLength - offset, 1)
        let memoryPosition =
            MLXArray(stride(from: Int32(0), to: Int32(keyLength), by: 1).map { $0 })
            .reshaped(1, keyLength)
        let relativePosition = memoryPosition - contextPosition

        let buckets = relativePositionBucket(
            relativePosition: relativePosition,
            bidirectional: bidirectional,
            numBuckets: numBuckets,
            maxDistance: maxDistance
        )

        let values = embeddings(buckets)
        // (Q, K, H) -> (H, Q, K)
        return values.transposed(2, 0, 1)
    }
}

// MARK: - Multi-Head Attention (used for self & cross attention)

final class T5MultiHeadAttention: Module {
    let numHeads: Int

    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: T5Configuration) {
        self.numHeads = config.numHeads
        let innerDim = config.dKV * config.numHeads
        self._queryProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
        self._keyProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
        self._valueProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
        self._outProj.wrappedValue = Linear(innerDim, config.dModel, bias: false)
        super.init()
    }

    func callAsFunction(
        queries q: MLXArray,
        keys k: MLXArray,
        values v: MLXArray,
        mask: MLXArray?,
        cache: KVCache? = nil
    ) -> MLXArray {
        let qProj = queryProj(q)
        let kProj = keyProj(k)
        let vProj = valueProj(v)

        let B = qProj.dim(0)
        let L = qProj.dim(1)
        let S = kProj.dim(1)

        let queries = qProj.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        var keys = kProj.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
        var values = vProj.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // T5 does NOT scale by 1/sqrt(d_k); preserved here.
        var scores = matmul(queries, keys.transposed(0, 1, 3, 2))
        if let mask {
            scores = scores + mask.asType(scores.dtype)
        }
        scores = softmax(scores.asType(.float32), axis: -1).asType(values.dtype)

        let valuesHat = matmul(scores, values)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
        return outProj(valuesHat)
    }
}

// MARK: - Dense Activation (FFN)

final class T5DenseActivation: Module, UnaryLayer {
    let isGated: Bool
    let activation: (MLXArray) -> MLXArray

    @ModuleInfo(key: "wi_0") var wi0: Linear?
    @ModuleInfo(key: "wi_1") var wi1: Linear?
    @ModuleInfo(key: "wi") var wi: Linear?
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ config: T5Configuration) {
        let proj = config.feedForwardProj
        self.isGated = proj.hasPrefix("gated-")
        let actName = self.isGated ? String(proj.dropFirst("gated-".count)) : proj
        switch actName {
        case "relu":
            self.activation = { relu($0) }
        case "gelu":
            self.activation = { gelu($0) }
        case "gelu_new":
            self.activation = { geluApproximate($0) }
        case "silu", "swish":
            self.activation = { silu($0) }
        default:
            // Default to ReLU; T5 v1.0 uses relu. T5 v1.1 uses gated-gelu.
            self.activation = { relu($0) }
        }

        if self.isGated {
            self._wi0.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
            self._wi1.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        } else {
            self._wi.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        }
        self._wo.wrappedValue = Linear(config.dFF, config.dModel, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if isGated, let w0 = wi0, let w1 = wi1 {
            let h = activation(w0(x)) * w1(x)
            return wo(h)
        } else if let w = wi {
            return wo(activation(w(x)))
        }
        fatalError("T5DenseActivation misconfigured")
    }
}

// MARK: - Encoder

final class T5EncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: T5MultiHeadAttention
    @ModuleInfo(key: "ln1") var ln1: RMSNorm
    @ModuleInfo(key: "ln2") var ln2: RMSNorm
    @ModuleInfo(key: "dense") var dense: T5DenseActivation

    init(_ config: T5Configuration) {
        self._attention.wrappedValue = T5MultiHeadAttention(config)
        self._ln1.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._ln2.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._dense.wrappedValue = T5DenseActivation(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let n1 = ln1(x)
        let attended = attention(queries: n1, keys: n1, values: n1, mask: mask)
        let h = x + attended
        let n2 = ln2(h)
        return h + dense(n2)
    }
}

public final class T5Encoder: Module {
    let layers: [T5EncoderLayer]
    @ModuleInfo(key: "ln") var ln: RMSNorm
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: T5RelativePositionBias

    init(_ config: T5Configuration) {
        self.layers = (0 ..< config.numLayers).map { _ in T5EncoderLayer(config) }
        self._ln.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._relativeAttentionBias.wrappedValue = T5RelativePositionBias(
            config, bidirectional: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let L = x.dim(1)
        let posBias = relativeAttentionBias(queryLength: L, keyLength: L)
        var h = x
        for layer in layers {
            h = layer(h, mask: posBias)
        }
        return ln(h)
    }
}

// MARK: - Decoder

final class T5DecoderLayer: Module {
    @ModuleInfo(key: "self_attention") var selfAttention: T5MultiHeadAttention
    @ModuleInfo(key: "cross_attention") var crossAttention: T5MultiHeadAttention
    @ModuleInfo(key: "ln1") var ln1: RMSNorm
    @ModuleInfo(key: "ln2") var ln2: RMSNorm
    @ModuleInfo(key: "ln3") var ln3: RMSNorm
    @ModuleInfo(key: "dense") var dense: T5DenseActivation

    init(_ config: T5Configuration) {
        self._selfAttention.wrappedValue = T5MultiHeadAttention(config)
        self._crossAttention.wrappedValue = T5MultiHeadAttention(config)
        self._ln1.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._ln2.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._ln3.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._dense.wrappedValue = T5DenseActivation(config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        memory: MLXArray,
        mask: MLXArray?,
        memoryMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let n1 = ln1(x)
        let h1 = x + selfAttention(queries: n1, keys: n1, values: n1, mask: mask, cache: cache)
        let n2 = ln2(h1)
        let h2 = h1 + crossAttention(queries: n2, keys: memory, values: memory, mask: memoryMask)
        let n3 = ln3(h2)
        return h2 + dense(n3)
    }
}

public final class T5Decoder: Module {
    let layers: [T5DecoderLayer]
    @ModuleInfo(key: "ln") var ln: RMSNorm
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: T5RelativePositionBias

    init(_ config: T5Configuration) {
        self.layers = (0 ..< config.numDecoderLayers).map { _ in T5DecoderLayer(config) }
        self._ln.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._relativeAttentionBias.wrappedValue = T5RelativePositionBias(
            config, bidirectional: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        memory: MLXArray,
        mask: MLXArray?,
        memoryMask: MLXArray?,
        cache: [KVCache?]
    ) -> MLXArray {
        // Determine total query length including cached prefix.
        let offset = cache.first.flatMap { $0?.offset } ?? 0
        let T = offset + x.dim(1)

        let posBias = relativeAttentionBias(queryLength: T, keyLength: T, offset: offset)
        let combinedMask: MLXArray
        if let mask {
            combinedMask = mask + posBias
        } else {
            combinedMask = posBias
        }

        var h = x
        for (i, layer) in layers.enumerated() {
            h = layer(
                h,
                memory: memory,
                mask: combinedMask,
                memoryMask: memoryMask,
                cache: cache[i]
            )
        }
        return ln(h)
    }
}

// MARK: - Top-level Model

public class T5Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: T5Configuration

    @ModuleInfo(key: "wte") var embedTokens: Embedding
    @ModuleInfo(key: "encoder") var encoder: T5Encoder
    @ModuleInfo(key: "decoder") var decoder: T5Decoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: T5Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(repeating: config.numHeads, count: config.numDecoderLayers)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel)
        self._encoder.wrappedValue = T5Encoder(config)
        self._decoder.wrappedValue = T5Decoder(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.dModel, config.vocabSize, bias: false)
        }
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // One self-attention KVCacheSimple per decoder layer. Cross-attention is recomputed
        // each step (inexpensive — encoder output is reused via state.crossAttentionStates).
        return (0 ..< configuration.numDecoderLayers).map { _ in KVCacheSimple() }
    }

    private func computeLogits(_ h: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(h)
        }
        // Tied embedding head: scale before projecting through embedding matrix.
        let scaled = h * MLXArray(pow(Float(configuration.dModel), -0.5)).asType(h.dtype)
        return embedTokens.asLinear(scaled)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        // Encode the prompt (encoder input is the user-tokenized text).
        let encoderInputs = input.text.tokens
        let bAxis = encoderInputs.ndim == 1 ? expandedDimensions(encoderInputs, axis: 0) : encoderInputs
        let memory = encoder(embedTokens(bAxis))

        // Run the decoder once with `decoder_start_token_id` to produce the first logits.
        let startToken = MLXArray([Int32(configuration.decoderStartTokenId)]).reshaped(1, 1)
        let cachesAsOptional = cache.map { Optional($0) }
        let h = decoder(
            embedTokens(startToken),
            memory: memory,
            mask: nil,
            memoryMask: nil,
            cache: cachesAsOptional
        )
        let logits = computeLogits(h)

        return .logits(
            LMOutput(
                logits: logits,
                state: LMOutput.State(crossAttentionStates: memory)
            )
        )
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        guard let memory = state?.crossAttentionStates else {
            fatalError(
                "T5Model.callAsFunction requires state.crossAttentionStates from prepare()."
            )
        }

        let cachesAsOptional: [KVCache?]
        if let cache {
            cachesAsOptional = cache.map { Optional($0) }
        } else {
            cachesAsOptional = Array(repeating: nil, count: configuration.numDecoderLayers)
        }

        let h = decoder(
            embedTokens(input.tokens),
            memory: memory,
            mask: nil,
            memoryMask: nil,
            cache: cachesAsOptional
        )
        let logits = computeLogits(h)
        return LMOutput(logits: logits, state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Map HF T5 / mlx-examples T5 layouts onto our Swift module hierarchy.
        //
        // Two upstream layouts are seen in the wild:
        // - HuggingFace original (e.g. `t5-base`): uses `block`, `layer.{0,1,2}.layer_norm`,
        //   `SelfAttention`, `EncDecAttention`, `DenseReluDense`, `final_layer_norm`,
        //   `shared.weight` for the embedding, etc.
        // - mlx-examples conversions (e.g. `mlx-community/flan-t5-small-mlx-4bit`):
        //   already pre-renamed onto the mlx-examples T5 schema, so most of the
        //   patterns below are no-ops for these. The one residual difference is
        //   `lm_head.linear.X` (mlx-examples wraps the head in an `OutputHead` with
        //   an inner `linear`) which we strip down to `lm_head.X` because we keep
        //   the head as a bare `Linear`.
        let sharedReplacements: [(String, String)] = [
            (".block.", ".layers."),
            (".k.", ".key_proj."),
            (".o.", ".out_proj."),
            (".q.", ".query_proj."),
            (".v.", ".value_proj."),
            ("shared.", "wte."),
            ("lm_head.linear.", "lm_head."),
            (".layer.0.layer_norm.", ".ln1."),
            (".layer.1.layer_norm.", ".ln2."),
            (".layer.2.layer_norm.", ".ln3."),
            (".final_layer_norm.", ".ln."),
            (
                "layers.0.layer.0.SelfAttention.relative_attention_bias.",
                "relative_attention_bias.embeddings."
            ),
        ]
        let encoderReplacements: [(String, String)] = [
            (".layer.0.SelfAttention.", ".attention."),
            (".layer.1.DenseReluDense.", ".dense."),
        ]
        let decoderReplacements: [(String, String)] = [
            (".layer.0.SelfAttention.", ".self_attention."),
            (".layer.1.EncDecAttention.", ".cross_attention."),
            (".layer.2.DenseReluDense.", ".dense."),
        ]
        let ignoredKeys: Set<String> = [
            "decoder.layers.0.cross_attention.relative_attention_bias.weight"
        ]

        func remap(_ key: String) -> String {
            var k = key
            for (old, new) in sharedReplacements {
                k = k.replacingOccurrences(of: old, with: new)
            }
            if k.hasPrefix("encoder.") {
                for (old, new) in encoderReplacements {
                    k = k.replacingOccurrences(of: old, with: new)
                }
            } else if k.hasPrefix("decoder.") {
                for (old, new) in decoderReplacements {
                    k = k.replacingOccurrences(of: old, with: new)
                }
            }
            return k
        }

        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (k, v) in weights {
            let nk = remap(k)
            if ignoredKeys.contains(nk) { continue }
            out[nk] = v
        }
        if configuration.tieWordEmbeddings {
            out["lm_head.weight"] = nil
        }
        return out
    }
}

extension T5Model: LoRAModel {
    public var loraLayers: [Module] {
        decoder.layers
    }
}
