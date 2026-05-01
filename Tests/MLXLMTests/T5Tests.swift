// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

public class T5Tests: XCTestCase {

    /// Build a minimal T5Configuration via the JSONDecoder path so we exercise
    /// the same coding keys (snake_case) used at load time.
    private func makeTinyConfig(
        gated: Bool = false,
        tieEmbeddings: Bool = true
    ) throws -> T5Configuration {
        let json: [String: Any] = [
            "model_type": "t5",
            "vocab_size": 64,
            "d_model": 16,
            "d_kv": 4,
            "d_ff": 32,
            "num_layers": 2,
            "num_decoder_layers": 2,
            "num_heads": 4,
            "relative_attention_num_buckets": 8,
            "relative_attention_max_distance": 32,
            "layer_norm_epsilon": 1.0e-6,
            "feed_forward_proj": gated ? "gated-gelu" : "relu",
            "tie_word_embeddings": tieEmbeddings,
            "decoder_start_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 0,
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(T5Configuration.self, from: data)
    }

    // MARK: - Sanitize Tests

    /// HuggingFace T5 stores layers as `encoder.block.{i}.layer.{0,1}.{...}` and
    /// `decoder.block.{i}.layer.{0,1,2}.{...}`. Verify our sanitize remaps
    /// these onto the Swift module hierarchy.
    func testSanitizeKeyRemapping() throws {
        let config = try makeTinyConfig()
        let model = T5Model(config)

        // Construct a placeholder weights dict mirroring HF naming, with the
        // smallest possible array (shape doesn't matter for the key-remap test).
        let placeholder = MLXArray.zeros([1])
        let hfKeys: [String: MLXArray] = [
            // Shared embedding & encoder relative bias source layer
            "shared.weight": placeholder,
            "encoder.block.0.layer.0.SelfAttention.q.weight": placeholder,
            "encoder.block.0.layer.0.SelfAttention.k.weight": placeholder,
            "encoder.block.0.layer.0.SelfAttention.v.weight": placeholder,
            "encoder.block.0.layer.0.SelfAttention.o.weight": placeholder,
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": placeholder,
            "encoder.block.0.layer.0.layer_norm.weight": placeholder,
            "encoder.block.0.layer.1.DenseReluDense.wi.weight": placeholder,
            "encoder.block.0.layer.1.DenseReluDense.wo.weight": placeholder,
            "encoder.block.0.layer.1.layer_norm.weight": placeholder,
            "encoder.final_layer_norm.weight": placeholder,
            // Decoder layer 0 (self-attn carries the relative bias)
            "decoder.block.0.layer.0.SelfAttention.q.weight": placeholder,
            "decoder.block.0.layer.0.SelfAttention.k.weight": placeholder,
            "decoder.block.0.layer.0.SelfAttention.v.weight": placeholder,
            "decoder.block.0.layer.0.SelfAttention.o.weight": placeholder,
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": placeholder,
            "decoder.block.0.layer.0.layer_norm.weight": placeholder,
            "decoder.block.0.layer.1.EncDecAttention.q.weight": placeholder,
            "decoder.block.0.layer.1.EncDecAttention.k.weight": placeholder,
            "decoder.block.0.layer.1.EncDecAttention.v.weight": placeholder,
            "decoder.block.0.layer.1.EncDecAttention.o.weight": placeholder,
            "decoder.block.0.layer.1.layer_norm.weight": placeholder,
            "decoder.block.0.layer.2.DenseReluDense.wi.weight": placeholder,
            "decoder.block.0.layer.2.DenseReluDense.wo.weight": placeholder,
            "decoder.block.0.layer.2.layer_norm.weight": placeholder,
            "decoder.final_layer_norm.weight": placeholder,
            "lm_head.weight": placeholder,
        ]

        let sanitized = model.sanitize(weights: hfKeys)

        // Spot-check a representative key from each remap family.
        XCTAssertNotNil(sanitized["wte.weight"], "shared. should remap to wte.")
        XCTAssertNotNil(
            sanitized["encoder.layers.0.attention.query_proj.weight"],
            "encoder self-attention q.weight should remap to attention.query_proj.weight"
        )
        XCTAssertNotNil(
            sanitized["encoder.layers.0.attention.key_proj.weight"],
            "encoder self-attention k.weight should remap to attention.key_proj.weight"
        )
        XCTAssertNotNil(
            sanitized["encoder.layers.0.attention.out_proj.weight"],
            "encoder self-attention o.weight should remap to attention.out_proj.weight"
        )
        XCTAssertNotNil(
            sanitized["encoder.layers.0.dense.wi.weight"],
            "encoder DenseReluDense.wi should remap to dense.wi"
        )
        XCTAssertNotNil(
            sanitized["encoder.layers.0.ln1.weight"],
            "encoder layer.0 layer_norm should remap to ln1"
        )
        XCTAssertNotNil(
            sanitized["encoder.layers.0.ln2.weight"],
            "encoder layer.1 layer_norm should remap to ln2"
        )
        XCTAssertNotNil(sanitized["encoder.ln.weight"])

        // Encoder block-0 carries the relative attention bias for the encoder stack.
        XCTAssertNotNil(
            sanitized["encoder.relative_attention_bias.embeddings.weight"],
            "encoder block.0 relative_attention_bias should be hoisted to encoder.relative_attention_bias.embeddings"
        )

        // Decoder split: self-attn vs cross-attn vs dense.
        XCTAssertNotNil(sanitized["decoder.layers.0.self_attention.query_proj.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.cross_attention.query_proj.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.dense.wi.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.ln1.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.ln2.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.ln3.weight"])
        XCTAssertNotNil(sanitized["decoder.ln.weight"])
        // Decoder hoists its block-0 self-attn relative bias the same way.
        XCTAssertNotNil(
            sanitized["decoder.relative_attention_bias.embeddings.weight"]
        )

        // tie_word_embeddings = true should drop lm_head.weight from the loaded set.
        XCTAssertNil(
            sanitized["lm_head.weight"],
            "lm_head.weight should be dropped when tie_word_embeddings == true"
        )

        // The HF cross-attention also persists a (zero) relative_attention_bias on layer.1
        // for layer 0; we drop that explicitly to match Python reference.
        XCTAssertNil(
            sanitized["decoder.layers.0.cross_attention.relative_attention_bias.weight"]
        )
    }

    func testSanitizeKeepsLmHeadWhenNotTied() throws {
        let config = try makeTinyConfig(tieEmbeddings: false)
        let model = T5Model(config)

        let placeholder = MLXArray.zeros([1])
        let weights: [String: MLXArray] = [
            "shared.weight": placeholder,
            "lm_head.weight": placeholder,
        ]

        let sanitized = model.sanitize(weights: weights)
        XCTAssertNotNil(sanitized["wte.weight"])
        XCTAssertNotNil(
            sanitized["lm_head.weight"],
            "lm_head.weight should be retained when tie_word_embeddings == false"
        )
    }

    // MARK: - Forward / Generation Shape Tests

    /// `prepare(...)` should produce logits of shape (B, 1, vocabSize) and stash the
    /// encoder memory in `LMOutput.State.crossAttentionStates` so subsequent
    /// `callAsFunction(...)` calls can resolve cross-attention without re-encoding.
    func testPrepareReturnsLogitsAndCarriesEncoderMemory() throws {
        // Values must match `makeTinyConfig`.
        let expectedVocabSize = 64
        let expectedDModel = 16
        let expectedPromptLen = 6

        let config = try makeTinyConfig()
        let model = T5Model(config)

        let promptTokens = MLXArray(
            (0 ..< expectedPromptLen).map { Int32($0 % expectedVocabSize) }
        ).reshaped(1, expectedPromptLen)
        let input = LMInput(tokens: promptTokens)

        let cache = model.newCache(parameters: nil)
        let result = try model.prepare(input, cache: cache, windowSize: nil)

        switch result {
        case .logits(let lm):
            XCTAssertEqual(lm.logits.dim(0), 1)
            XCTAssertEqual(lm.logits.dim(2), expectedVocabSize)
            XCTAssertNotNil(lm.state?.crossAttentionStates)
            // Memory should be (B=1, S=expectedPromptLen, dModel)
            let memory = try XCTUnwrap(lm.state?.crossAttentionStates)
            XCTAssertEqual(memory.dim(0), 1)
            XCTAssertEqual(memory.dim(1), expectedPromptLen)
            XCTAssertEqual(memory.dim(2), expectedDModel)

            // A subsequent decoder step with a single token + the carried state should
            // produce a single-token logits tensor and consume the self-attention cache.
            let nextToken = LMInput.Text(tokens: MLXArray([Int32(2)]).reshaped(1, 1))
            let stepOut = model.callAsFunction(nextToken, cache: cache, state: lm.state)
            XCTAssertEqual(stepOut.logits.dim(0), 1)
            XCTAssertEqual(stepOut.logits.dim(1), 1)
            XCTAssertEqual(stepOut.logits.dim(2), expectedVocabSize)
            XCTAssertNotNil(stepOut.state?.crossAttentionStates)

        case .tokens:
            XCTFail("T5Model.prepare must return .logits, not .tokens")
        }
    }
}
