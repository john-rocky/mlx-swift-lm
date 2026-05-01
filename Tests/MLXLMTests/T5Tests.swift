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

    // MARK: - Real-world Configuration Decoding

    /// Smoke-test that a real `config.json` (from `mlx-community/flan-t5-small-mlx-4bit`,
    /// FLAN-T5 v1.1 family) decodes through `T5Configuration` without
    /// throwing. FLAN-T5 v1.1 uses `gated-gelu` activation and untied
    /// embeddings, both of which exercise edge cases of our decoder path.
    func testRealFlanT5SmallConfigurationDecodes() throws {
        let realConfigJSON = """
            {
              "d_ff": 1024,
              "d_kv": 64,
              "d_model": 512,
              "decoder_start_token_id": 0,
              "eos_token_id": 1,
              "feed_forward_proj": "gated-gelu",
              "is_encoder_decoder": true,
              "layer_norm_epsilon": 1e-06,
              "model_type": "t5",
              "n_positions": 512,
              "num_decoder_layers": 8,
              "num_heads": 6,
              "num_layers": 8,
              "pad_token_id": 0,
              "relative_attention_max_distance": 128,
              "relative_attention_num_buckets": 32,
              "tie_word_embeddings": false,
              "use_cache": true,
              "vocab_size": 32128
            }
            """
        let data = try XCTUnwrap(realConfigJSON.data(using: .utf8))
        let config = try JSONDecoder().decode(T5Configuration.self, from: data)

        // Instantiation must succeed against the same config the loader will see.
        let model = T5Model(config)

        // For tie_word_embeddings == false we expect an lm_head to be wired up.
        // Sanitize a (non-quantized) lm_head weight and confirm it survives.
        let placeholder = MLXArray.zeros([1])
        let sanitized = model.sanitize(weights: ["lm_head.weight": placeholder])
        XCTAssertNotNil(sanitized["lm_head.weight"])
    }

    // MARK: - Real-weight Load + Generation Test (skipped when model not in HF cache)

    /// End-to-end integration test against `mlx-community/flan-t5-small-mlx-4bit`.
    /// Loads the actual quantized weights, runs `prepare(...)` followed by greedy
    /// decoding, and checks the first generated token id matches the Python
    /// reference (mlx-examples T5).
    ///
    /// Skipped automatically when the model is not present in the local
    /// HuggingFace cache, so this is safe to run in any environment.
    func testLoadFlanT5SmallAndGenerateMatchesPythonReference() throws {
        let homeDir = ProcessInfo.processInfo.environment["HOME"] ?? "/Users/majimadaisuke"
        let snapshotsDir = URL(fileURLWithPath:
            "\(homeDir)/.cache/huggingface/hub/models--mlx-community--flan-t5-small-mlx-4bit/snapshots")
        guard
            let snap = try? FileManager.default.contentsOfDirectory(
                at: snapshotsDir, includingPropertiesForKeys: nil
            ).first,
            FileManager.default.fileExists(atPath: snap.appendingPathComponent("config.json").path),
            FileManager.default.fileExists(
                atPath: snap.appendingPathComponent("model.safetensors").path)
        else {
            throw XCTSkip(
                "mlx-community/flan-t5-small-mlx-4bit not in local HF cache; skipping E2E test")
        }

        // Load T5 config.
        let configData = try Data(
            contentsOf: snap.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(T5Configuration.self, from: configData)

        // Decode the quantization block separately (BaseConfiguration handles this for the
        // factory path; here we re-parse the same JSON to keep the test self-contained).
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

        let model = T5Model(config)
        try loadWeights(
            modelDirectory: snap,
            model: model,
            quantization: baseConfig.quantization,
            perLayerQuantization: baseConfig.perLayerQuantization
        )

        // Reference input ids: tokenizer.encode("translate English to German: That is good.")
        // for the `t5-small` family.
        let refInputIds: [Int32] = [13959, 1566, 12, 2968, 10, 466, 19, 207, 5, 1]
        let promptTokens = MLXArray(refInputIds).reshaped(1, refInputIds.count)
        let input = LMInput(tokens: promptTokens)

        let cache = model.newCache(parameters: nil)
        let prepareResult = try model.prepare(input, cache: cache, windowSize: nil)

        guard case .logits(let firstStep) = prepareResult else {
            XCTFail("T5Model.prepare must return .logits")
            return
        }

        // Greedy first token must match the Python reference (716, captured from
        // mlx-examples T5 running the same checkpoint with the same input ids).
        let logits1D = firstStep.logits[0, -1, 0...]
        let firstToken = argMax(logits1D, axis: -1).item(Int32.self)
        XCTAssertEqual(
            Int(firstToken), 716,
            "First greedy token must match Python mlx-examples T5 reference"
        )

        // Encoder output spot-check: Python ref [0,0,:5] ≈ [-0.3616, -0.5402, 0.071, 1.3423, 0.0216].
        let memory = try XCTUnwrap(firstStep.state?.crossAttentionStates)
        let memChannel0 = memory[0, 0, 0 ..< 5].asArray(Float.self)
        let pyExpected: [Float] = [-0.3616, -0.5402, 0.0710, 1.3423, 0.0216]
        for i in 0 ..< 5 {
            XCTAssertEqual(
                memChannel0[i], pyExpected[i], accuracy: 0.05,
                "Encoder output channel \(i) drifts from Python reference"
            )
        }

        // Walk the decoder forward N steps, checking that state.crossAttentionStates is
        // threaded through each call so the decoded token sequence matches Python's
        // greedy output.
        var generated: [Int] = [Int(firstToken)]
        var lastState: LMOutput.State? = firstStep.state
        var lastToken = firstToken
        for _ in 0 ..< 4 {
            let nextInput = LMInput.Text(tokens: MLXArray([lastToken]).reshaped(1, 1))
            let stepOut = model.callAsFunction(nextInput, cache: cache, state: lastState)
            let nextToken = argMax(stepOut.logits[0, -1, 0...], axis: -1).item(Int32.self)
            generated.append(Int(nextToken))
            lastState = stepOut.state
            lastToken = nextToken
        }

        // Python reference (mlx-examples T5, greedy, same checkpoint+input):
        //   [716, 30153, 26202, 19935, 30120, ...]
        let pyTokens = [716, 30153, 26202, 19935, 30120]
        XCTAssertEqual(
            generated, pyTokens,
            "Multi-step greedy token sequence must match Python mlx-examples T5"
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
