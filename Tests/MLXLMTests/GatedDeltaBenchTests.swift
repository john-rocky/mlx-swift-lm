// Copyright © 2026 Apple Inc.
//
// Microbench comparing the fused metal `gatedDeltaKernel` against the
// `gatedDeltaOps` expression-graph fallback. Skipped by default; run via:
//
//   swift test --filter GatedDeltaBenchTests
//
// to print median latency over N iterations on the current device.

import MLX
import MLXLMCommon
import XCTest

final class GatedDeltaBenchTests: XCTestCase {

    struct Shape {
        let name: String
        let B: Int
        let T: Int
        let Hk: Int
        let Hv: Int
        let Dk: Int
        let Dv: Int
    }

    private func makeInputs(_ s: Shape, dtype: DType) -> (
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) {
        let q = MLXRandom.normal([s.B, s.T, s.Hk, s.Dk]).asType(dtype)
        let k = MLXRandom.normal([s.B, s.T, s.Hk, s.Dk]).asType(dtype)
        let v = MLXRandom.normal([s.B, s.T, s.Hv, s.Dv]).asType(dtype)
        let g = MLXRandom.uniform(low: 0.5, high: 1.0, [s.B, s.T, s.Hv]).asType(dtype)
        let beta = MLXRandom.uniform(low: 0.0, high: 1.0, [s.B, s.T, s.Hv]).asType(dtype)
        let state = MLXArray.zeros([s.B, s.Hv, s.Dv, s.Dk], dtype: dtype)
        return (q, k, v, g, beta, state)
    }

    private func median(_ xs: [Double]) -> Double {
        let sorted = xs.sorted()
        return sorted[sorted.count / 2]
    }

    func testGatedDeltaKernelVsOps() throws {
        // The microbench needs the metal kernel to compile; skip on platforms
        // where it isn't available.
        let probe = MLXArray.zeros([1, 1, 1, 1])
        let probeState = MLXArray.zeros([1, 1, 1, 1])
        eval(probe)
        _ = probe
        _ = probeState

        // Realistic per-token (T=1) shapes: Qwen 3.5 linear_attn config.
        // Hk=16, Hv=64, Dk=192, Dv=128 are the defaults from
        // Qwen35Configuration.TextConfiguration.
        let shapes: [Shape] = [
            Shape(name: "decode_T1", B: 1, T: 1, Hk: 16, Hv: 64, Dk: 192, Dv: 128),
            Shape(name: "decode_T8", B: 1, T: 8, Hk: 16, Hv: 64, Dk: 192, Dv: 128),
            Shape(name: "prefill_T64", B: 1, T: 64, Hk: 16, Hv: 64, Dk: 192, Dv: 128),
            Shape(name: "prefill_T256", B: 1, T: 256, Hk: 16, Hv: 64, Dk: 192, Dv: 128),
        ]

        let dtype: DType = .bfloat16
        let warmup = 3
        let iterations = 20

        print(
            "\n--- gatedDelta kernel vs ops bench (\(dtype), B=1, Hk=16, Hv=64, Dk=192, Dv=128) ---")
        print(
            "shape          | T   |  ops median (ms) | kernel median (ms) | speedup")
        print(
            "---------------|-----|------------------|--------------------|--------")

        for shape in shapes {
            let inputs = makeInputs(shape, dtype: dtype)

            // Warmup
            for _ in 0 ..< warmup {
                let (yK, sK) = gatedDeltaKernel(
                    q: inputs.q, k: inputs.k, v: inputs.v,
                    g: inputs.g, beta: inputs.beta, state: inputs.state)
                eval(yK, sK)

                let (yO, sO) = gatedDeltaOps(
                    q: inputs.q, k: inputs.k, v: inputs.v,
                    g: inputs.g, beta: inputs.beta, state: inputs.state)
                eval(yO, sO)
            }

            // Kernel timing
            var kernelTimes: [Double] = []
            for _ in 0 ..< iterations {
                let start = Date.timeIntervalSinceReferenceDate
                let (y, newState) = gatedDeltaKernel(
                    q: inputs.q, k: inputs.k, v: inputs.v,
                    g: inputs.g, beta: inputs.beta, state: inputs.state)
                eval(y, newState)
                kernelTimes.append((Date.timeIntervalSinceReferenceDate - start) * 1000)
            }

            // Ops timing
            var opsTimes: [Double] = []
            for _ in 0 ..< iterations {
                let start = Date.timeIntervalSinceReferenceDate
                let (y, newState) = gatedDeltaOps(
                    q: inputs.q, k: inputs.k, v: inputs.v,
                    g: inputs.g, beta: inputs.beta, state: inputs.state)
                eval(y, newState)
                opsTimes.append((Date.timeIntervalSinceReferenceDate - start) * 1000)
            }

            let opsMedian = median(opsTimes)
            let kernelMedian = median(kernelTimes)
            let speedup = opsMedian / kernelMedian
            let nameCol = shape.name.padding(toLength: 14, withPad: " ", startingAt: 0)
            let line = String(
                format: "%@ | %3d | %16.3f | %18.3f | %5.2fx",
                nameCol, shape.T, opsMedian, kernelMedian, speedup)
            print(line)
        }
        print("---")
    }
}
