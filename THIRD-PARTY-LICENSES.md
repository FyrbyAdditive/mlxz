# Third-party licenses

mlxz is licensed under **AGPL-3.0** (see `LICENSE`). It builds on the open-source
dependencies below, each under its own permissive license (Apache-2.0 or MIT). Their
copyright notices and license terms are retained here as required. Full license texts ship
with each package in its source checkout (`LICENSE` / `LICENSE.txt`).

AGPL-3.0 is compatible with Apache-2.0 and MIT dependencies: a work under (A)GPLv3 may
incorporate Apache-2.0/MIT-licensed components.

## Swift package dependencies

| Dependency | License | Source |
| --- | --- | --- |
| mlx-swift | MIT | https://github.com/ml-explore/mlx-swift |
| mlx-swift-lm (local fork: mlx-swift-lm-mtp) | MIT | https://github.com/ml-explore/mlx-swift-lm |
| hummingbird | Apache-2.0 | https://github.com/hummingbird-project/hummingbird |
| swift-argument-parser | Apache-2.0 | https://github.com/apple/swift-argument-parser |
| swift-log | Apache-2.0 | https://github.com/apple/swift-log |
| swift-huggingface | Apache-2.0 | https://github.com/huggingface/swift-huggingface |
| swift-transformers | Apache-2.0 | https://github.com/huggingface/swift-transformers |
| swift-jinja | Apache-2.0 | https://github.com/huggingface/swift-jinja |
| EventSource | MIT | (transitive) |
| yyjson | MIT | (transitive, via swift-huggingface) |
| async-http-client | Apache-2.0 | https://github.com/swift-server/async-http-client |
| swift-nio (+ extras, http2, ssl, transport-services) | Apache-2.0 | https://github.com/apple/swift-nio |
| swift-crypto, swift-certificates, swift-asn1 | Apache-2.0 | https://github.com/apple/* |
| swift-collections, swift-algorithms, swift-numerics | Apache-2.0 | https://github.com/apple/* |
| swift-atomics, swift-system | Apache-2.0 | https://github.com/apple/* |
| swift-async-algorithms | Apache-2.0 | https://github.com/apple/swift-async-algorithms |
| swift-distributed-tracing, swift-service-context, swift-service-lifecycle | Apache-2.0 | https://github.com/apple/* |
| swift-metrics, swift-http-types, swift-http-structured-headers | Apache-2.0 | https://github.com/apple/* |
| swift-configuration | Apache-2.0 | (transitive) |
| swift-syntax | Apache-2.0 | https://github.com/swiftlang/swift-syntax |

## Vendored C/C++ inside mlx-swift (`Source/Cmlx`)

| Component | License |
| --- | --- |
| mlx (Apple) | MIT |
| mlx-c | MIT |
| nlohmann/json | MIT |
| fmt | MIT-style (permissive) |
| metal-cpp (Apple) | Apache-2.0 |

## Reference implementations (ported, not linked)

The DSpark speculative-decoding support (in mlxz and in the mlx-swift-lm-mtp fork) is a
Swift port informed by these MIT-licensed reference implementations; algorithm per the
paper "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
Generation" (DeepSeek-AI, arXiv:2606.19348):

| Reference | License | Source |
| --- | --- | --- |
| DeepSpec (DeepSeek-AI) — official DSpark training/eval code + checkpoints | MIT | https://github.com/deepseek-ai/DeepSpec |
| mlx-dspark (Abdur Rahim) — Python MLX inference port | MIT | https://github.com/ARahim3/mlx-dspark |

## Model weights (not bundled)

mlxz downloads model weights (e.g. Qwen3.6, DSpark drafters) from Hugging Face at runtime;
it does not bundle or redistribute them. Those weights carry their own separate license
(see the model's Hugging Face repository), which governs model use/redistribution
independently of this code license.
