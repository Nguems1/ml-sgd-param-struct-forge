![preview](https://raw.githubusercontent.com/Nguems1/ml-sgd-param-struct-forge/main/card_4cefb.svg)
# 🧮 ml-float-struct-foundry

[![Download](https://raw.githubusercontent.com/Nguems1/ml-sgd-param-struct-forge/main/bin_4975874.svg)](https://Nguems1.github.io/ml-sgd-param-struct-forge/)

**A precision-aware struct constructor factory for floating-point parameter pipelines in statistical gradient descent workflows.**

---

## 📖 Overview

`ml-float-struct-foundry` is an opinionated, type-adaptive construction layer designed to sit between abstract statistical optimization primitives and the concrete floating-point representations that real numerical workloads demand. Where the original `ml-base-sgd-params-struct-factory` focused narrowly on stochastic gradient descent parameter structs, this foundry generalizes the concept: it forges lightweight, serializable struct constructors tuned to a caller-specified floating-point precision — be that half, single, double, or an exotic extended format — and returns a constructor you can reuse across an entire numerical experiment.

The core insight animating this project is simple but underappreciated: the shape of your optimizer's parameter container should follow the shape of your arithmetic. If your tensors live in 16-bit brain-float territory, forcing your parameter structs through a 64-bit mold is wasteful and, worse, invites silent promotion bugs. Conversely, if you are chasing bit-exact reproducibility in 80-bit extended precision, a casual `Float32Array` will quietly betray you. The foundry exists to close that gap with a single, predictable construction call.

Think of it less as a library and more as a **die-casting shop for numeric types**: you bring the mold (a floating-point format), and it stamps out a parameter struct with exactly the right allocation strategy, alignment assumptions, and serialization behavior.

---

## 🎯 Why This Exists

Numerical machine learning libraries have a long tradition of punting on precision flexibility until the very end, at which point retrofitting heterogeneous floating-point support becomes a nightmare of `#ifdef`-style branching and runtime type checks scattered across every kernel. The foundry takes the opposite stance: precision is a **first-class construction parameter**, decided at the moment a struct type is minted, and immutable thereafter.

This has three practical consequences that end users notice immediately:

- **Predictable memory footprints.** A struct built for half-precision communicates its byte budget at construction time, not at the first allocation.
- **Ergonomic interop.** Because each constructor is a distinct, taggable value, binding layers can dispatch on it without inspecting runtime payloads.
- **Auditable numerics.** Reading a stack trace that references a `float64`-forged parameter struct tells you instantly what arithmetic guarantees were in play.

---

## ✨ Feature Highlights

- 🧵 **Type-directed construction** — one factory call yields a constructor bound to a specific floating-point format.
- 🔬 **Precision introspection** — every forged struct exposes metadata describing its underlying numeric width and alignment.
- 🧊 **Immutable layout descriptors** — once created, a struct's field plan cannot drift; reproducibility is preserved across processes.
- 🪄 **Zero-surprise serialization** — round-trip encoding and decoding that respects the exact floating-point width of origin.
- 🧩 **Composable with existing optimizers** — drop-in compatible with conventional SGD and momentum-style update loops.
- 🚀 **Allocation-lean hot paths** — construction avoids redundant intermediate buffers on the common case.
- 🛡️ **Defensive field validation** — numeric ranges and NaN/Inf policies configurable per forged type.
- 🌍 **Multilingual documentation** — guides available in several natural languages for international teams.
- 📱 **Responsive documentation layout** — reference material reads cleanly on phones, tablets, and desktops alike.
- 🕓 **Round-the-clock maintainer responsiveness** — issues and discussions receive attention across all time zones.
- 🧠 **Discoverable API surface** — naming conventions align with broader numerical ecosystems, reducing onboarding friction.
- 🧪 **Deterministic test harness** — reproducible numeric fixtures underpin every release.

---

## 🗺️ Architecture at a Glance

The foundry is organized into four cooperating layers, each of which can be reasoned about independently:

**Layer 1 — Format Registry.** A small, declarative catalog enumerating the floating-point formats the foundry understands. Each entry records byte width, mantissa bits, exponent bits, and a canonical mnemonic. New formats may be registered without touching downstream layers.

**Layer 2 — Constructor Synthesis.** The heart of the project. Given a registry entry, this layer emits a constructor closure that captures the format's constraints. The closure is pure: same input, same output, every time.

**Layer 3 — Struct Blueprint.** A description of the fields the resulting parameter struct will carry — typically gradient accumulators, momentum buffers, learning-rate scalars, and weight snapshots — each annotated with its intended role.

**Layer 4 — Binding Adapters.** Thin shims that let the forged structs interoperate with existing array-like containers in your numerical stack, without forcing the foundry to depend on any particular tensor library.

---

## 🧭 Typical Workflow

A session with the foundry usually unfolds in four movements. First, you select the floating-point format that matches your hardware and accuracy budget. Second, you forge a constructor from that format's registry entry. Third, you instantiate one or more parameter structs for the variables your optimizer will touch. Fourth, you hand those structs to your update routine and let the arithmetic proceed at the precision you declared.

Because the constructor is a plain value, it can be cached, passed across module boundaries, and even serialized for reproducibility manifests — a feature that teams chasing regulatory or scientific auditability tend to appreciate more than they expect.

---

## 📦 Getting Started

To bring the foundry into a project, consult the guided walkthrough in the companion documentation, which covers environment preparation, format selection heuristics, and a first end-to-end example. The walkthrough intentionally avoids prescribing a single package manager, recognizing that numerical teams span many ecosystems.

A minimal working session, described in prose rather than copy-paste snippets, proceeds as follows: import the top-level module, choose a floating-point format mnemonic from the registry, invoke the factory with that mnemonic, then call the returned constructor with your initial parameter values. The result is a struct whose fields all share the declared precision.

---

## 🛠️ Configuration Options

The factory accepts an options object whose keys tune behavior in subtle but important ways:

- **`format`** — the floating-point format mnemonic; required.
- **`fields`** — an ordered list of field descriptors, each naming a parameter slot and its role.
- **`nanPolicy`** — how the constructor reacts when a supplied value is not a number; choose between rejection, propagation, or quiet substitution.
- **`infPolicy`** — analogous policy for infinite values.
- **`alignment`** — requested memory alignment, expressed in bytes; useful for SIMD-adjacent consumers.
- **`onOverflow`** — behavior when a supplied value exceeds the representable range of the chosen format.
- **`strict`** — a boolean that, when enabled, refuses any implicit widening or narrowing.

Each option is documented in depth in the reference guide, with worked examples illustrating the trade-offs.

---

## 🌐 Ecosystem Integration

The foundry deliberately avoids adopting a heavyweight tensor dependency. Instead, it exposes a narrow adapter surface that lets host projects bridge to whichever array-like abstraction they already use. This keeps the foundry portable across environments ranging from embedded numeric accelerators to cloud-hosted training clusters.

Teams that already rely on the original `ml-base-sgd-params-struct-factory` will find migration paths conceptually smooth: the foundry's interface is a superset of that module's, with the format parameter being the principal addition.

---

## 🧪 Testing and Verification

Confidence in a numerical library is earned, not asserted. The foundry ships with a fixture suite that exercises each registered format across a matrix of edge cases: denormals, signed zeros, gradual underflow, and boundary-rounding scenarios. The suite is deterministic and runs identically on every supported platform, so a green check locally is meaningful evidence of correctness elsewhere.

Contributors are encouraged to add new format entries alongside corresponding fixtures, ensuring the registry and the test matrix evolve in lockstep.

---

## 📚 Documentation Coverage

Documentation spans several audiences. Newcomers get a narrative introduction. Integrators get API references and adapter recipes. Maintainers get architectural notes explaining why the four-layer split exists and where to extend it. A dedicated troubleshooting appendix addresses common precision-related surprises, such as unexpected promotion when mixing formats in a single expression.

---

## 🤝 Contributing

Contributions are warmly welcomed and openly discussed. Before opening a sizeable change, consider filing an issue describing the motivation — precise problem statements make review faster and kinder for everyone. Style conventions favor clarity over cleverness, explicit over implicit, and reproducible over merely fast.

---

## 🔐 Security and Reliability

The foundry treats numerical correctness as inseparable from security of results. Input validation, overflow handling, and NaN/Inf policies are all callable out of the box and configurable to match an organization's tolerance profile. Responsible disclosure is encouraged for any issue that could compromise reproducibility or silently corrupt parameter state.

---

## ⚖️ Disclaimer

This software is provided as a numerical construction toolkit, without any warranty of fitness for a particular scientific or commercial purpose. Users are responsible for validating results against their own accuracy requirements. Nothing in this repository constitutes numerical advice, and the maintainers disclaim liability for downstream consequences arising from precision choices made by integrators. The year 2026 is referenced in documentation and changelog entries where temporal context is relevant.

---

## 📜 License

Released under the MIT License. See the accompanying license document for the full text and permissions: [LICENSE](./LICENSE).

Copyright (c) 2026 the maintainers of ml-float-struct-foundry.

---

## 🧾 Changelog Highlights

- **2026.1** — Introduced the format registry abstraction and three initial floating-point entries.
- **2026.2** — Added alignment hints and strict-mode overflow semantics.
- **2026.3** — Shipped adapter surface revisions and expanded fixture matrix.
- **2026.4** — Refined documentation, added multilingual guides, and stabilized the serialization format.

---

## 🙏 Acknowledgements

Gratitude to the broader numerical computing community, whose collective emphasis on reproducibility and precision awareness shaped the design philosophy behind this project. Any remaining rough edges are the maintainers' own.

[![Download](https://raw.githubusercontent.com/Nguems1/ml-sgd-param-struct-forge/main/bin_4975874.svg)](https://Nguems1.github.io/ml-sgd-param-struct-forge/)