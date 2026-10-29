# Architecture and safety

Forja runs inference engines as WebAssembly components in a Rust host. An engine
owns tensor names, model dimensions, kernel programs, algorithm choices and graph
planning. A separate import-free chat component renders messages and parses output.
The host owns model-file grants, quotas, validated views, GPU allocation and Metal 4
execution. No host switch selects a model family or a chip-specific fast path.

A guest submits programs in a restricted kernel language, never Metal shader
source. The validator checks indexing and shapes against the buffers granted to
that operation. Buffer views are validated at creation, and checked 64-bit offset
and size arithmetic refuses overflow. The language cannot express an unchecked
access outside those capabilities.

Generic primitives are the small trusted boundary for operations the kernel
language cannot express. Each exposes constraints and catalog properties.
Scalar epilogues remain guest programs attached to those primitives. Profiles
choose compatible variants and position rules; [SmolLM2's registry](../engines/smollm2/src/variants.rs)
is a concrete example. Device-dependent choices belong there, not in the host.

Host quotas meter resources and work requested by a guest. An allocation or
command-buffer failure propagates as an error. A profile's precision and variant
choices also form part of its numerical and benchmark identity: comparisons must
preserve those settings and the actual model bytes.

Read the [engine guide](engines.md) to build a component and the
[serving guide](serving.md) for the operator contracts around weight files,
authentication and prefix-cache isolation.
