use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaskDensity {
    Sparse,
    Dense,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Activation {
    Off,
    AutoNoTrigger,
    Trigger(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BenchmarkCase {
    name: &'static str,
    activation: Activation,
    density: MaskDensity,
    tools: usize,
    schema_bytes: usize,
}

fn cases() -> Vec<BenchmarkCase> {
    use Activation::{AutoNoTrigger, Off, Trigger};
    use MaskDensity::{Dense, Sparse};

    vec![
        BenchmarkCase {
            name: "no-tools",
            activation: Off,
            density: Dense,
            tools: 0,
            schema_bytes: 0,
        },
        BenchmarkCase {
            name: "auto-no-trigger",
            activation: AutoNoTrigger,
            density: Dense,
            tools: 1,
            schema_bytes: 1_024,
        },
        BenchmarkCase {
            name: "trigger-1-sparse",
            activation: Trigger(1),
            density: Sparse,
            tools: 1,
            schema_bytes: 1_024,
        },
        BenchmarkCase {
            name: "trigger-7-sparse",
            activation: Trigger(7),
            density: Sparse,
            tools: 1,
            schema_bytes: 1_024,
        },
        BenchmarkCase {
            name: "trigger-33-sparse",
            activation: Trigger(33),
            density: Sparse,
            tools: 1,
            schema_bytes: 1_024,
        },
        BenchmarkCase {
            name: "trigger-1-dense",
            activation: Trigger(1),
            density: Dense,
            tools: 1,
            schema_bytes: 1_024,
        },
        BenchmarkCase {
            name: "tools-16",
            activation: Trigger(1),
            density: Sparse,
            tools: 16,
            schema_bytes: 1_024,
        },
        BenchmarkCase {
            name: "tools-128",
            activation: Trigger(1),
            density: Sparse,
            tools: 128,
            schema_bytes: 1_024,
        },
        BenchmarkCase {
            name: "schema-64k",
            activation: Trigger(1),
            density: Sparse,
            tools: 1,
            schema_bytes: 64 * 1_024,
        },
        BenchmarkCase {
            name: "schema-256k",
            activation: Trigger(1),
            density: Sparse,
            tools: 1,
            schema_bytes: 256 * 1_024,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_serving_constraint_axes() {
        let cases = cases();
        assert_eq!(
            cases
                .iter()
                .map(|case| case.name)
                .collect::<BTreeSet<_>>()
                .len(),
            cases.len()
        );
        assert!(cases.iter().any(|case| case.activation == Activation::Off));
        assert!(
            cases
                .iter()
                .any(|case| case.activation == Activation::AutoNoTrigger)
        );
        for position in [1, 7, 33] {
            assert!(
                cases
                    .iter()
                    .any(|case| case.activation == Activation::Trigger(position))
            );
        }
        for density in [MaskDensity::Sparse, MaskDensity::Dense] {
            assert!(cases.iter().any(|case| case.density == density));
        }
        for tools in [1, 16, 128] {
            assert!(cases.iter().any(|case| case.tools == tools));
        }
        for schema_bytes in [1_024, 64 * 1_024, 256 * 1_024] {
            assert!(cases.iter().any(|case| case.schema_bytes == schema_bytes));
        }
    }
}
