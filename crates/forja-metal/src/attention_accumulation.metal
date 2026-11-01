constant uint attention_bounded_sums [[function_constant(219)]];
// Limit each partial to 64 contributions to bound rounding with constant register storage.
constant constexpr uint ATTENTION_PARTIAL_PERIOD = FORJA_ATTENTION_PARTIAL_PERIOD;
