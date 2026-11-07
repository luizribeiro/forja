# Numerical behavior

Catalog libraries compile with Metal Fast math, with a separate Relaxed copy.
Scalar programs and fused epilogues compile Safe. Attention retains its current
math mode and reference bits; its existing `fast::exp` calls are intentional.

Math mode and default floating-point functions are separate Metal settings.
Selecting Safe alone does not select Precise functions. Arithmetic contraction
and explicit `fma` calls also affect reference bits. A reduction class describes
the required batch and chunk invariances under those choices.
