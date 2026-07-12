# Solver benchmarking spike

This throwaway executable compares two ways of advancing the same lumped-mass,
bilateral Hooke rope for one simulated second:

- RopeSim's fixed-step backward Euler implementation (dense finite-difference
  Jacobian and handwritten dense Gaussian elimination), at 240 steps/s.
- Diffsol's adaptive variable-order BDF integrator, with an analytic
  Jacobian-vector product, graph-colored sparse Jacobian, and faer's sparse LU.
  It is run both with scientific-ish tolerances and with looser tolerances that
  are more plausible for an interactive visual simulation.

Run it with:

```text
cargo run --release -p solver-spike
```

The executable reports median setup and integration times over five measured
runs after one warm-up. Setup is reported separately because Diffsol discovers
and colors the Jacobian sparsity while constructing the problem; a real app can
reuse that work until the rope topology changes.

This is a directional performance spike, not a rigorous accuracy study. The
methods do not use identical error controls: RopeSim's backward Euler has a
fixed 1/240 s step. Strict Diffsol uses `rtol = 1e-5` and component-wise absolute
tolerances of `1e-7 m` for position and `1e-6 m/s` for velocity; interactive
Diffsol uses `rtol = 1e-3`, `atol(position) = 1e-5 m`, and
`atol(velocity) = 1e-4 m/s`.

## Result (2026-07-12)

Windows x86-64, Rust 1.97.0, `--release`:

| Pieces | Solver | Setup | Integrate 1 s | Internal steps |
|---:|---|---:|---:|---:|
| 20 | current backward Euler | 0.9 us | 2.80 ms | 240 |
| 20 | Diffsol BDF, interactive | 122.5 us | 14.02 ms | 4,072 |
| 20 | Diffsol BDF, strict | 132.5 us | 13.84 ms | 4,080 |
| 40 | current backward Euler | 0.8 us | 13.72 ms | 240 |
| 40 | Diffsol BDF, interactive | 192.5 us | 39.85 ms | 7,602 |
| 40 | Diffsol BDF, strict | 223.7 us | 43.92 ms | 7,924 |
| 64 | current backward Euler | 1.2 us | 44.97 ms | 240 |
| 64 | Diffsol BDF, interactive | 418.7 us | 91.50 ms | 11,928 |
| 64 | Diffsol BDF, strict | 377.6 us | 92.99 ms | 12,562 |

Absolute timings will vary by machine. The direction is clear for this test:
Diffsol's sparse path scales better from 20 to 64 pieces, but adaptive BDF is
still about 2x slower at 64 pieces and much slower at smaller ropes. Relaxing
the tolerances barely changes the internal step count. The stiff, initially
unstretched rope generates fast axial modes which adaptive BDF tries to resolve;
fixed-step backward Euler instead damps them numerically. Diffsol also solves a
first-order position-and-velocity system (four scalars per node), while the
current position-eliminated backward Euler solve has two unknowns per dynamic
node.

### Recommendation from the spike

Do not replace the production integrator wholesale with Diffsol BDF yet. For
the immediate interactive simulator, the best next optimization is an analytic
block-tridiagonal Jacobian and solve for the existing position-form backward
Euler method. If a general sparse factorization is preferred over a specialized
chain solver, faer's sparse LU can replace the handwritten dense linear solve,
but it does not replace Newton globalization.

Keep Diffsol as a candidate when the project needs adaptive high-accuracy
integration, more complex viscoelastic state, or a DAE formulation for the
inextensible model. Before that adoption, benchmark a fixed-order/L-stable
configuration on drag-and-release workloads rather than assuming variable-order
BDF is a drop-in interactive replacement.
