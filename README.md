# RopeSim

RopeSim is a two-dimensional weighted-rope simulator. The project currently
contains a platform-independent Rust physics library and a native `eframe`
application.

## Run

```powershell
cargo run -p ropesim-app --release
```

## Test

```powershell
cargo test --workspace
```

## Interaction scaling benchmark

```powershell
cargo run -p ropesim-physics --example hybrid_benchmark --release
```

The benchmark reports mean, median, and p99 step times for hybrid SLS/QKV
dragging and free backward Euler at 64, 256, 512, and 1,024 links. It includes
axial-only and bending-enabled SLS paths, fully converged release-handoff time,
and bounded-correction fallback counts.

## Dynamic-rope calibration fixture

```powershell
cargo run -p ropesim-physics --example rope_calibration --release
```

The fixture compares every constitutive model with the published single-rope
measurements for a [Petzl VOLTA GUIDE 9 mm](https://www.petzl.com/INT/en/Sport/Ropes/VOLTA-GUIDE-9-mm).
It uses an idealized EN 892/UIAA
80 kg, factor-1.77 fall, the production backward-Euler path at 240 Hz, and no
environmental damping. The report includes static and maximum dynamic
elongation plus peak tension at both ends of the distributed rope.

## Experimental bending probe

```powershell
cargo run -p ropesim-physics --example bending_probe --release
```

This sensitivity probe compares axial-only behavior with several uncalibrated
bending-rigidity values during held end shortening at 20 and 64 links. It also
reports the ideal vertical first-arrest fixture, which should be nearly
insensitive to bending. Both bending controls default to zero.

The implementation offers mesh-scaled Hooke spring, Kelvin-Voigt, tension-only
quadratic Kelvin-Voigt, and standard linear solid rope models,
mass-proportional air damping, optional experimental bending elasticity and
viscosity, and hybrid XPBD/backward-Euler manipulation of the payload. The UI
offers semi-implicit Euler, classical fourth-order Runge-Kutta (RK4),
second-order L-stable Rosenbrock ROS2, TR-BDF2, and fully converged backward
Euler integration. Implicit bending uses a linear-time block-pentadiagonal
solve, with sparse LU retained as a safety fallback.

The default scene represents a 12 m Petzl VOLTA GUIDE 9 mm reference rope
weighing 0.648 kg (54 g/m), with an 80 kg payload. Its standard-linear-solid
preset is calibrated to the published 7.6% static elongation, 34% dynamic
elongation, and 8.5 kN impact force. Every rope model loads its own recommended
parameters when selected; Hooke and quadratic Kelvin-Voigt intentionally favor
lively illustrative behavior over material accuracy.
