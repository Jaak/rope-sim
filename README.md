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

The implementation offers mesh-scaled Hooke spring, Kelvin-Voigt, and standard
linear solid rope models, mass-proportional air damping, and direct kinematic
dragging of the payload. The UI offers semi-implicit Euler, classical
fourth-order Runge-Kutta (RK4), second-order L-stable Rosenbrock ROS2 with a
linear-time block-tridiagonal solve, and fully converged backward Euler
integration.

The default scene represents a 12 m, 10.5 mm low-stretch rope weighing
0.90 kg (75 g/m), with an 80 kg payload and an effective axial rigidity of
30 kN.
