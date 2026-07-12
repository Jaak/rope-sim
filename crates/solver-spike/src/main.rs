use std::error::Error;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use diffsol::{FaerSparseLU, FaerSparseMat, FaerVec, OdeBuilder, OdeSolverMethod, Vector};
use ropesim_physics::{IntegratorKind, Simulation, SimulationConfig};

const SIMULATED_SECONDS: f64 = 1.0;
const FIXED_DT: f64 = 1.0 / 240.0;
const WARMUP_RUNS: usize = 1;
const MEASURED_RUNS: usize = 5;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Debug)]
struct RopeOde {
    segment_count: usize,
    rest_length: f64,
    stiffness: f64,
    damping: f64,
    gravity_x: f64,
    gravity_y: f64,
    masses: Vec<f64>,
}

impl RopeOde {
    fn from_config(config: SimulationConfig) -> Self {
        let rest_length = config.rope_length / config.segment_count as f64;
        let element_mass = config.rope_mass / config.segment_count as f64;
        let mut masses = vec![element_mass; config.segment_count + 1];
        masses[0] = 0.5 * element_mass;
        masses[config.segment_count] = 0.5 * element_mass + config.payload_mass;

        Self {
            segment_count: config.segment_count,
            rest_length,
            stiffness: config.axial_rigidity / rest_length,
            damping: config.air_damping_rate,
            gravity_x: config.gravity.x,
            gravity_y: config.gravity.y,
            masses,
        }
    }

    fn state_len(&self) -> usize {
        4 * (self.segment_count + 1)
    }

    fn write_initial_state(&self, state: &mut FaerVec<f64>) {
        state.fill(0.0);
        for node in 0..=self.segment_count {
            state[pos_y(node)] = -(node as f64) * self.rest_length;
        }
    }

    fn rhs(&self, state: &FaerVec<f64>, out: &mut FaerVec<f64>) {
        out.fill(0.0);

        for node in 1..=self.segment_count {
            out[pos_x(node)] = state[vel_x(node)];
            out[pos_y(node)] = state[vel_y(node)];
            out[vel_x(node)] = self.gravity_x - self.damping * state[vel_x(node)];
            out[vel_y(node)] = self.gravity_y - self.damping * state[vel_y(node)];
        }

        for left in 0..self.segment_count {
            let right = left + 1;
            let dx = state[pos_x(right)] - state[pos_x(left)];
            let dy = state[pos_y(right)] - state[pos_y(left)];
            let length = dx.hypot(dy);
            if length <= f64::EPSILON {
                continue;
            }

            let scale = self.stiffness * (length - self.rest_length) / length;
            let force_x = scale * dx;
            let force_y = scale * dy;

            if left != 0 {
                out[vel_x(left)] += force_x / self.masses[left];
                out[vel_y(left)] += force_y / self.masses[left];
            }
            out[vel_x(right)] -= force_x / self.masses[right];
            out[vel_y(right)] -= force_y / self.masses[right];
        }
    }

    fn jacobian_vector(
        &self,
        state: &FaerVec<f64>,
        direction: &FaerVec<f64>,
        out: &mut FaerVec<f64>,
    ) {
        out.fill(0.0);

        for node in 1..=self.segment_count {
            out[pos_x(node)] = direction[vel_x(node)];
            out[pos_y(node)] = direction[vel_y(node)];
            out[vel_x(node)] = -self.damping * direction[vel_x(node)];
            out[vel_y(node)] = -self.damping * direction[vel_y(node)];
        }

        for left in 0..self.segment_count {
            let right = left + 1;
            let dx = state[pos_x(right)] - state[pos_x(left)];
            let dy = state[pos_y(right)] - state[pos_y(left)];
            let length = dx.hypot(dy);
            if length <= f64::EPSILON {
                continue;
            }

            let nx = dx / length;
            let ny = dy / length;
            let transverse = 1.0 - self.rest_length / length;
            let axial = self.rest_length / length;
            let h_xx = self.stiffness * (transverse + axial * nx * nx);
            let h_xy = self.stiffness * axial * nx * ny;
            let h_yy = self.stiffness * (transverse + axial * ny * ny);

            let delta_x = direction[pos_x(right)] - direction[pos_x(left)];
            let delta_y = direction[pos_y(right)] - direction[pos_y(left)];
            let force_x = h_xx * delta_x + h_xy * delta_y;
            let force_y = h_xy * delta_x + h_yy * delta_y;

            if left != 0 {
                out[vel_x(left)] += force_x / self.masses[left];
                out[vel_y(left)] += force_y / self.masses[left];
            }
            out[vel_x(right)] -= force_x / self.masses[right];
            out[vel_y(right)] -= force_y / self.masses[right];
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    setup: Duration,
    integrate: Duration,
    end_y: f64,
    internal_steps: usize,
}

#[derive(Clone, Copy)]
struct Tolerances {
    relative: f64,
    position_absolute: f64,
    velocity_absolute: f64,
}

const STRICT_TOLERANCES: Tolerances = Tolerances {
    relative: 1e-5,
    position_absolute: 1e-7,
    velocity_absolute: 1e-6,
};

const INTERACTIVE_TOLERANCES: Tolerances = Tolerances {
    relative: 1e-3,
    position_absolute: 1e-5,
    velocity_absolute: 1e-4,
};

fn pos_x(node: usize) -> usize {
    4 * node
}

fn pos_y(node: usize) -> usize {
    4 * node + 1
}

fn vel_x(node: usize) -> usize {
    4 * node + 2
}

fn vel_y(node: usize) -> usize {
    4 * node + 3
}

fn config(segment_count: usize) -> SimulationConfig {
    SimulationConfig {
        segment_count,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    }
}

fn run_hand_rolled(segment_count: usize) -> BenchResult<Timing> {
    let setup_start = Instant::now();
    let mut simulation = Simulation::new(config(segment_count))?;
    let setup = setup_start.elapsed();

    let integrate_start = Instant::now();
    let steps = (SIMULATED_SECONDS / FIXED_DT).round() as usize;
    for _ in 0..steps {
        black_box(simulation.step(FIXED_DT)?);
    }
    let integrate = integrate_start.elapsed();

    Ok(Timing {
        setup,
        integrate,
        end_y: black_box(simulation.payload_position().y),
        internal_steps: steps,
    })
}

fn run_diffsol_strict(segment_count: usize) -> BenchResult<Timing> {
    run_diffsol(segment_count, STRICT_TOLERANCES)
}

fn run_diffsol_interactive(segment_count: usize) -> BenchResult<Timing> {
    run_diffsol(segment_count, INTERACTIVE_TOLERANCES)
}

fn run_diffsol(segment_count: usize, tolerances: Tolerances) -> BenchResult<Timing> {
    let setup_start = Instant::now();
    let rope = Arc::new(RopeOde::from_config(config(segment_count)));
    let nstates = rope.state_len();

    let rhs_rope = Arc::clone(&rope);
    let jac_rope = Arc::clone(&rope);
    let init_rope = Arc::clone(&rope);
    let mut absolute_tolerances = vec![tolerances.position_absolute; nstates];
    for node in 0..=segment_count {
        absolute_tolerances[vel_x(node)] = tolerances.velocity_absolute;
        absolute_tolerances[vel_y(node)] = tolerances.velocity_absolute;
    }

    let problem = OdeBuilder::<FaerSparseMat<f64>>::new()
        .rtol(tolerances.relative)
        .atol(absolute_tolerances)
        .h0(FIXED_DT)
        .rhs_implicit(
            move |state, _parameters, _time, out| rhs_rope.rhs(state, out),
            move |state, _parameters, _time, direction, out| {
                jac_rope.jacobian_vector(state, direction, out);
            },
        )
        .init(
            move |_parameters, _time, state| init_rope.write_initial_state(state),
            nstates,
        )
        .build()?;
    let mut solver = problem.bdf::<FaerSparseLU<f64>>()?;
    let setup = setup_start.elapsed();

    let integrate_start = Instant::now();
    let mut internal_steps = 0;
    while solver.state().t < SIMULATED_SECONDS {
        solver.step()?;
        internal_steps += 1;
    }
    let end_state = solver.interpolate(SIMULATED_SECONDS)?;
    let integrate = integrate_start.elapsed();

    Ok(Timing {
        setup,
        integrate,
        end_y: black_box(end_state[pos_y(segment_count)]),
        internal_steps,
    })
}

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn median_us(samples: impl Iterator<Item = Duration>) -> f64 {
    let mut samples: Vec<_> = samples.collect();
    median_duration(&mut samples).as_secs_f64() * 1e6
}

fn benchmark(
    segment_count: usize,
    runner: fn(usize) -> BenchResult<Timing>,
) -> BenchResult<Timing> {
    for _ in 0..WARMUP_RUNS {
        black_box(runner(segment_count)?);
    }

    let mut samples = Vec::with_capacity(MEASURED_RUNS);
    for _ in 0..MEASURED_RUNS {
        samples.push(runner(segment_count)?);
    }

    let setup = Duration::from_secs_f64(median_us(samples.iter().map(|sample| sample.setup)) / 1e6);
    let integrate =
        Duration::from_secs_f64(median_us(samples.iter().map(|sample| sample.integrate)) / 1e6);
    let representative = samples[MEASURED_RUNS / 2];
    Ok(Timing {
        setup,
        integrate,
        end_y: representative.end_y,
        internal_steps: representative.internal_steps,
    })
}

fn main() -> BenchResult<()> {
    println!("Rope solver spike: 1 simulated second, median of {MEASURED_RUNS} release runs");
    println!("CPU: {}", std::env::consts::ARCH);
    println!();
    println!(
        "{:<8} {:<24} {:>12} {:>14} {:>9} {:>12}",
        "pieces", "solver", "setup (us)", "integrate (ms)", "steps", "payload y"
    );

    for segment_count in [20, 40, 64] {
        let hand_rolled = benchmark(segment_count, run_hand_rolled)?;
        let diffsol_interactive = benchmark(segment_count, run_diffsol_interactive)?;
        let diffsol_strict = benchmark(segment_count, run_diffsol_strict)?;

        print_timing(segment_count, "current backward Euler", hand_rolled);
        print_timing(
            segment_count,
            "Diffsol BDF (interactive)",
            diffsol_interactive,
        );
        print_timing(segment_count, "Diffsol BDF (strict)", diffsol_strict);
    }

    Ok(())
}

fn print_timing(segment_count: usize, name: &str, timing: Timing) {
    println!(
        "{segment_count:<8} {name:<24} {:>12.1} {:>14.3} {:>9} {:>12.6}",
        timing.setup.as_secs_f64() * 1e6,
        timing.integrate.as_secs_f64() * 1e3,
        timing.internal_steps,
        timing.end_y,
    );
}
