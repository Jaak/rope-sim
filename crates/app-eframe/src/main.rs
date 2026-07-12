use std::time::Instant;

use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2 as EguiVec2,
};
use ropesim_physics::{
    Diagnostics, IntegratorKind, KinematicTarget, ReconfigureOutcome, RopeModelKind, Simulation,
    SimulationConfig, Vec2,
};

const DEFAULT_FIXED_DT: f64 = 1.0 / 240.0;
const MAX_FRAME_DT: f64 = 0.1;
const MAX_STEPS_PER_FRAME: usize = 32;
const PAYLOAD_RADIUS: f32 = 13.0;
const GRAB_RADIUS: f32 = 24.0;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([760.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RopeSim",
        options,
        Box::new(|creation_context| Ok(Box::new(RopeSimApp::new(creation_context)))),
    )
}

struct RopeSimApp {
    simulation: Simulation,
    config: SimulationConfig,
    diagnostics: Diagnostics,
    paused: bool,
    single_step_requested: bool,
    accumulator: f64,
    time_scale: f64,
    integration_substeps: usize,
    last_frame: Instant,
    dragging_payload: bool,
    previous_drag_position: Option<Vec2>,
    drag_velocity: Vec2,
    error_message: Option<String>,
}

impl RopeSimApp {
    fn new(creation_context: &eframe::CreationContext<'_>) -> Self {
        creation_context.egui_ctx.set_visuals(egui::Visuals::dark());
        let config = SimulationConfig::default();
        let simulation = Simulation::new(config).expect("default configuration must be valid");
        let diagnostics = simulation.diagnostics();
        let integration_substeps = simulation
            .recommended_substeps(DEFAULT_FIXED_DT)
            .expect("default time step must be valid");

        Self {
            simulation,
            config,
            diagnostics,
            paused: false,
            single_step_requested: false,
            accumulator: 0.0,
            time_scale: 1.0,
            integration_substeps,
            last_frame: Instant::now(),
            dragging_payload: false,
            previous_drag_position: None,
            drag_velocity: Vec2::ZERO,
            error_message: None,
        }
    }

    fn controls(&mut self, root_ui: &mut egui::Ui) {
        let previous_config = self.config;
        let mut reset_requested = false;

        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(290.0)
            .show(root_ui, |ui| {
                ui.heading("RopeSim");
                ui.label(format!(
                    "{} · {}",
                    self.config.rope_model.display_name(),
                    self.config.integrator.display_name()
                ));
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    let pause_label = if self.paused { "Resume" } else { "Pause" };
                    if ui.button(pause_label).clicked() {
                        self.paused = !self.paused;
                        self.accumulator = 0.0;
                    }
                    if ui
                        .add_enabled(self.paused, egui::Button::new("Single step"))
                        .clicked()
                    {
                        self.single_step_requested = true;
                    }
                    if ui.button("Reset").clicked() {
                        reset_requested = true;
                    }
                });

                ui.separator();
                ui.heading("Rope");
                ui.add(egui::Slider::new(&mut self.config.segment_count, 1..=64).text("Pieces"));
                ui.add(
                    egui::Slider::new(&mut self.config.rope_length, 2.0..=30.0).text("Length (m)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.rope_mass, 0.2..=10.0)
                        .logarithmic(true)
                        .text("Rope mass (kg)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.payload_mass, 10.0..=200.0)
                        .logarithmic(true)
                        .text("Payload (kg)"),
                );

                ui.separator();
                ui.heading("Material and environment");
                egui::ComboBox::from_id_salt("rope_model")
                    .selected_text(self.config.rope_model.display_name())
                    .show_ui(ui, |ui| {
                        for model in RopeModelKind::ALL {
                            ui.selectable_value(
                                &mut self.config.rope_model,
                                model,
                                model.display_name(),
                            );
                        }
                    });
                let rigidity_label = if self.config.rope_model == RopeModelKind::StandardLinearSolid
                {
                    "Relaxed rigidity EA_inf (N)"
                } else {
                    "Axial rigidity EA (N)"
                };
                ui.add(
                    egui::Slider::new(&mut self.config.axial_rigidity, 1_000.0..=100_000.0)
                        .logarithmic(true)
                        .text(rigidity_label),
                );
                if self.config.rope_model == RopeModelKind::StandardLinearSolid {
                    ui.add(
                        egui::Slider::new(
                            &mut self.config.transient_axial_rigidity,
                            100.0..=100_000.0,
                        )
                        .logarithmic(true)
                        .text("Transient rigidity EA_1 (N)"),
                    );
                }
                if matches!(
                    self.config.rope_model,
                    RopeModelKind::KelvinVoigt | RopeModelKind::StandardLinearSolid
                ) {
                    ui.add(
                        egui::Slider::new(&mut self.config.axial_viscosity, 0.01..=1_000_000.0)
                            .logarithmic(true)
                            .text("Axial viscosity eta*A (N*s)"),
                    );
                }
                if self.config.rope_model == RopeModelKind::StandardLinearSolid {
                    ui.small(format!(
                        "Relaxation time: {:.3} s",
                        self.config.axial_viscosity / self.config.transient_axial_rigidity
                    ));
                }
                ui.add(
                    egui::Slider::new(&mut self.config.air_damping_rate, 0.0..=5.0)
                        .text("Air damping (1/s)"),
                );

                let mut gravity_magnitude = -self.config.gravity.y;
                if ui
                    .add(
                        egui::Slider::new(&mut gravity_magnitude, 0.0..=20.0)
                            .text("Gravity (m/s²)"),
                    )
                    .changed()
                {
                    self.config.gravity.y = -gravity_magnitude;
                }

                ui.separator();
                ui.heading("Numerics");
                egui::ComboBox::from_id_salt("integrator")
                    .selected_text(self.config.integrator.display_name())
                    .show_ui(ui, |ui| {
                        for integrator in IntegratorKind::ALL {
                            ui.selectable_value(
                                &mut self.config.integrator,
                                integrator,
                                integrator.display_name(),
                            );
                        }
                    });

                ui.separator();
                ui.heading("Playback");
                ui.add(egui::Slider::new(&mut self.time_scale, 0.1..=2.0).text("Time scale"));
                ui.label(format!("Fixed step: {:.3} ms", DEFAULT_FIXED_DT * 1000.0));
                ui.label(format!(
                    "Automatic substeps: {} (effective {:.3} ms)",
                    self.integration_substeps,
                    DEFAULT_FIXED_DT * 1000.0 / self.integration_substeps as f64
                ));

                ui.separator();
                ui.heading("Diagnostics");
                egui::Grid::new("diagnostics_grid")
                    .num_columns(2)
                    .spacing([12.0, 3.0])
                    .show(ui, |ui| {
                        diagnostic_row(ui, "Time", self.diagnostics.simulation_time, "s");
                        diagnostic_row(ui, "Kinetic", self.diagnostics.kinetic_energy, "J");
                        diagnostic_row(ui, "Elastic", self.diagnostics.elastic_energy, "J");
                        diagnostic_row(ui, "Gravity", self.diagnostics.gravitational_energy, "J");
                        diagnostic_row(ui, "Total", self.diagnostics.total_mechanical_energy, "J");
                        diagnostic_row(
                            ui,
                            "Max tension strain",
                            100.0 * self.diagnostics.maximum_tensile_strain,
                            "%",
                        );
                        diagnostic_row(
                            ui,
                            "Min segment",
                            self.diagnostics.minimum_segment_length,
                            "m",
                        );
                        diagnostic_row(ui, "Max speed", self.diagnostics.maximum_node_speed, "m/s");
                        diagnostic_row(
                            ui,
                            "Endpoint power",
                            self.diagnostics.prescribed_endpoint_power,
                            "W",
                        );
                        diagnostic_row(
                            ui,
                            "Endpoint work",
                            self.diagnostics.cumulative_prescribed_work,
                            "J",
                        );
                        diagnostic_row(
                            ui,
                            "Rejected steps",
                            self.diagnostics.rejected_steps as f64,
                            "",
                        );
                        diagnostic_row(
                            ui,
                            "Linear solves",
                            self.diagnostics.linear_solves as f64,
                            "",
                        );
                        diagnostic_row(
                            ui,
                            "Newton iterations",
                            self.diagnostics.nonlinear_iterations as f64,
                            "",
                        );
                        diagnostic_row(
                            ui,
                            "Adaptive retries",
                            self.diagnostics.adaptive_retries as f64,
                            "",
                        );
                        diagnostic_row(
                            ui,
                            "Timestep",
                            self.diagnostics.explicit_stable_timestep * 1000.0,
                            "ms",
                        );
                    });

                ui.add_space(8.0);
                ui.small("Drag the payload directly. Its mouse velocity is retained on release.");

                if let Some(message) = &self.error_message {
                    ui.add_space(8.0);
                    ui.colored_label(Color32::LIGHT_RED, message);
                }
            });

        if reset_requested {
            self.reset_simulation();
        } else if self.config != previous_config {
            match self.simulation.reconfigure(self.config) {
                Ok(ReconfigureOutcome::Updated) => {
                    self.diagnostics = self.simulation.diagnostics();
                    self.integration_substeps = self
                        .simulation
                        .recommended_substeps(DEFAULT_FIXED_DT)
                        .expect("fixed time step and validated configuration must be valid");
                    self.error_message = None;
                }
                Ok(ReconfigureOutcome::Reset) => self.after_simulation_reset(),
                Err(error) => {
                    self.config = previous_config;
                    self.error_message = Some(error.to_string());
                }
            }
        }
    }

    fn viewport(&mut self, root_ui: &mut egui::Ui, frame_dt: f64) {
        egui::CentralPanel::default().show(root_ui, |ui| {
            let size = ui.available_size().max(EguiVec2::new(1.0, 1.0));
            let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
            let transform = ViewTransform::new(response.rect, self.config);

            self.update_drag(&response, transform, frame_dt);
            self.advance_simulation(frame_dt);
            self.paint_scene(&painter, response.rect, transform);
        });
    }

    fn update_drag(&mut self, response: &egui::Response, transform: ViewTransform, frame_dt: f64) {
        let pointer_position = response.interact_pointer_pos();
        if response.drag_started()
            && pointer_position.is_some_and(|pointer| {
                pointer.distance(transform.world_to_screen(self.simulation.payload_position()))
                    <= GRAB_RADIUS
            })
        {
            self.dragging_payload = true;
            self.previous_drag_position = None;
            self.drag_velocity = Vec2::ZERO;
        }

        let primary_down = response.ctx.input(|input| input.pointer.primary_down());
        if self.dragging_payload && primary_down {
            if let Some(pointer) = pointer_position {
                let world_position = transform.screen_to_world(pointer);
                let measured_velocity = self
                    .previous_drag_position
                    .map(|previous| (world_position - previous) / frame_dt.max(1.0e-6))
                    .unwrap_or(Vec2::ZERO);

                // Mild smoothing removes frame-time noise without changing the
                // prescribed position of the payload.
                self.drag_velocity = self.drag_velocity * 0.55 + measured_velocity * 0.45;
                self.previous_drag_position = Some(world_position);
                let target = KinematicTarget::new(world_position, self.drag_velocity);
                if self.paused {
                    self.simulation.set_payload_target(Some(target));
                } else {
                    let base_duration =
                        (frame_dt.min(MAX_FRAME_DT) * self.time_scale).max(DEFAULT_FIXED_DT);
                    if let Err(error) = self
                        .simulation
                        .interpolate_payload_target(target, base_duration)
                    {
                        self.error_message = Some(error.to_string());
                    }
                }
            }
        } else if self.dragging_payload {
            self.dragging_payload = false;
            self.previous_drag_position = None;
            let release_velocity = self.simulation.payload_velocity();
            self.simulation.release_payload(release_velocity);
        }
    }

    fn advance_simulation(&mut self, frame_dt: f64) {
        if self.paused && !self.single_step_requested {
            self.diagnostics = self.simulation.diagnostics();
            return;
        }

        if self.single_step_requested {
            self.accumulator = DEFAULT_FIXED_DT;
            self.single_step_requested = false;
        } else {
            self.accumulator += frame_dt.min(MAX_FRAME_DT) * self.time_scale;
        }

        let mut steps = 0;
        while self.accumulator >= DEFAULT_FIXED_DT && steps < MAX_STEPS_PER_FRAME {
            self.integration_substeps = match self.simulation.recommended_substeps(DEFAULT_FIXED_DT)
            {
                Ok(substeps) => substeps,
                Err(error) => {
                    self.recover_from_step_error(error.to_string());
                    return;
                }
            };
            let substep_dt = DEFAULT_FIXED_DT / self.integration_substeps as f64;

            let mut failed = None;
            for _ in 0..self.integration_substeps {
                match self.simulation.step(substep_dt) {
                    Ok(diagnostics) => self.diagnostics = diagnostics,
                    Err(error) => {
                        failed = Some(error.to_string());
                        break;
                    }
                }
            }
            if let Some(error) = failed {
                self.recover_from_step_error(error);
                return;
            }

            self.error_message = None;
            self.accumulator -= DEFAULT_FIXED_DT;
            steps += 1;
        }

        if steps == MAX_STEPS_PER_FRAME {
            self.accumulator = self.accumulator.min(DEFAULT_FIXED_DT);
        }
    }

    fn paint_scene(&self, painter: &egui::Painter, rect: Rect, transform: ViewTransform) {
        painter.rect_filled(rect, 0.0, Color32::from_rgb(18, 22, 29));

        let anchor = transform.world_to_screen(self.config.anchor);
        paint_dotted_circle(
            painter,
            anchor,
            self.config.rope_length as f32 * transform.pixels_per_metre,
            Color32::from_rgb(72, 91, 108),
        );
        let floor_y = transform
            .world_to_screen(Vec2::new(0.0, -self.config.rope_length * 1.4))
            .y;
        painter.line_segment(
            [
                Pos2::new(rect.left(), floor_y),
                Pos2::new(rect.right(), floor_y),
            ],
            Stroke::new(1.0, Color32::from_gray(48)),
        );

        let positions: Vec<Pos2> = self
            .simulation
            .positions()
            .iter()
            .copied()
            .map(|position| transform.world_to_screen(position))
            .collect();

        painter.add(Shape::line(
            positions.clone(),
            Stroke::new(3.0, Color32::from_rgb(205, 181, 132)),
        ));

        if self.config.segment_count <= 32 {
            for position in positions
                .iter()
                .skip(1)
                .take(positions.len().saturating_sub(2))
            {
                painter.circle_filled(*position, 2.5, Color32::from_rgb(118, 153, 183));
            }
        }

        painter.circle_filled(anchor, 7.0, Color32::from_rgb(235, 235, 235));
        painter.line_segment(
            [
                anchor + EguiVec2::new(-16.0, -10.0),
                anchor + EguiVec2::new(16.0, -10.0),
            ],
            Stroke::new(4.0, Color32::from_gray(150)),
        );

        let payload = positions[positions.len() - 1];
        let payload_color = if self.dragging_payload {
            Color32::from_rgb(246, 186, 73)
        } else {
            Color32::from_rgb(215, 104, 91)
        };
        painter.circle_filled(payload, PAYLOAD_RADIUS, payload_color);
        painter.circle_stroke(payload, PAYLOAD_RADIUS, Stroke::new(2.0, Color32::WHITE));

        painter.text(
            rect.left_top() + EguiVec2::new(12.0, 10.0),
            Align2::LEFT_TOP,
            if self.paused { "PAUSED" } else { "RUNNING" },
            FontId::monospace(13.0),
            if self.paused {
                Color32::from_rgb(246, 186, 73)
            } else {
                Color32::from_rgb(120, 205, 155)
            },
        );
    }

    fn reset_simulation(&mut self) {
        self.simulation =
            Simulation::new(self.config).expect("slider-constrained configuration must be valid");
        self.after_simulation_reset();
    }

    fn after_simulation_reset(&mut self) {
        self.diagnostics = self.simulation.diagnostics();
        self.integration_substeps = self
            .simulation
            .recommended_substeps(DEFAULT_FIXED_DT)
            .expect("fixed time step and validated configuration must be valid");
        self.accumulator = 0.0;
        self.dragging_payload = false;
        self.previous_drag_position = None;
        self.drag_velocity = Vec2::ZERO;
        self.error_message = None;
    }

    fn recover_from_step_error(&mut self, error: String) {
        self.simulation =
            Simulation::new(self.config).expect("slider-constrained configuration must be valid");
        self.after_simulation_reset();
        self.paused = true;
        self.error_message = Some(format!(
            "The simulation became unstable and was reset safely: {error}"
        ));
    }
}

impl eframe::App for RopeSimApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let frame_dt = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;

        self.controls(ui);
        self.viewport(ui, frame_dt);
        ui.ctx().request_repaint();
    }
}

#[derive(Clone, Copy)]
struct ViewTransform {
    anchor_screen: Pos2,
    anchor_world: Vec2,
    pixels_per_metre: f32,
}

impl ViewTransform {
    fn new(rect: Rect, config: SimulationConfig) -> Self {
        let horizontal_scale = rect.width() / (2.8 * config.rope_length as f32);
        let vertical_scale = rect.height() / (1.65 * config.rope_length as f32);
        let pixels_per_metre = horizontal_scale.min(vertical_scale).max(1.0);

        Self {
            anchor_screen: Pos2::new(rect.center().x, rect.top() + 55.0),
            anchor_world: config.anchor,
            pixels_per_metre,
        }
    }

    fn world_to_screen(self, world: Vec2) -> Pos2 {
        Pos2::new(
            self.anchor_screen.x + ((world.x - self.anchor_world.x) as f32 * self.pixels_per_metre),
            self.anchor_screen.y - ((world.y - self.anchor_world.y) as f32 * self.pixels_per_metre),
        )
    }

    fn screen_to_world(self, screen: Pos2) -> Vec2 {
        Vec2::new(
            self.anchor_world.x
                + ((screen.x - self.anchor_screen.x) / self.pixels_per_metre) as f64,
            self.anchor_world.y
                - ((screen.y - self.anchor_screen.y) / self.pixels_per_metre) as f64,
        )
    }
}

fn diagnostic_row(ui: &mut egui::Ui, name: &str, value: f64, unit: &str) {
    ui.label(name);
    ui.monospace(format_diagnostic(value, unit));
    ui.end_row();
}

fn format_diagnostic(value: f64, unit: &str) -> String {
    let number = if !value.is_finite() {
        "non-finite".to_owned()
    } else if value != 0.0 && !(1.0e-3..1.0e6).contains(&value.abs()) {
        format!("{value:.3e}")
    } else {
        format!("{value:.3}")
    };
    format!("{number:>12} {unit}")
}

fn paint_dotted_circle(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32) {
    let circumference = std::f32::consts::TAU * radius;
    let dot_count = (circumference / 10.0).round().clamp(32.0, 240.0) as usize;

    for index in 0..dot_count {
        let angle = std::f32::consts::TAU * index as f32 / dot_count as f32;
        let offset = EguiVec2::new(angle.cos(), angle.sin()) * radius;
        painter.circle_filled(center + offset, 1.35, color);
    }
}

#[cfg(test)]
mod tests {
    use super::format_diagnostic;

    #[test]
    fn huge_diagnostics_use_bounded_scientific_notation() {
        let formatted = format_diagnostic(f64::MAX, "J");
        assert_eq!(formatted.trim(), "1.798e308 J");
        assert!(formatted.len() <= 14);
    }

    #[test]
    fn non_finite_diagnostics_have_a_bounded_label() {
        assert_eq!(format_diagnostic(f64::INFINITY, "W").trim(), "non-finite W");
    }
}
