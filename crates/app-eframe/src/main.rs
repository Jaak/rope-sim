use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2 as EguiVec2,
};
use ropesim_physics::{
    Diagnostics, IntegratorKind, KinematicTarget, MotionCommand, ReconfigureOutcome, RopeModelKind,
    Simulation, SimulationConfig, Vec2,
};

mod scenario;

use scenario::ScenarioController;

const DEFAULT_FIXED_DT: f64 = 1.0 / 240.0;
const MAX_FRAME_DT: f64 = 0.1;
const MAX_STEPS_PER_FRAME: usize = 32;
const PAYLOAD_RADIUS: f32 = 13.0;
const GRAB_RADIUS: f32 = 24.0;
const SCENARIO_DIRECTORY: &str = "scenarios";
const PERFORMANCE_SMOOTHING_SECONDS: f64 = 0.5;
const TAIL_FPS_WINDOW_SECONDS: f64 = 5.0;
const TAIL_FPS_REFRESH_SECONDS: f64 = 0.25;

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
    performance: PerformanceMetrics,
    dragging_payload: bool,
    viewport_id: Option<egui::Id>,
    previous_drag_position: Option<Vec2>,
    drag_velocity: Vec2,
    scenarios: ScenarioController,
    scenario_name: String,
    scenario_status: Option<String>,
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
            performance: PerformanceMetrics::default(),
            dragging_payload: false,
            viewport_id: None,
            previous_drag_position: None,
            drag_velocity: Vec2::ZERO,
            scenarios: ScenarioController::default(),
            scenario_name: "recorded-motion".to_owned(),
            scenario_status: None,
            error_message: None,
        }
    }

    fn controls(&mut self, root_ui: &mut egui::Ui) {
        let previous_config = self.config;
        let mut reset_requested = false;
        let mut scenario_replay_started = false;

        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(290.0)
            .show(root_ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.heading("RopeSim");
                        ui.label(format!(
                            "{} · {}",
                            self.config.rope_model.display_name(),
                            self.config.integrator.display_name()
                        ));
                        ui.add_space(8.0);

                        ui.horizontal(|ui| {
                            let pause_label = if self.paused { "Resume" } else { "Pause" };
                            if ui
                                .button(pause_label)
                                .on_hover_text("Shortcut: Space")
                                .clicked()
                            {
                                self.toggle_paused();
                            }
                            if ui
                                .add_enabled(self.paused, egui::Button::new("Single step"))
                                .on_hover_text("Shortcut: Right Arrow")
                                .clicked()
                            {
                                self.request_single_step();
                            }
                            if ui.button("Reset").clicked() {
                                reset_requested = true;
                            }
                        });

                        ui.separator();
                        ui.heading("Rope");
                        ui.add(
                            egui::Slider::new(&mut self.config.segment_count, 1..=64)
                                .text("Pieces"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.config.rope_length, 2.0..=30.0)
                                .text("Length (m)"),
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
                        let rigidity_label =
                            if self.config.rope_model == RopeModelKind::StandardLinearSolid {
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
                                egui::Slider::new(
                                    &mut self.config.axial_viscosity,
                                    0.01..=1_000_000.0,
                                )
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
                        ui.add(
                            egui::Slider::new(&mut self.time_scale, 0.1..=2.0).text("Time scale"),
                        );
                        ui.label(format!("Fixed step: {:.3} ms", DEFAULT_FIXED_DT * 1000.0));
                        ui.label(format!(
                            "Automatic substeps: {} (effective {:.3} ms)",
                            self.integration_substeps,
                            DEFAULT_FIXED_DT * 1000.0 / self.integration_substeps as f64
                        ));

                        ui.separator();
                        ui.heading("Recorded scenario");
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    !self.scenarios.is_active(),
                                    egui::Button::new("Record"),
                                )
                                .on_hover_text("Reset and record configuration plus payload motion")
                                .clicked()
                            {
                                self.start_scenario_recording();
                            }
                            if ui
                                .add_enabled(self.scenarios.is_active(), egui::Button::new("Stop"))
                                .clicked()
                            {
                                self.stop_scenario_activity();
                            }
                            if ui
                                .add_enabled(
                                    self.scenarios.has_recording() && !self.scenarios.is_active(),
                                    egui::Button::new("Replay"),
                                )
                                .on_hover_text("Replay with the currently selected integrator")
                                .clicked()
                            {
                                self.start_scenario_replay();
                                scenario_replay_started = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Fixture name");
                            ui.text_edit_singleline(&mut self.scenario_name);
                        });
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    self.scenarios.has_recording() && !self.scenarios.is_active(),
                                    egui::Button::new("Save JSON"),
                                )
                                .clicked()
                            {
                                self.save_recorded_scenario();
                            }
                            if ui
                                .add_enabled(
                                    !self.scenarios.is_active(),
                                    egui::Button::new("Load JSON"),
                                )
                                .clicked()
                            {
                                self.load_recorded_scenario();
                            }
                        });
                        if self.scenarios.is_recording() {
                            ui.colored_label(Color32::from_rgb(235, 95, 95), "Recording...");
                        } else if self.scenarios.is_replaying() {
                            ui.colored_label(Color32::from_rgb(100, 180, 240), "Replaying...");
                        } else if let Some((duration, command_count)) =
                            self.scenarios.recording_summary()
                        {
                            ui.small(format!(
                                "Saved in memory: {duration:.2} s, {command_count} commands"
                            ));
                        } else {
                            ui.small("Record starts from a reset state.");
                        }
                        if let Some(status) = &self.scenario_status {
                            ui.small(status);
                        }

                        ui.separator();
                        ui.heading("Diagnostics");
                        egui::Grid::new("performance_diagnostics_grid")
                            .num_columns(2)
                            .spacing([12.0, 3.0])
                            .show(ui, |ui| {
                                diagnostic_row(
                                    ui,
                                    "Frame rate",
                                    self.performance.frames_per_second(),
                                    "FPS",
                                );
                                diagnostic_row(
                                    ui,
                                    "1% low",
                                    self.performance.one_percent_low_fps(),
                                    "FPS",
                                );
                                diagnostic_row(
                                    ui,
                                    "Physics time",
                                    self.performance.physics_time_ms(),
                                    "ms/frame",
                                );
                                diagnostic_row(
                                    ui,
                                    "Physics load",
                                    self.performance.physics_load_percent(),
                                    "%",
                                );
                                diagnostic_row(
                                    ui,
                                    "Simulation rate",
                                    self.performance.simulation_rate(),
                                    "x real time",
                                );
                                diagnostic_row(ui, "Time", self.diagnostics.simulation_time, "s");
                            });

                        egui::CollapsingHeader::new("Physical state")
                            .default_open(false)
                            .show(ui, |ui| {
                                egui::Grid::new("physical_diagnostics_grid")
                                    .num_columns(2)
                                    .spacing([12.0, 3.0])
                                    .show(ui, |ui| {
                                        diagnostic_row(
                                            ui,
                                            "Kinetic",
                                            self.diagnostics.kinetic_energy,
                                            "J",
                                        );
                                        diagnostic_row(
                                            ui,
                                            "Elastic",
                                            self.diagnostics.elastic_energy,
                                            "J",
                                        );
                                        diagnostic_row(
                                            ui,
                                            "Gravity",
                                            self.diagnostics.gravitational_energy,
                                            "J",
                                        );
                                        diagnostic_row(
                                            ui,
                                            "Total",
                                            self.diagnostics.total_mechanical_energy,
                                            "J",
                                        );
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
                                        diagnostic_row(
                                            ui,
                                            "Max speed",
                                            self.diagnostics.maximum_node_speed,
                                            "m/s",
                                        );
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
                                    });
                            });

                        egui::CollapsingHeader::new("Solver details")
                            .default_open(false)
                            .show(ui, |ui| {
                                egui::Grid::new("solver_diagnostics_grid")
                                    .num_columns(2)
                                    .spacing([12.0, 3.0])
                                    .show(ui, |ui| {
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
                            });

                        ui.add_space(8.0);
                        ui.small(
                            "Drag the payload directly. Its mouse velocity is retained on release.",
                        );

                        if let Some(message) = &self.error_message {
                            ui.add_space(8.0);
                            ui.colored_label(Color32::LIGHT_RED, message);
                        }
                    });
            });

        if reset_requested {
            self.reset_simulation();
        } else if self.config != previous_config && !scenario_replay_started {
            self.stop_scenario_activity();
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

    fn handle_keyboard_shortcuts(&mut self, context: &egui::Context) {
        // Leave arrow keys and Space available to controls while they have
        // keyboard focus, but treat the simulation viewport as shortcut scope.
        let focused_widget = context.memory(|memory| memory.focused());
        if focused_widget.is_some() && focused_widget != self.viewport_id {
            return;
        }

        let (toggle_pause, step_forward) = context.input_mut(|input| {
            let space_pressed = input.events.iter().any(|event| {
                matches!(
                    event,
                    egui::Event::Key {
                        key: egui::Key::Space,
                        pressed: true,
                        modifiers: egui::Modifiers::NONE,
                        ..
                    }
                )
            });
            let toggle_pause = input.events.iter().any(|event| {
                matches!(
                    event,
                    egui::Event::Key {
                        key: egui::Key::Space,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                        ..
                    }
                )
            });

            if space_pressed {
                input.events.retain(|event| {
                    !matches!(
                        event,
                        egui::Event::Key {
                            key: egui::Key::Space,
                            pressed: true,
                            modifiers: egui::Modifiers::NONE,
                            ..
                        }
                    )
                });
            }

            let step_forward = input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight);
            (toggle_pause, step_forward)
        });

        if toggle_pause {
            self.toggle_paused();
        }
        if step_forward {
            self.request_single_step();
        }
    }

    fn toggle_paused(&mut self) {
        self.paused = !self.paused;
        self.accumulator = 0.0;
        self.single_step_requested = false;
    }

    fn request_single_step(&mut self) {
        if self.paused {
            self.single_step_requested = true;
        }
    }

    fn viewport(&mut self, root_ui: &mut egui::Ui, frame_dt: f64) {
        egui::CentralPanel::default().show(root_ui, |ui| {
            let size = ui.available_size().max(EguiVec2::new(1.0, 1.0));
            let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
            self.viewport_id = Some(response.id);
            let transform = ViewTransform::new(response.rect, self.config);

            self.update_drag(&response, transform, frame_dt);
            let simulation_time_before = self.diagnostics.simulation_time;
            let physics_start = Instant::now();
            self.advance_simulation(frame_dt);
            self.performance.observe_physics(
                physics_start.elapsed().as_secs_f64(),
                frame_dt,
                (self.diagnostics.simulation_time - simulation_time_before).max(0.0),
            );
            self.paint_scene(&painter, response.rect, transform);
        });
    }

    fn update_drag(&mut self, response: &egui::Response, transform: ViewTransform, frame_dt: f64) {
        if self.scenarios.is_replaying() {
            return;
        }

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
                    self.scenarios.record(
                        self.diagnostics.simulation_time,
                        MotionCommand::SetTarget(target),
                    );
                } else {
                    let base_duration =
                        (frame_dt.min(MAX_FRAME_DT) * self.time_scale).max(DEFAULT_FIXED_DT);
                    if let Err(error) = self
                        .simulation
                        .interpolate_payload_target(target, base_duration)
                    {
                        self.error_message = Some(error.to_string());
                    } else {
                        self.scenarios.record(
                            self.diagnostics.simulation_time,
                            MotionCommand::InterpolateTarget {
                                target,
                                duration: base_duration,
                            },
                        );
                    }
                }
            }
        } else if self.dragging_payload {
            self.dragging_payload = false;
            self.previous_drag_position = None;
            let release_velocity = self.simulation.payload_velocity();
            self.simulation.release_payload(release_velocity);
            self.scenarios.record(
                self.diagnostics.simulation_time,
                MotionCommand::Release {
                    velocity: release_velocity,
                },
            );
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
            if let Err(error) = self.apply_due_replay_commands() {
                self.recover_from_step_error(error);
                return;
            }
            if self
                .scenarios
                .finish_replay_if_complete(self.diagnostics.simulation_time)
            {
                self.paused = true;
                self.accumulator = 0.0;
                break;
            }

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
                match self.simulation.step_without_diagnostics(substep_dt) {
                    Ok(()) => {}
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

            self.diagnostics = self.simulation.diagnostics();
            self.error_message = None;
            self.accumulator -= DEFAULT_FIXED_DT;
            steps += 1;

            if self
                .scenarios
                .finish_replay_if_complete(self.diagnostics.simulation_time)
            {
                self.paused = true;
                self.accumulator = 0.0;
                break;
            }
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

    fn start_scenario_recording(&mut self) {
        self.reset_simulation();
        self.scenarios
            .begin_recording(self.config, DEFAULT_FIXED_DT);
        self.scenario_status = None;
        self.paused = false;
        self.single_step_requested = false;
    }

    fn start_scenario_replay(&mut self) {
        let selected_integrator = self.config.integrator;
        let Some(recorded_config) = self.scenarios.recorded_config_for(selected_integrator) else {
            return;
        };

        self.config = recorded_config;
        self.reset_simulation();
        if self.scenarios.begin_replay() {
            self.paused = false;
            self.single_step_requested = false;
        }
    }

    fn stop_scenario_activity(&mut self) {
        self.scenarios.stop(self.diagnostics.simulation_time);
    }

    fn save_recorded_scenario(&mut self) {
        let result = (|| {
            let path = scenario_path(&self.scenario_name)?;
            let json = self
                .scenarios
                .saved_json()
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "there is no recorded scenario to save".to_owned())?;
            fs::create_dir_all(SCENARIO_DIRECTORY)
                .map_err(|error| format!("could not create scenario directory: {error}"))?;
            fs::write(&path, json)
                .map_err(|error| format!("could not write {}: {error}", path.display()))?;
            Ok::<_, String>(path)
        })();

        match result {
            Ok(path) => {
                self.scenario_status = Some(format!("Saved {}", path.display()));
                self.error_message = None;
            }
            Err(error) => self.error_message = Some(error),
        }
    }

    fn load_recorded_scenario(&mut self) {
        let result = (|| {
            let path = scenario_path(&self.scenario_name)?;
            let json = fs::read_to_string(&path)
                .map_err(|error| format!("could not read {}: {error}", path.display()))?;
            self.scenarios
                .load_json(&json)
                .map_err(|error| error.to_string())?;
            Ok::<_, String>(path)
        })();

        match result {
            Ok(path) => {
                self.scenario_status = Some(format!("Loaded {}", path.display()));
                self.error_message = None;
            }
            Err(error) => self.error_message = Some(error),
        }
    }

    fn apply_due_replay_commands(&mut self) -> Result<(), String> {
        for command in self
            .scenarios
            .take_due_commands(self.diagnostics.simulation_time)
        {
            match command {
                MotionCommand::SetTarget(target) => {
                    self.simulation.set_payload_target(Some(target));
                }
                MotionCommand::InterpolateTarget { target, duration } => self
                    .simulation
                    .interpolate_payload_target(target, duration)
                    .map_err(|error| error.to_string())?,
                MotionCommand::Release { velocity } => {
                    self.simulation.release_payload(velocity);
                }
            }
        }
        Ok(())
    }

    fn reset_simulation(&mut self) {
        self.stop_scenario_activity();
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
        let post_failure_duration = if self.scenarios.is_recording() {
            DEFAULT_FIXED_DT
        } else {
            0.0
        };
        self.scenarios
            .stop(self.diagnostics.simulation_time + post_failure_duration);
        self.scenario_status = None;
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
        self.performance.observe_frame(frame_dt);

        self.handle_keyboard_shortcuts(ui.ctx());
        self.controls(ui);
        self.viewport(ui, frame_dt);
        ui.ctx().request_repaint();
    }
}

#[derive(Default)]
struct PerformanceMetrics {
    frame_interval: Option<f64>,
    physics_time: Option<f64>,
    simulation_rate: Option<f64>,
    frame_intervals: VecDeque<f64>,
    frame_window_duration: f64,
    tail_refresh_elapsed: f64,
    one_percent_low_fps: f64,
}

impl PerformanceMetrics {
    fn observe_frame(&mut self, frame_interval: f64) {
        if !frame_interval.is_finite() || frame_interval <= 0.0 {
            return;
        }
        update_smoothed_value(&mut self.frame_interval, frame_interval, frame_interval);
        self.frame_intervals.push_back(frame_interval);
        self.frame_window_duration += frame_interval;
        while self.frame_intervals.len() > 1
            && self.frame_window_duration
                - self
                    .frame_intervals
                    .front()
                    .copied()
                    .expect("a nonempty frame window has a first sample")
                >= TAIL_FPS_WINDOW_SECONDS
        {
            self.frame_window_duration -= self
                .frame_intervals
                .pop_front()
                .expect("a nonempty frame window can remove its first sample");
        }

        self.tail_refresh_elapsed += frame_interval;
        if self.one_percent_low_fps == 0.0 || self.tail_refresh_elapsed >= TAIL_FPS_REFRESH_SECONDS
        {
            self.refresh_one_percent_low_fps();
            self.tail_refresh_elapsed = 0.0;
        }
    }

    fn observe_physics(&mut self, physics_time: f64, frame_interval: f64, simulation_advance: f64) {
        update_smoothed_value(&mut self.physics_time, physics_time, frame_interval);
        update_smoothed_value(
            &mut self.simulation_rate,
            simulation_advance / frame_interval,
            frame_interval,
        );
    }

    fn frames_per_second(&self) -> f64 {
        self.frame_interval
            .filter(|interval| *interval > 0.0)
            .map_or(0.0, |interval| interval.recip())
    }

    fn one_percent_low_fps(&self) -> f64 {
        self.one_percent_low_fps
    }

    fn physics_time_ms(&self) -> f64 {
        1000.0 * self.physics_time.unwrap_or(0.0)
    }

    fn physics_load_percent(&self) -> f64 {
        match (self.physics_time, self.frame_interval) {
            (Some(physics_time), Some(frame_interval)) if frame_interval > 0.0 => {
                100.0 * physics_time / frame_interval
            }
            _ => 0.0,
        }
    }

    fn simulation_rate(&self) -> f64 {
        self.simulation_rate.unwrap_or(0.0)
    }

    fn refresh_one_percent_low_fps(&mut self) {
        let mut intervals: Vec<_> = self.frame_intervals.iter().copied().collect();
        intervals.sort_by(|left, right| right.total_cmp(left));
        let tail_count = ((intervals.len() as f64 * 0.01).ceil() as usize).max(1);
        let mean_tail_interval = intervals[..tail_count].iter().sum::<f64>() / tail_count as f64;
        self.one_percent_low_fps = mean_tail_interval.recip();
    }
}

fn update_smoothed_value(value: &mut Option<f64>, sample: f64, sample_interval: f64) {
    if !sample.is_finite() || sample < 0.0 || !sample_interval.is_finite() || sample_interval <= 0.0
    {
        return;
    }

    if let Some(previous) = value {
        let weight = 1.0 - (-sample_interval / PERFORMANCE_SMOOTHING_SECONDS).exp();
        *previous += weight * (sample - *previous);
    } else {
        *value = Some(sample);
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

fn scenario_path(name: &str) -> Result<PathBuf, String> {
    let trimmed = name.trim();
    let stem = trimmed.strip_suffix(".json").unwrap_or(trimmed);
    if stem.is_empty()
        || !stem
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err("fixture name must contain only ASCII letters, numbers, '-' or '_'".to_owned());
    }
    Ok(PathBuf::from(SCENARIO_DIRECTORY).join(format!("{stem}.json")))
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
    use std::path::PathBuf;

    use super::{PerformanceMetrics, format_diagnostic, scenario_path};

    #[test]
    fn performance_metrics_report_frame_and_simulation_throughput() {
        let mut metrics = PerformanceMetrics::default();
        metrics.observe_frame(1.0 / 60.0);
        metrics.observe_physics(0.004, 1.0 / 60.0, 1.0 / 60.0);

        assert!((metrics.frames_per_second() - 60.0).abs() < 1.0e-12);
        assert!((metrics.one_percent_low_fps() - 60.0).abs() < 1.0e-12);
        assert!((metrics.physics_time_ms() - 4.0).abs() < 1.0e-12);
        assert!((metrics.physics_load_percent() - 24.0).abs() < 1.0e-12);
        assert!((metrics.simulation_rate() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn one_percent_low_fps_retains_slow_frames() {
        let mut metrics = PerformanceMetrics::default();
        for _ in 0..99 {
            metrics.observe_frame(0.01);
        }
        metrics.observe_frame(0.1);

        assert!((metrics.one_percent_low_fps() - 10.0).abs() < 1.0e-12);
    }

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

    #[test]
    fn scenario_names_cannot_escape_the_fixture_directory() {
        assert_eq!(
            scenario_path("rapid-drag.json").unwrap(),
            PathBuf::from("scenarios").join("rapid-drag.json")
        );
        assert!(scenario_path("../outside").is_err());
        assert!(scenario_path("nested/name").is_err());
    }
}
