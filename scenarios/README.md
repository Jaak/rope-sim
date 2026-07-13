# Recorded scenarios

The native app saves versioned JSON motion fixtures in this directory. Enter a fixture name in the
**Recorded scenario** controls, record and stop a motion, then choose **Save JSON**.

`cargo test --workspace` automatically discovers every `*.json` file here and replays it using the
integrators listed in its `test_integrators` field. New recordings default to Backward Euler and
TR-BDF2. The list can be edited in the JSON when a fixture should cover another integrator.

Commit useful JSON fixtures so solver regressions reproduce in local tests and CI.
