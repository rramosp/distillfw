- [x] review userguide, maybe remove the localhost calls, and/or explain well what is the backend it is hitting. (Updated with detailed backend architecture and 3 distinct execution targets in docs/userguide.md)
- [x] in any task in the UI disable buttons when process start, add a stop button. Then, when a task finishes, show its result (data, artifacts, metrics, etc.), and change the buttons to say 'Start Over' (which effectively clears the state and lets you start a new task). If there are any errors, the show them and show the original buttons.
- [x] in teacher inference, keep track on how many retries and error types, making them available in the interface and the API
- [x] In the UI change '5. Training telemetry' to '5. Model training' and add in the bottom the button 'start distillation training' (which should start the training pipeline)
- [x] in the UI, in the 'Interactive Model Inference Playground', make inference and show the answer of (1) the student model before distillation, (2) the teacher model, and (3) the student model after distillation.

- [ ] local UI with two modes: simulated backend, and GCP backend

