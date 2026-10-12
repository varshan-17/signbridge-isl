# Verification — 12 October 2026

## Pretrained recognition update

- `npm test`: 14 tests passed. Added pretrained alphabet preprocessing, valid model weights, absent-hand/uncertain/digit rejection, anatomical word hand slots, shoulder normalization, label-order checks and ONNX/PyTorch output parity. Maximum ONNX parity error is below 0.0001 on the fixed test tensor.
- Built-in alphabet: licensed SignLens weights, 35 original classes; the app exposes A–Z. Model and feature dimensions validated.
- Built-in words: official AI4Bharat INCLUDE checkpoint, 263 original classes and 262 unique readable labels. The source training-label order is retained. Word inference runs locally with the bundled WASM runtime.
- Browser: alphabet ready by default; Words mode initializes the ONNX session and displays its supported vocabulary; existing personal sign library remains present.
- Live browser check: the alphabet and word models both reached their ready state; the word list showed 262 labels, including HELLO and THANK YOU. The A–Z photo selector loaded local image files, and the selected B photo rendered at its expected URL. The word reference switched to an ISLRTC search link for the entered HELLO label and hid the unrelated letter photo. Browser console showed no errors.
- The 26 image files are copied into the production build. The photo set is the RealSign fingerspelling visualization set (CC0); source credit and license are included in the app and repository.
- Production build: TypeScript and Vite build passed. Both pretrained models initialize at the production preview; HELLO and THANK YOU appear in the supported-word list. Mobile word controls fit at 390 × 844 without horizontal overflow.
- Runtime assets and model notices are bundled. No API or Python server is needed to run the app.
- Remaining limit: accuracy on real ISL letters and word captures has not been measured. Synthetic/model parity checks verify integration, not signer-independent accuracy.

## Initial personal-recognition prototype

- `npm test`: 9 tests passed. Includes hand-slot consistency, translation/scale invariance, unknown and ambiguous signs, sequence speed changes, whole-hand movement, duplicate prevention, repeated letters, word spacing, and malformed-library rejection.
- `npm run build`: TypeScript checks and production build passed. Local hand-tracking model and WASM assets included in `dist/vision`.
- Dependency install audit: 0 reported vulnerabilities after removing the unnecessary third-party test runner. Tests use Node's built-in runner.
- Edge browser: initial workspace renders; simulated demo produces `HELLO WORLD`; message editing and session saving work; speech enters the speaking state; sign-training letter selection and word-label entry work; recording is disabled while the camera is off; library empty state renders.
- MediaPipe tracker loaded in the browser and the interface reported CAMERA LIVE at approximately 11 FPS. No sign-recognition accuracy has been measured with real ISL samples.
- Responsive viewport: 390 × 844; document width equals viewport width (390 px), with stacked workspace cards. Mobile navigation adjusted to four columns so every section remains reachable.

Not verified: recognition accuracy across real signers, end-to-end recognition of an actual recorded ISL letter/word, audible pronunciation quality, long-running camera performance, and deployment to an external host. Camera permission was not granted during the latest browser check, so live signing still needs a user camera test. The app's demo output is explicitly simulated.
