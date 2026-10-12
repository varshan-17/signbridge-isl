# SignBridge — Indian Sign Language to text and speech

A React + TypeScript webcam app with **built-in ISL alphabet and word recognition**, message editing, and text-to-speech. Recognition runs locally in the browser. Personal sign recordings remain optional.

## Run

Requires Node.js 22.18+ (Node 24 recommended).

```sh
npm install
npm run setup:vision
npm run setup:models
npm run dev
```

Open the localhost URL printed by Vite, normally http://127.0.0.1:5173. Camera access requires localhost or HTTPS. The converted word model is included in `public/models`; no Python server or API key is required.

```sh
npm test
npm run build
npm run preview
```

Deploy `dist` to an HTTPS static host at the domain root. Tracking models, recognition weights, fonts and runtime assets are served by the app; camera frames are not uploaded.

## Essential controls

- **Letters:** recognize ISL A–Z using a pretrained SignLens landmark classifier. Hold a letter steadily for about a second to add it. Lower your hands briefly before repeating the same letter. Press **Space** to complete a fingerspelled word.
- **Words:** recognize isolated signs from the pretrained AI4Bharat INCLUDE vocabulary (263 classes). Keep your face, shoulders and hands in view. Press **Recognize a word**, wait for the countdown, and perform one sign during the four-second capture. The supported vocabulary is listed below that button. Examples include HELLO, THANK YOU, GOOD MORNING, FAMILY, SCHOOL and HOSPITAL.
- **My signs:** use your existing personal sign library. Teach at least three examples per label in Teach signs; import/export backups work as before.
- **Speak message:** reads the editable text using a device/browser English voice. Automatic speech reads a recognized word or the latest fingerspelled word when Space is pressed.

The simulated demo remains explicitly labelled and does not run sign recognition. Saved messages last for the current session; personal landmark recordings are stored in this browser.

## Model details and limits

### Built-in letters

The SignLens model contains 35 output classes (A–Z and digits 1–9). This app exposes the requested alphabet; digit predictions are rejected rather than relabelled as letters. Its 156 input features match the published browser pipeline: aspect correction, wrist centering, palm scaling, the trained rotation convention, 15 joint angles per hand, and raw wrist-x ordering of the two slots. A compact Float32 weight file runs the four-layer network locally.

The app requires an 80% model score, a margin over competing classes, the expected hand count, and a stable temporal match. These thresholds reduce uncertain output; they do **not** guarantee that an unsupported sign will be rejected. Some similar or motion-dependent letters can be confused.

### Built-in words

The official OpenHands INCLUDE LSTM checkpoint is converted to ONNX. Class order comes from the lexicographically sorted unique `Word` entries in the released training split, as required by the original dataset implementation. Readable labels use its normalized gloss map; semantically identical labels are combined after softmax.

Input uses the published 27-joint MediaPipe preset: seven face/upper-body joints and ten joints per hand. Only x/y coordinates are used, with mean shoulder centering and shoulder-distance scaling across the clip. Hands are associated with anatomical pose wrists. Captures are resampled to 48 frames; ONNX inference uses a single WASM thread. The original released network's tensor convention is preserved. A parity test compares the ONNX output with the original PyTorch forward pass.

Word results require a 70% model score and a 25 percentage-point margin over another label. Poor captures and uncertain outputs are rejected. The four-second isolated-sign workflow does not perform continuous sentence segmentation or ISL grammar translation.

### Personal signs

Your existing short-sequence recognizer uses normalized two-hand landmarks, relative wrist positions and whole-hand trajectories with dynamic time warping. It needs at least three examples per label and up to eight can be stored. Personal scores measure similarity to those examples.

**Real-world accuracy of this app has not been measured.** The presence of pretrained weights and model scores is not an accuracy claim. Lighting, occlusion, signer differences, signs outside the vocabulary, body/face cues, and long or similar movements can cause mistakes. Review the text before relying on it. This remains a recognition prototype for isolated signs.

Camera access stops when the tab is hidden, the camera disconnects, or Stop is pressed. Word pose tracking runs only during a capture. Browser speech may use the device's speech provider.

## Source map

- `src/App.tsx`: focused recognition modes, captures, text and speech; existing personal-library controls.
- `src/builtinLetters.ts`: exact alphabet feature pipeline and pretrained MLP.
- `src/builtinWords.ts`: local ONNX word inference.
- `src/wordFeatures.ts`: anatomical hand slots, joint selection and clip normalization.
- `src/useCamera.ts`: camera, hand tracker and capture-only body tracker.
- `src/recognition.ts`: personal recognition and duplicate prevention.
- `tests`: algorithm and model conversion checks.
- `scripts/setup-models.mjs`: pinned alphabet download and local body-tracker setup.
- `scripts/convert-word-model.py`: adaptation of the released word checkpoint.

See [MODEL_NOTICES.md](MODEL_NOTICES.md) for provenance and license information. Model files include source/checkpoint hashes.

## Reproduce the word conversion (optional)

The app already includes the converted model; these steps are only for reproducing it. Use Python 3.12 with CPU PyTorch and ONNX:

```sh
node scripts/download-word-source.mjs
python -m pip install --target tmp/model-conversion torch onnx
python scripts/convert-word-model.py
npm test
```

The conversion reads the checkpoint with PyTorch's restricted weights-only loader. Inert placeholders represent the two serialized training callback names; their code is not imported or executed.

## References

- [SignLens ISL model](https://huggingface.co/venmugilrajan/SignLens-Landmark-Model)
- [AI4Bharat OpenHands and pretrained checkpoints](https://github.com/AI4Bharat/OpenHands/releases/tag/checkpoints_v1)
- [INCLUDE dataset](https://huggingface.co/datasets/ai4bharat/INCLUDE)
- [Official ISLRTC learning resources](https://islrtc.nic.in/document-category/awareness-material/)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js)

The local A–Z photo guide uses the 26 visualization images from the [RealSign ISL fingerspelling dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset), released under CC0. Word references link to the [ISLRTC dictionary](https://divyangjan.depwd.gov.in/islrtc/), where signs can be watched as videos.
