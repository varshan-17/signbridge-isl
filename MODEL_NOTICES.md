# Pretrained model provenance

## SignLens alphabet classifier

Author: Venmugil Rajan. Published model license: Apache License 2.0.

- Model: https://huggingface.co/venmugilrajan/SignLens-Landmark-Model
- Pinned revision: `62c07b470c5a67dcadda6e7429a76e6a1766b749`
- Original asset: `isl_landmarks_weights.json`
- Published browser preprocessing reference: https://huggingface.co/spaces/venmugilrajan/SignLens/blob/main/app.js
- Adaptation: matrices are packed as Float32 values in `public/models/isl-letters.bin`; the feature calculation is implemented in TypeScript, with conservative abstention and duplicate prevention added.
- The original source JSON hash and source URL are stored in `isl-letters.json`.

Publisher accuracy claims have not been independently reproduced in this app and are not displayed as app accuracy.

## OpenHands INCLUDE word classifier

Author/project: AI4Bharat OpenHands. Project license: Apache License 2.0.

- Source: https://github.com/AI4Bharat/OpenHands
- Checkpoint release: https://github.com/AI4Bharat/OpenHands/releases/tag/checkpoints_v1
- Original assets: `include_lstm.zip`, `include_metadata.zip`
- Adaptation: the 263-class checkpoint is exported to `public/models/isl-words.onnx` with a fixed 48-frame input; the original network forwarding convention, weights and output class order are preserved. Numeric prefixes in the gloss labels are removed and the source normalized gloss mapping supplies readable text. No vocabulary entries or scores are fabricated.
- The checkpoint and ONNX SHA-256 hashes are stored in `isl-words.json`.

OpenHands citation:

Prem Selvaraj, Gokul NC, Pratyush Kumar and Mitesh Khapra, “OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages,” 2021, arXiv:2110.05877.

Underlying INCLUDE dataset citation:

Advaith Sridhar, Rohith Gandhi Ganesan, Pratyush Kumar and Mitesh Khapra, “INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition,” ACM Multimedia 2020, DOI:10.1145/3394171.3413528.

No source dataset videos or camera recordings are bundled. The released model is a closed-vocabulary research classifier; it can produce high scores for wrong or out-of-vocabulary signs. It is not a validated continuous ISL translator.

The shared Apache license text is included at `licenses/Apache-2.0.txt`.

## Tracking and inference runtimes

MediaPipe tracking models and runtime are provided by Google. ONNX Runtime Web is provided by Microsoft under its package MIT license. These dependencies and their accompanying package licenses retain their own terms.
