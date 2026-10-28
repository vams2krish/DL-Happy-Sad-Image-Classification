# DL-Happy-Sad-Image-Classification

**Happy vs. sad image classification using a TensorFlow/Keras convolutional neural network.**

An educational computer vision project covering image loading, preprocessing, CNN training, evaluation, single-image prediction, and model persistence. Explore the full workflow in [main.ipynb](main.ipynb), or load the included trained model to try inference directly.

## Project photos

| Happy example | Sad example |
| --- | --- |
| ![Happy example input](Happy-Guy.jpg) | ![Sad example input used in the notebook](Sad-Guy.jpg) |

These are sample inputs, not benchmark results.

## Features

- Two directory-based classes: `happy` (0) and `sad` (1).
- Images resized to 256 × 256, with pixel values normalized by dividing by 255.
- Three convolution/max-pooling blocks and a dense binary classifier.
- Training loss and accuracy plots, plus TensorBoard logging.
- Precision, recall, and binary accuracy evaluation.
- Saved model loading and single-image predictions.

## Repository structure

```text
DL-Happy-Sad-Image-Classification/
├── main.ipynb                 # Complete training and inference walkthrough
├── Happy-Guy.jpg              # Example input
├── Sad-Guy.jpg                # Notebook inference example
├── data/
│   ├── happy/                # Images labeled happy
│   └── sad/                  # Images labeled sad
├── models/
│   └── happysadmodel.h5      # Saved Keras model
├── .gitignore
└── README.md
```

Training creates a local `logs/` directory, excluded from Git.

## Installation

The notebook's recorded environment used Python 3.12, TensorFlow 2.17.0, Keras 3.4.1, NumPy 1.26.4, OpenCV 4.10.0.84, Matplotlib 3.9.1, and Pillow 10.2.0. These versions document that environment; they are not a complete dependency lockfile.

With Python 3.12 installed:

```bash
git clone https://github.com/vams2krish/DL-Happy-Sad-Image-Classification.git
cd DL-Happy-Sad-Image-Classification
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and open the notebook:

```bash
python -m pip install tensorflow==2.17.0 keras==3.4.1 numpy==1.26.4 opencv-python==4.10.0.84 matplotlib==3.9.1 pillow==10.2.0 jupyterlab
python -m jupyterlab main.ipynb
```

Run from the repository root so relative paths resolve. CPU execution is supported; a compatible GPU is optional.

## Training workflow

1. Open `main.ipynb` and choose your environment's Python kernel.
2. Review the image inspection section before executing it: it contains an `os.remove` call that can delete files identified as unsupported.
3. Load and inspect image batches, then normalize pixel values.
4. Create the training, validation, and test datasets.
5. Build the CNN and train for the configured 200 epochs.
6. Inspect training curves and evaluate precision, recall, and accuracy.
7. Predict on `Sad-Guy.jpg`, or replace that path with another input.
8. Save and reload `models/happysadmodel.h5`. Saving overwrites that file.

**Inspection compatibility:** the notebook calls `Image.what(image_path)`, which is not a Pillow API. Replace that call with this snippet while keeping the surrounding logic:

```python
with Image.open(image_path) as candidate:
    tip = (candidate.format or "").lower()
```

The existing exception handler prints issues and continues; an inspection exception does not itself remove an unreadable image. Back up your dataset before cleaning it.

To view TensorBoard logs after training:

```bash
tensorboard --logdir logs
```

## Model architecture

| Stage | Configuration |
| --- | --- |
| Input | 256 × 256 × 3 image, normalized to [0, 1] |
| Block 1 | Conv2D: 16 filters, 3 × 3, stride 1, ReLU → MaxPooling2D |
| Block 2 | Conv2D: 32 filters, 3 × 3, stride 1, ReLU → MaxPooling2D |
| Block 3 | Conv2D: 16 filters, 3 × 3, stride 1, ReLU → MaxPooling2D |
| Dense stage | Flatten → Dense(256), ReLU |
| Output | Dense(1), sigmoid |
| Optimizer / loss | Adam / binary cross-entropy |
| Training | 200 epochs, validation data, TensorBoard callback |

The loader uses its default batch size of 32. Split sizes are computed at the **batch level** as `int(len(data) * 0.7)`, `int(len(data) * 0.2)`, and `int(len(data) * 0.1) + 1`. These approximate the intended proportions; they are not exact image-level 70/20/10 splits.

## Predict with the saved model

Run this standalone example from the repository root to use the saved model without retraining:

```python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/happysadmodel.h5", compile=False)
image = tf.keras.utils.load_img("Sad-Guy.jpg")
image = tf.keras.utils.img_to_array(image)
image = tf.image.resize(image, (256, 256)) / 255.0
score = float(model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0])

label = "sad" if score > 0.5 else "happy"
print(f"Predicted class: {label}; sigmoid score: {score:.4f}")
```

A score above 0.5 maps to `sad`; a score at or below 0.5 maps to `happy`. This output is not a calibrated measure of someone's emotional state.

The example uses RGB input to match training. The notebook's original inference cells use OpenCV, which loads BGR images. When using that approach, convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before resizing and predicting.

## Evaluation and limitations

The recorded notebook test pass reports **precision 1.0, recall 1.0, and binary accuracy 1.0**. These are historical saved outputs, not a newly reproduced evaluation or evidence of general accuracy.

The dataset is small and mixes photos, illustrations, group scenes, and graphics. The directory loader shuffles by default, and separate `take`/`skip` pipelines do not establish fixed held-out membership across iterations. Changing membership or overlap can make scores optimistic. For reliable evaluation, split file paths once with a fixed seed, check for duplicates across splits, and reserve an untouched test set before training.

Additional limitations:

- Labels describe dataset categories; images alone do not establish actual feelings or mental health.
- The notebook does not implement augmentation, transfer learning, early stopping, or a deployment interface.
- HDF5 is a legacy Keras model format; future training runs can save new artifacts as `.keras`.
- Runtime and results depend on the environment and dataset.

## Data and licensing

The repository includes labeled images in `data/` and two sample images at the root. Filenames reference multiple external sources, but a complete provenance and license manifest is not included. Verify image permissions and attribution requirements before reuse or redistribution.

There is currently no project license file. Public visibility alone does not grant a software or image license.

## Future improvements

- Establish deterministic image-level splits and audit duplicate images.
- Correct the image inspection helper and standardize RGB preprocessing.
- Evaluate on a larger, independently collected and appropriately licensed dataset.
- Add augmentation, early stopping, and a transfer-learning baseline.
- Package validated inference in a script or small application.
