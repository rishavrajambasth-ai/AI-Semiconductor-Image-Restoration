AI-Based Restoration of Degraded Semiconductor Images

1. Project Overview

This project presents an AI-based image restoration system for degraded semiconductor inspection images.

The system takes degraded ".npy" images as input and produces restored grayscale ".npy" images suitable for semiconductor inspection workflows.

The restoration model is a U-Net-style convolutional neural network implemented using PyTorch.

2. Model Architecture

The restoration network consists of:

- Encoder with convolutional blocks
- Batch Normalization and ReLU activation
- Max-pooling for feature extraction
- Bottleneck feature representation
- Transposed-convolution decoder
- Skip connections between encoder and decoder
- Final convolution layer
- Sigmoid output to constrain predictions to "[0, 1]"

The model accepts a 3-channel representation of the input and produces a restored image.

3. Repository Structure

MOSFET/
├── run.py
├── model.py
├── requirements.txt
├── README.md
└── models/
    └── restoration_model.pth

4. Requirements

Python dependencies are listed in "requirements.txt".

The model uses:

- Python
- PyTorch
- NumPy
- OpenCV
- Pillow
- scikit-image
- Matplotlib
- tqdm
- LPIPS

The trained model weights are included in the "models/" directory.

5. Input Format

The program accepts ".npy" files from an input directory.

Supported input arrays include:

- "(H, W)"
- "(H, W, 1)"
- "(H, W, 3)"

Input values are normalized to the range "[0, 1]".

6. Execution

Run the system using:

python run.py <input-dir> <output-dir>

Example:

python run.py ../data/degraded test_output_256

The program automatically:

1. Reads all ".npy" files from the input directory.
2. Loads the trained model.
3. Restores each image.
4. Produces a grayscale output.
5. Resizes the restored image to the required "256 × 256" target resolution.
6. Clips output values to "[0, 1]".
7. Saves the result as "float32".
8. Preserves the original filename.

7. Output Format

Every output file is saved as a NumPy ".npy" file.

Expected output properties:

Shape: 256 × 256
Data type: float32
Value range: [0, 1]
Channels: Grayscale
NaN/Inf: None

For the tested degraded dataset:

Input files:  3200
Output files: 3200
Failed files: 0

All output filenames match their corresponding input filenames.

8. Validation

The generated output directory was validated for:

- Correct number of output files
- Correct filenames
- Correct "256 × 256" resolution
- "float32" data type
- Values within "[0, 1]"
- Absence of NaN and Inf values

Validation result:

Files checked: 3200
Invalid shape/dtype: 0
NaN/Inf files: 0
Input files: 3200
Output files: 3200
Names match: True

9. Hardware

The implementation automatically selects CUDA when a CUDA-enabled PyTorch installation and NVIDIA GPU are available. Otherwise, it falls back to CPU execution.

The trained model weights are loaded locally from:

models/restoration_model.pth

No external API or runtime model download is required.

10. Reproducibility

The repository contains the inference entry point, model definition, trained model weights, dependency specification, and execution instructions required to reproduce the restoration pipeline.

11. Team Details

Team Name: MOSFET

Members:

- RISHAV RAJ AMBASTH
- SONU KUMAR

College: DCRUST MURTHAL