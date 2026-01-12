# TFLite Micro Hello World (Colab → `.tflite` → C++ Executable)

## Overview
This project demonstrates an end-to-end workflow for running a TensorFlow Lite Micro model in C++. A small model is trained in Google Colab, exported as a `.tflite` file, converted into a C array, and then linked into a custom-built `hello_world` executable using the TensorFlow Lite Micro static library.

## Goals
- Train and export a TensorFlow model as a **TFLite** file (`.tflite`) in Colab.
- Build the TensorFlow Lite Micro runtime library (`libtensorflow-microlite.a`) locally.
- Embed the exported `.tflite` model into C/C++ source code using `xxd -i`.
- Compile and run a standalone C++ executable (`hello_world`) outside of Bazel, using a custom Makefile.
- Verify successful inference by running the program and capturing console output.

## Workflow Summary

### 1) Train and export a model in Colab
A Jupyter/Colab notebook was used to train the Hello World model and export it as a `.tflite` file (e.g., `hello_world_float.tflite`). The provided reference notebook from the TensorFlow Lite Micro repository was used as guidance.

### 2) Clone and build TensorFlow Lite Micro
The `tflite-micro` repository was cloned and the default Makefile target was used to build the static library:

- Output library:
  - `gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a`

This library provides the embedded inference runtime and operator implementations used by the C++ executable.

### 3) Convert the `.tflite` model to C source
The exported `.tflite` model was converted to a C array using `xxd -i`, producing a `.cc` file containing a byte array representation of the model. For readability and integration, the array symbol was renamed to a `g_*_model_data` style name and a matching header file was created to expose it (e.g., `hello_world_float_model_data.h`).

### 4) Implement `hello_world.cc` and build outside Bazel
A custom `hello_world.cc` and Makefile were created (based on code snippets from the official TFLite Micro Hello World example). The executable links against `libtensorflow-microlite.a`, loads the embedded model data, allocates a tensor arena, registers required operators, and runs inference while printing results to the terminal.

### 5) Run and verify output
The executable was built and run locally:
- `make`
- `./hello_world`

Successful console output verified that model inference was executing correctly.

## Deliverables
- Colab notebook used to train/export the model (or a note referencing the provided notebook).
- Exported `.tflite` model file committed to the repo.
- Model converted to C source (`*_model_data.cc`) and header (`*_model_data.h`).
- `hello_world.cc` implementation and Makefile.
- Screenshot of `hello_world` output.

## Key Takeaway
This project validates the complete embedded ML deployment pipeline: training in Python, exporting a TFLite model, embedding it into C/C++, linking against TensorFlow Lite Micro, and running inference in a standalone executable using a custom build system.
