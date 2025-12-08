import os
#hard override of environment to ensure bundled tf.keras is used regardless of shell state.
#remove any legacy flag entirely and set KERAS_BACKEND to tensorflow. This was a persistant issue in my expirimentation.
os.environ.pop("TF_USE_LEGACY_KERAS", None)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import time
import numpy as np
"""Model building & compression pipeline.

We intentionally use the bundled Keras inside tensorflow. Some shells may
export TF_USE_LEGACY_KERAS=1 which makes TensorFlow expect the external
`tf_keras` package and raises an ImportError when tfmot probes keras.
We strip that env var BEFORE importing tensorflow so the lazy loader binds
to the internal keras implementation.
"""

legacy_flag = os.environ.get("TF_USE_LEGACY_KERAS")
if legacy_flag:
    print(f"[INFO] Detected TF_USE_LEGACY_KERAS={legacy_flag!r}; overridden to use bundled keras.")

import tensorflow as tf
#diagnostics: show effective legacy flag and keras binding state after TF import
print("[DEBUG] After TF import: TF_USE_LEGACY_KERAS=", os.environ.get("TF_USE_LEGACY_KERAS"))
print("[DEBUG] tf version:", tf.__version__)
print("[DEBUG] Has tf.keras?", hasattr(tf, "keras"))
#ensure tf.keras and tf.compat.v1.keras resolve to bundled keras before tfmot probes compat paths
try:
    _ = tf.keras.layers.Dense
    print("[DEBUG] tf.keras initialized OK")
except Exception as e:
    print("[DEBUG] tf.keras lazy loader failed:", repr(e))
    from tensorflow import keras as _bundled_keras
    tf.keras = _bundled_keras
    print("[DEBUG] Bound tf.keras to tensorflow.keras explicitly")
try:
    _ = tf.compat.v1.keras.layers.Dense
    print("[DEBUG] tf.compat.v1.keras initialized OK")
except Exception as e:
    print("[DEBUG] tf.compat.v1.keras lazy loader failed:", repr(e))
    from tensorflow import keras as _bundled_keras
    setattr(tf.compat.v1, "keras", _bundled_keras)
    print("[DEBUG] Bound tf.compat.v1.keras to tensorflow.keras explicitly")
try:
    # Ensure tf.compat.v2.keras is also resolvable for tfmot clustering registry
    _ = tf.compat.v2.keras.layers.GRUCell
    print("[DEBUG] tf.compat.v2.keras initialized OK")
except Exception as e:
    print("[DEBUG] tf.compat.v2.keras lazy loader failed:", repr(e))
    setattr(tf.compat.v2, "keras", tf.keras)
    print("[DEBUG] Bound tf.compat.v2.keras to tf.keras explicitly")

#double-sanitize legacy flag prior to importing tfmot (some packages may re-export env)
os.environ.pop("TF_USE_LEGACY_KERAS", None)
print("[DEBUG] Right before tfmot import: TF_USE_LEGACY_KERAS=", os.environ.get("TF_USE_LEGACY_KERAS"))
#import only the keras submodules we need to avoid initializing clustering path early
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from tensorflow_model_optimization.sparsity.keras import PolynomialDecay
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.sparsity.keras import UpdatePruningStep
from tensorflow_model_optimization.quantization.keras import quantize_model
import tensorflow_model_optimization as tfmot  # keep alias for references
import pandas as pd
import shutil
import matplotlib.pyplot as plt  # <-- for graphs

# ------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load and preprocess MNIST
# ------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#normalize to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#add channel dimension -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#simple validation split from the start of the training set
x_val = x_train[:5000]
y_val = y_train[:5000]
x_train = x_train[5000:]
y_train = y_train[5000:]

BATCH_SIZE = 128
EPOCHS_BASELINE = 10
EPOCHS_PRUNE = 10
EPOCHS_QAT = 10
EPOCHS_PRUNED_QAT = 10

# ------------------------------------------------------------
# Model definition (small CNN ~tens of k params)
# ------------------------------------------------------------
def build_baseline_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="mnist_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ------------------------------------------------------------
#utility functions for evaluation
# ------------------------------------------------------------
def get_file_size_kb(path):
    return os.path.getsize(path) / 1024.0

def compute_sparsity(model):
    """Percentage of weights that are exactly zero."""
    total = 0
    zeros = 0
    for w in model.get_weights():
        total += w.size
        zeros += np.sum(w == 0)
    return 100.0 * zeros / float(total)

def measure_keras_latency(model, x_source, num_runs=1000):
    """Average per-sample latency (ms) using Keras model.predict."""
    idx = np.random.choice(len(x_source), size=num_runs, replace=False)
    samples = x_source[idx]
    start = time.time()
    for i in range(num_runs):
        _ = model.predict(samples[i:i+1], verbose=0)
    end = time.time()
    return (end - start) * 1000.0 / num_runs

def evaluate_keras_model(name, model, x_test, y_test, model_path):
    print(f"\nEvaluating {name} (Keras FP32) ...")
    # Use return_dict to avoid mis-ordering and pick the right accuracy key
    results = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    loss = float(results.get("loss", 0.0))
    # Try common accuracy keys
    acc_key = None
    for k in ("accuracy", "sparse_categorical_accuracy"):
        if k in results:
            acc_key = k
            break
    if acc_key is None:
        # Fallback: pick any key that ends with 'accuracy'
        for k in results.keys():
            if k.endswith("accuracy"):
                acc_key = k
                break
    acc_val = float(results.get(acc_key, 0.0)) if acc_key else 0.0

    # Sanity fixup: if loss is very low but accuracy is ~random, recompute manually
    if acc_val < 0.2 and loss < 0.2:
        preds = model.predict(x_test, verbose=0)
        preds_labels = np.argmax(preds, axis=1)
        acc_val = (preds_labels == y_test).mean()
        print(f"[FIXUP] Recomputed accuracy due to mismatch: {acc_val*100.0:.2f}%")

    latency_ms = measure_keras_latency(model, x_test)
    sparsity = compute_sparsity(model)
    size_kb = get_file_size_kb(model_path)
    metrics = {
        "model": name,
        "kind": "Keras",
        "file_path": model_path,
        "size_kb": size_kb,
        "accuracy": acc_val * 100.0,
        "loss": loss,
        "latency_ms": latency_ms,
        "sparsity": sparsity,
    }
    print(metrics)
    return metrics

# ------------------------------------------------------------
# TFLite helpers (PTQ & evaluation)
# ------------------------------------------------------------
def representative_data_gen():
    # Small subset of training data for calibration
    for i in range(500):
        image = x_train[i].astype(np.float32)
        # TFLite expects a batch dimension
        yield [np.expand_dims(image, axis=0)]

def convert_to_tflite_int8(model, tflite_path):
    """Convert a Keras model to fully-int8 TFLite using PTQ."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    #debug printout so it's obvious what quantization config is being used
    try:
        fully_quantize = int(
            (converter.representative_dataset is not None)
            and (tf.lite.OpsSet.TFLITE_BUILTINS_INT8 in converter.target_spec.supported_ops)
            and (converter.inference_input_type == tf.int8)
            and (converter.inference_output_type == tf.int8)
        )
        in_type = getattr(converter.inference_input_type, 'name', str(converter.inference_input_type))
        out_type = getattr(converter.inference_output_type, 'name', str(converter.inference_output_type))
        print(f"fully_quantize: {fully_quantize}, input_inference_type: {in_type}, output_inference_type: {out_type}")
    except Exception as _e:
        print(f"[warn] Unable to print converter quantization config: {_e}")
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return tflite_path

def convert_to_tflite_fp32(keras_model, tflite_path):
    """Convert a Keras model to TFLite FP32 (no quantization)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return tflite_path

def evaluate_tflite_model(name, tflite_path, x_test, y_test,
                          keras_model_for_sparsity=None):
    """
    Run accuracy, loss, latency for an INT8 TFLite model.
    Sparsity is taken from the corresponding Keras model
    (before conversion), if provided.
    """
    print(f"\nEvaluating {name} (TFLite INT8) ...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    #helpful diagnostic so you can see the actual input/output dtypes and quant params
    try:
        print(
            "[INT8 evaluator] input dtype:", getattr(input_details["dtype"], "__name__", str(input_details["dtype"])) ,
            "output dtype:", getattr(output_details["dtype"], "__name__", str(output_details["dtype"]))
        )
        print(
            f"[INT8 evaluator] input quant: scale={input_scale}, zero_point={input_zero_point}; "
            f"output quant: scale={output_scale}, zero_point={output_zero_point}"
        )
    except Exception as _e:
        print(f"[warn] Unable to print INT8 evaluator dtypes/quant params: {_e}")

    correct = 0
    total_loss = 0.0
    total = len(x_test)

    start = time.time()
    for i in range(total):
        img = x_test[i:i+1].astype(np.float32)

        # Quantize input
        if input_scale > 0:
            img_q = img / input_scale + input_zero_point
            img_q = np.clip(np.rint(img_q), -128, 127).astype(np.int8)
        else:
            img_q = img.astype(np.int8)

        interpreter.set_tensor(input_details["index"], img_q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])

        # Dequantize output, treat as logits
        if output_scale > 0:
            logits = (output.astype(np.float32) - output_zero_point) * output_scale
        else:
            logits = output.astype(np.float32)

        probs = tf.nn.softmax(logits, axis=-1).numpy()
        pred = int(np.argmax(probs, axis=1)[0])
        correct += int(pred == int(y_test[i]))
        total_loss += -np.log(float(probs[0, int(y_test[i])]) + 1e-7)
    end = time.time()
    latency_ms = (end - start) * 1000.0 / total

    accuracy = 100.0 * correct / float(total)
    loss = total_loss / float(total)
    size_kb = get_file_size_kb(tflite_path)
    sparsity = (compute_sparsity(keras_model_for_sparsity)
                if keras_model_for_sparsity is not None else None)

    metrics = {
        "model": name,
        "kind": "TFLite INT8",
        "file_path": tflite_path,
        "size_kb": size_kb,
        "accuracy": accuracy,
        "loss": loss,
        "latency_ms": latency_ms,
        "sparsity": sparsity,
    }
    print(metrics)
    return metrics

# ------------------------------------------------------------
# 1) Baseline CNN (FP32)
# ------------------------------------------------------------
all_metrics = []

baseline_model = build_baseline_model()
baseline_model.summary()
total_params = baseline_model.count_params()
print(f"Baseline model parameter count: {total_params} (target ~75k)")

baseline_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS_BASELINE,
    batch_size=BATCH_SIZE,
    verbose=2,
)

baseline_path = os.path.join(MODEL_DIR, "baseline_fp32.h5")
baseline_model.save(baseline_path)

metrics_baseline = evaluate_keras_model(
    "Baseline_FP32", baseline_model, x_test, y_test, baseline_path
)
all_metrics.append(metrics_baseline)

# ------------------------------------------------------------
# 2) PTQ (INT8) of baseline
# ------------------------------------------------------------
ptq_tflite_path = os.path.join(MODEL_DIR, "ptq_int8.tflite")
convert_to_tflite_int8(baseline_model, ptq_tflite_path)

metrics_ptq = evaluate_tflite_model(
    "PTQ_INT8", ptq_tflite_path, x_test, y_test,
    keras_model_for_sparsity=baseline_model,
)
all_metrics.append(metrics_ptq)

def evaluate_tflite_fp32(name, tflite_path, x_test, y_test,
                         keras_model_for_sparsity=None, num_samples=None):
    """
    Run accuracy, loss, latency for a FP32 TFLite model using float inputs.
    Sparsity is taken from the corresponding Keras model (if provided).
    """
    print(f"\nEvaluating {name} (TFLite FP32) ...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Expect float32 model
    if input_details["dtype"].name != "float32":
        print("[WARN] FP32 evaluator: input dtype is not float32; proceeding anyway.")

    total = len(x_test) if num_samples is None else min(num_samples, len(x_test))
    correct = 0
    total_loss = 0.0

    # Optional warm-up to stabilize timing
    for _ in range(10):
        interpreter.set_tensor(input_details["index"], x_test[0:1].astype(np.float32))
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])

    start = time.time()
    for i in range(total):
        img = x_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details["index"], img)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details["index"]).astype(np.float32)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        pred = int(np.argmax(probs, axis=1)[0])
        correct += int(pred == int(y_test[i]))
        total_loss += -np.log(float(probs[0, int(y_test[i])]) + 1e-7)
    end = time.time()
    latency_ms = (end - start) * 1000.0 / total

    accuracy = 100.0 * correct / float(total)
    loss = total_loss / float(total)
    size_kb = get_file_size_kb(tflite_path)
    sparsity = (compute_sparsity(keras_model_for_sparsity)
                if keras_model_for_sparsity is not None else None)

    metrics = {
        "model": name,
        "kind": "TFLite FP32",
        "file_path": tflite_path,
        "size_kb": size_kb,
        "accuracy": accuracy,
        "loss": loss,
        "latency_ms": latency_ms,
        "sparsity": sparsity,
    }
    print(metrics)
    return metrics
# ------------------------------------------------------------
# 3) QAT (Quantization-Aware Training) -> INT8
# ------------------------------------------------------------
quantize_model = tfmot.quantization.keras.quantize_model

#start from trained baseline weights
qat_model = quantize_model(baseline_model)

qat_model.compile(
#also evaluate a TFLite FP32 version of the baseline for apples-to-apples runtime comparison
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

qat_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS_QAT,
    batch_size=BATCH_SIZE,
    verbose=2,
)

qat_path = os.path.join(MODEL_DIR, "qat_fp32.h5")
qat_model.save(qat_path)

qat_tflite_path = os.path.join(MODEL_DIR, "qat_int8.tflite")
convert_to_tflite_int8(qat_model, qat_tflite_path)

metrics_qat = evaluate_tflite_model(
    "QAT_INT8", qat_tflite_path, x_test, y_test,
    keras_model_for_sparsity=qat_model,
)
all_metrics.append(metrics_qat)

# ------------------------------------------------------------
# 4) Pruned model (50% sparsity target, FP32)
# ------------------------------------------------------------
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

num_images = x_train.shape[0]
steps_per_epoch = int(np.ceil(num_images / float(BATCH_SIZE)))
end_step = steps_per_epoch * EPOCHS_PRUNE

pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,   # 50% target
        begin_step=0,
        end_step=end_step,
    )
}

#wrap the trained baseline with pruning
pruned_model = prune_low_magnitude(baseline_model, **pruning_params)

pruned_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
]

pruned_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS_PRUNE,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

#strip pruning wrappers -> sparse FP32 model
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

#strip_pruning can drop the compiled state; ensure the model is compiled before
#saving/evaluating.
pruned_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

pruned_path = os.path.join(MODEL_DIR, "pruned_fp32.h5")
pruned_model.save(pruned_path)

metrics_pruned = evaluate_keras_model(
    "Pruned_FP32", pruned_model, x_test, y_test, pruned_path
)
all_metrics.append(metrics_pruned)

# ------------------------------------------------------------
# 5) Pruned + PTQ (INT8)
# ------------------------------------------------------------
pruned_ptq_tflite_path = os.path.join(MODEL_DIR, "pruned_ptq_int8.tflite")
convert_to_tflite_int8(pruned_model, pruned_ptq_tflite_path)

metrics_pruned_ptq = evaluate_tflite_model(
    "Pruned+PTQ_INT8", pruned_ptq_tflite_path, x_test, y_test,
    keras_model_for_sparsity=pruned_model,
)
all_metrics.append(metrics_pruned_ptq)

# ------------------------------------------------------------
# 6) Pruned + QAT (INT8)
# ------------------------------------------------------------
pruned_qat_model = quantize_model(pruned_model)

pruned_qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

pruned_qat_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS_PRUNED_QAT,
    batch_size=BATCH_SIZE,
    verbose=1,
)

#capture and log per-epoch accuracies for plotting/debugging
try:
    _hist = pruned_qat_model.history  # Keras attaches last History to model when using Model.fit
    _train_acc = _hist.history.get("accuracy") or _hist.history.get("acc")
    _val_acc = _hist.history.get("val_accuracy") or _hist.history.get("val_acc")
    print(f"Pruned+QAT per-epoch train accuracy: {_train_acc}")
    print(f"Pruned+QAT per-epoch val accuracy:   {_val_acc}")
    #persist to disk so external scripts (plot_accuracy.py) can include this series
    import json
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "pruned_qat_accuracy.json"), "w") as _f:
        json.dump({"train_acc": _train_acc, "val_acc": _val_acc}, _f)
except Exception as _e:
    print(f"[warn] Unable to record Pruned+QAT per-epoch accuracies: {_e}")

pruned_qat_path = os.path.join(MODEL_DIR, "pruned_qat_fp32.h5")
pruned_qat_model.save(pruned_qat_path)

pruned_qat_tflite_path = os.path.join(MODEL_DIR, "pruned_qat_int8.tflite")
convert_to_tflite_int8(pruned_qat_model, pruned_qat_tflite_path)

metrics_pruned_qat = evaluate_tflite_model(
    "Pruned+QAT_INT8", pruned_qat_tflite_path, x_test, y_test,
    keras_model_for_sparsity=pruned_qat_model,
)
all_metrics.append(metrics_pruned_qat)

# ------------------------------------------------------------
# Summary + pick best TFLite Micro model
# ------------------------------------------------------------
summary_df = pd.DataFrame(all_metrics)
print("\n===== Summary over all models =====")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

#save a CSV too (optional, nice for later analysis)
summary_csv_path = os.path.join(MODEL_DIR, "model_comparison_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSaved summary CSV to {summary_csv_path}")

# ------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------
def make_bar_chart(df, metric, ylabel, title, filename):
    """
    Simple bar chart: each bar is (model,kind) for a given metric.
    NaN rows for that metric are dropped.
    """
    plot_df = df.copy()
    plot_df = plot_df[~plot_df[metric].isna()]
    if plot_df.empty:
        print(f"Skipping plot for {metric}, all values are NaN.")
        return

    labels = plot_df["model"] + "\n" + plot_df["kind"]
    values = plot_df[metric].values

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def make_size_vs_accuracy_scatter(df, filename):
    """
    Scatter: size_kb vs accuracy, with labels.
    """
    plot_df = df.copy()
    plot_df = plot_df[~plot_df["accuracy"].isna()]
    if plot_df.empty:
        print("Skipping size vs accuracy plot, no data.")
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(plot_df["size_kb"], plot_df["accuracy"])
    for _, row in plot_df.iterrows():
        label = f"{row['model']} ({row['kind']})"
        plt.text(row["size_kb"] * 1.01, row["accuracy"], label, fontsize=7)

    plt.xlabel("Model size (KB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Size vs Accuracy")
    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

# ------------------------------------------------------------
#create visual comparison plots
# ------------------------------------------------------------
make_bar_chart(
    summary_df,
    metric="size_kb",
    ylabel="Size (KB)",
    title="Model Size Comparison",
    filename="model_size_comparison.png",
)

make_bar_chart(
    summary_df,
    metric="accuracy",
    ylabel="Accuracy (%)",
    title="Model Accuracy Comparison",
    filename="model_accuracy_comparison.png",
)

make_bar_chart(
    summary_df,
    metric="latency_ms",
    ylabel="Latency (ms/sample)",
    title="Model Latency Comparison",
    filename="model_latency_comparison.png",
)

make_bar_chart(
    summary_df,
    metric="sparsity",
    ylabel="Sparsity (% zeros)",
    title="Model Sparsity Comparison",
    filename="model_sparsity_comparison.png",
)

make_size_vs_accuracy_scatter(
    summary_df,
    filename="size_vs_accuracy.png",
)

# ------------------------------------------------------------
#pick best INT8 model by accuracy and export for TFLite Micro
# ------------------------------------------------------------
int8_models = [m for m in all_metrics if "INT8" in m["kind"]]
if int8_models:
    best = max(int8_models, key=lambda d: d["accuracy"])
    best_export_path = os.path.join(MODEL_DIR, "best_tflm_model.tflite")
    shutil.copy(best["file_path"], best_export_path)
    print(f"\nBest INT8 model for TFLite Micro: {best['model']}")
    print(f"Exported to: {best_export_path}")
    # Optional: export as a C array for inclusion in a TFLite Micro project
    def export_tflite_micro_array(tflite_path, dest_header):
        with open(tflite_path, 'rb') as f:
            data = f.read()
        # Emit as unsigned char array
        # Keeping formatting simple; for very large models you could chunk lines.
        byte_values = ', '.join(str(b) for b in data)
        header = (
            '#ifndef BEST_MODEL_TFLITE_H\n'
            '#define BEST_MODEL_TFLITE_H\n'
            'const unsigned char g_best_model_tflite[] = { ' + byte_values + ' };\n'
            f'const int g_best_model_tflite_len = {len(data)};\n'
            '#endif\n'
        )
        dest_path = os.path.join(MODEL_DIR, dest_header)
        with open(dest_path, 'w') as f:
            f.write(header)
        print(f"Exported C header: {dest_path} (array length = {len(data)})")

    export_tflite_micro_array(best_export_path, 'best_tflm_model.h')
else:
    print("No INT8 models were created.")
