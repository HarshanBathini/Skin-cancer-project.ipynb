"""
Skin lesion classification (HAM10000, 7 classes) using MobileNetV2 transfer learning.

Usage:
    python train.py --data_dir data --epochs_head 8 --epochs_finetune 10

Expected data folder (download from Kaggle "Skin Cancer MNIST: HAM10000"):
    data/HAM10000_metadata.csv
    data/HAM10000_images_part_1/*.jpg
    data/HAM10000_images_part_2/*.jpg
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FULL_NAMES = {
    "akiec": "Actinic keratoses / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus (common mole)",
    "vasc": "Vascular lesion",
}
IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE


def load_dataframe(data_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")
    paths = {p.stem: str(p) for p in data_dir.rglob("*.jpg")}
    meta["path"] = meta["image_id"].map(paths)
    meta = meta.dropna(subset=["path"]).reset_index(drop=True)
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(CLASSES)})
    print(f"Loaded {len(meta)} images")
    print(meta["dx"].value_counts())
    return meta


def split_data(df: pd.DataFrame, seed=42):
    """Split by lesion_id so the same lesion never appears in train AND test (avoids data leakage)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df["lesion_id"]))
    train, temp = df.iloc[train_idx], df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp, groups=temp["lesion_id"]))
    return train, temp.iloc[val_idx], temp.iloc[test_idx]


def make_ds(df: pd.DataFrame, batch_size: int, training: bool):
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    if training:
        ds = ds.shuffle(len(df), seed=42)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_model():
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)  # only active during training
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def plot_history(histories, out_path):
    acc = sum([h.history["accuracy"] for h in histories], [])
    val_acc = sum([h.history["val_accuracy"] for h in histories], [])
    loss = sum([h.history["loss"] for h in histories], [])
    val_loss = sum([h.history["val_loss"] for h in histories], [])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(acc, label="train"); ax[0].plot(val_acc, label="val"); ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(loss, label="train"); ax[1].plot(val_loss, label="val"); ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_confusion(cm, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main(args):
    out = Path(args.out_dir); out.mkdir(exist_ok=True, parents=True)
    df = load_dataframe(Path(args.data_dir))
    train_df, val_df, test_df = split_data(df)
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    train_ds = make_ds(train_df, args.batch_size, True)
    val_ds = make_ds(val_df, args.batch_size, False)
    test_ds = make_ds(test_df, args.batch_size, False)

    # Dataset is heavily imbalanced (~67% are 'nv'), so weight rare classes higher
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=train_df["label"].values)
    class_weight = dict(enumerate(weights))
    print("Class weights:", class_weight)

    model, base = build_model()
    ckpt = str(out / "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ]

    # Phase 1: train only the new classification head
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                   class_weight=class_weight, callbacks=callbacks)

    # Phase 2: fine-tune the top layers of MobileNetV2 with a small learning rate
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
                   class_weight=class_weight, callbacks=callbacks)

    plot_history([h1, h2], out / "training_curves.png")

    # Final evaluation on the untouched test set
    y_true = test_df["label"].values
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=False, zero_division=0)
    print(report)
    (out / "classification_report.txt").write_text(report)
    plot_confusion(confusion_matrix(y_true, y_pred), out / "confusion_matrix.png")

    model.save(out / "skin_cancer_model.keras")
    (out / "classes.json").write_text(json.dumps({"classes": CLASSES, "names": FULL_NAMES}, indent=2))
    print(f"Done. Everything saved in {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_head", type=int, default=8)
    p.add_argument("--epochs_finetune", type=int, default=10)
    main(p.parse_args())
