#!/usr/bin/env python
import argparse
import json
import os
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras
from keras import ops
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Reshape, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def smooth_labels(labels, factor=0.01):
    labels = labels.copy()
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_s = ops.shape(source)[0]
    n_s = 64 if n_s is None else n_s
    n_t = ops.shape(target)[0]
    n_t = 64 if n_t is None else n_t
    n_samples = n_s + n_t
    total = ops.concatenate([source, target], axis=0)
    total0 = ops.expand_dims(total, axis=0)
    total1 = ops.expand_dims(total, axis=1)
    L2_distance = ops.sum(((total0 - total1) ** 2), axis=2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = ops.sum(L2_distance) / ops.cast((n_samples ** 2 - n_samples), dtype="float32")
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [ops.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    n_s = ops.shape(source)[0]
    n_s = 64 if n_s is None else n_s
    n_t = ops.shape(target)[0]
    n_t = 64 if n_t is None else n_t
    XX = ops.sum(kernels[:n_s, :n_s]) / ops.cast((n_s ** 2), dtype="float32")
    YY = ops.sum(kernels[-n_t:, -n_t:]) / ops.cast((n_t ** 2), dtype="float32")
    XY = ops.sum(kernels[:n_s, -n_t:]) / ops.cast((n_s * n_t), dtype="float32")
    YX = ops.sum(kernels[-n_t:, :n_s]) / ops.cast((n_s * n_t), dtype="float32")
    return XX + YY - XY - YX


def build_model(img_rows=8, img_cols=9, num_chan=4, num_classes=2):
    img_size = (img_rows, img_cols, num_chan)
    x1 = Conv2D(64, 5, activation="relu", padding="same", name="conv1")
    x2 = Conv2D(128, 4, activation="relu", padding="same", name="conv2")
    x3 = Conv2D(256, 4, activation="relu", padding="same", name="conv3")
    x4 = Conv2D(64, 1, activation="relu", padding="same", name="conv4")
    x5 = MaxPooling2D(2, 2)
    x6 = Flatten()
    x7 = Dense(512, activation="relu")
    x8 = Reshape((1, 512))

    inputs = [Input(shape=img_size) for _ in range(6)]
    feats = [x8(x7(x6(x5(x4(x3(x2(x1(inp)))))))) for inp in inputs]

    # branch1
    out_all1_1 = Concatenate(axis=1)([feats[0], feats[1], feats[2]])
    out_all2_1 = Concatenate(axis=1)([feats[3], feats[4], feats[5]])
    distance1 = MMD(out_all1_1, out_all2_1)
    out_all3_1 = Concatenate(axis=1)([feats[0], feats[1], feats[2], feats[3], feats[4], feats[5]])
    lstm1 = LSTM(128, name="lstm1")(out_all3_1)
    out1 = Dense(num_classes, activation="softmax", name="out")(lstm1)
    model1 = Model(inputs=inputs, outputs=out1)
    model1.add_loss(distance1)

    # branch2
    out_all1_2 = Concatenate(axis=1)([feats[0], feats[1], feats[5]])
    out_all2_2 = Concatenate(axis=1)([feats[2], feats[3], feats[4]])
    distance2 = MMD(out_all1_2, out_all2_2)
    out_all3_2 = Concatenate(axis=1)([feats[0], feats[1], feats[5], feats[2], feats[3], feats[4]])
    lstm2 = LSTM(128, name="lstm1")(out_all3_2)
    out2 = Dense(num_classes, activation="softmax", name="out")(lstm2)
    model2 = Model(inputs=inputs, outputs=out2)
    model2.add_loss(distance2)

    # branch3
    out_all1_3 = Concatenate(axis=1)([feats[5], feats[4], feats[3]])
    out_all2_3 = Concatenate(axis=1)([feats[2], feats[1], feats[0]])
    distance3 = MMD(out_all1_3, out_all2_3)
    out_all3_3 = Concatenate(axis=1)([feats[5], feats[4], feats[3], feats[2], feats[1], feats[0]])
    lstm3 = LSTM(128, name="lstm1")(out_all3_3)
    out3 = Dense(num_classes, activation="softmax", name="out")(lstm3)
    model3 = Model(inputs=inputs, outputs=out3)
    model3.add_loss(distance3)

    for m in (model1, model2, model3):
        m.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    return model1, model2, model3


def run_subject(
    mat_path,
    flag,
    epochs=50,
    batch_size=128,
    seed=7,
    save_models=False,
    model_dir=None,
    subject_id="00",
):
    t = 6
    file = sio.loadmat(mat_path)
    data = file["data"]
    y_v = to_categorical(file["valence_labels"][0], 2)
    y_a = to_categorical(file["arousal_labels"][0], 2)
    x = data.transpose([0, 2, 3, 1]).reshape((-1, t, 8, 9, 4))
    y_v2 = np.vstack([y_v[j * t] for j in range(len(y_v) // t)])
    y_a2 = np.vstack([y_a[j * t] for j in range(len(y_a) // t)])
    y = y_v2 if flag == "v" else y_a2

    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []
    for fold_idx, (train, test) in enumerate(kfold.split(x, y.argmax(1)), start=1):
        K.clear_session()
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) >= 2:
            strategy = tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.get_strategy()

        with strategy.scope():
            model1, model2, model3 = build_model()
        x_train, y_train = x[train], y[train]
        x_test, y_test = x[test], y[test]
        x_test_in = [x_test[:, i] for i in range(6)]
        early_stop = EarlyStopping(monitor="val_accuracy", min_delta=0, patience=5, mode="max", verbose=0, restore_best_weights=True)
        callbacks1 = []
        callbacks2 = []
        callbacks3 = []
        if save_models and model_dir is not None:
            fold_dir = os.path.join(model_dir, f"subject_{subject_id}", f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            callbacks1.append(
                ModelCheckpoint(
                    filepath=os.path.join(fold_dir, "best_model1.keras"),
                    monitor="val_accuracy",
                    mode="max",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0,
                )
            )
            callbacks2.append(
                ModelCheckpoint(
                    filepath=os.path.join(fold_dir, "best_model2.keras"),
                    monitor="val_accuracy",
                    mode="max",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0,
                )
            )
            callbacks3.append(
                ModelCheckpoint(
                    filepath=os.path.join(fold_dir, "best_model3.keras"),
                    monitor="val_accuracy",
                    mode="max",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0,
                )
            )

        for _ in range(2):
            model1.fit([x_train[:, i] for i in range(6)], smooth_labels(y_train), callbacks=callbacks1, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_test_in, y_test))
            model2.fit([x_train[:, i] for i in range(6)], smooth_labels(y_train), callbacks=callbacks2, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_test_in, y_test))
            model3.fit([x_train[:, i] for i in range(6)], smooth_labels(y_train), callbacks=callbacks3, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_test_in, y_test))
        model1.fit([x_train[:, i] for i in range(6)], smooth_labels(y_train), callbacks=[early_stop, *callbacks1], epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_test_in, y_test))
        model2.fit([x_train[:, i] for i in range(6)], smooth_labels(y_train), callbacks=[early_stop, *callbacks2], epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_test_in, y_test))
        model3.fit([x_train[:, i] for i in range(6)], smooth_labels(y_train), callbacks=[early_stop, *callbacks3], epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_test_in, y_test))

        p1 = np.argmax(model1.predict(x_test_in, verbose=0), axis=1)
        p2 = np.argmax(model2.predict(x_test_in, verbose=0), axis=1)
        p3 = np.argmax(model3.predict(x_test_in, verbose=0), axis=1)
        true = np.argmax(y_test, axis=1)
        vote = np.zeros_like(p1)
        for i in range(len(vote)):
            if p1[i] == p2[i]:
                vote[i] = p1[i]
            elif p1[i] == p3[i]:
                vote[i] = p1[i]
            elif p2[i] == p3[i]:
                vote[i] = p2[i]
            else:
                vote[i] = p1[i]
        score = float((vote == true).mean() * 100.0)
        fold_scores.append(score)
        print(f"    fold {fold_idx}: {score:.2f}%")
    return fold_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="/root/inpainting-work/interactive/TDMNN/deap_mat")
    ap.add_argument("--flag", choices=["v", "a"], default="a")
    ap.add_argument("--subjects", default="all", help="all or comma list like 1,2,3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out-json", default="/root/inpainting-work/interactive/TDMNN/deap_results_original_style.json")
    ap.add_argument("--save-models", action="store_true", help="Save best model checkpoints for each subject/fold/branch.")
    ap.add_argument("--model-dir", default="/root/inpainting-work/interactive/TDMNN/models_binary")
    args = ap.parse_args()

    if args.subjects == "all":
        subs = [f"{i:02d}" for i in range(1, 33)]
    else:
        subs = [f"{int(s):02d}" for s in args.subjects.split(",") if s.strip()]

    all_means = []
    all_stds = []
    details = {}
    for sid in subs:
        mat_path = os.path.join(args.dataset_dir, f"DE_s{sid}.mat")
        if not os.path.exists(mat_path):
            print(f"skip s{sid}: file not found")
            continue
        print(f"\nprocessing s{sid} ...")
        t0 = time.time()
        fold_scores = run_subject(
            mat_path,
            args.flag,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_models=args.save_models,
            model_dir=args.model_dir,
            subject_id=sid,
        )
        m = float(np.mean(fold_scores))
        s = float(np.std(fold_scores))
        all_means.append(m)
        all_stds.append(s)
        details[sid] = {"fold_scores": fold_scores, "mean": m, "std": s, "seconds": time.time() - t0}
        print(f"  subject mean={m:.2f}% std={s:.2f}%")

    out = {
        "task": "valence" if args.flag == "v" else "arousal",
        "subjects": details,
        "acc_all": all_means,
        "std_all": all_stds,
        # Paper-aligned summary (common reporting style):
        # ACC = mean over subject means, STD = std over subject means.
        "acc_avg": float(np.mean(all_means)) if all_means else None,
        "std_subject": float(np.std(all_means)) if all_means else None,
        # Kept for compatibility with earlier outputs:
        # mean of within-subject 5-fold std values.
        "std_avg": float(np.mean(all_stds)) if all_stds else None,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nAcc_all:", all_means)
    print("Std_all:", all_stds)
    print("Acc_avg:", out["acc_avg"])
    print("Std_subject:", out["std_subject"])
    print("Std_avg:", out["std_avg"])
    print("saved:", args.out_json)


if __name__ == "__main__":
    main()
