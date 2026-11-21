# 最後訓練及正式測試：直接將成格貼入 Colab 執行
from pathlib import Path
import csv
import gc
import hashlib
import json
import math
import shutil
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision

PROJECT = Path("/content/drive/MyDrive/DIT5411")
SOURCE = PROJECT / "images_64"
ROOT = PROJECT / "final_results"
CACHE = Path("/content/dit5411_cache")
LOCAL = Path("/content/images_64")
N_CLASSES, BATCH, VIEWS, SEED = 13065, 128, 200, 42
SPECS = [("ANN", "ann", 3), ("Simple CNN", "simple_cnn", 3),
         ("Deep CNN", "deep_cnn", 4)]


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def copy_saved(source, destination):
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, partial)
    assert partial.stat().st_size == Path(source).stat().st_size
    partial.replace(destination)


def read_manifest(path):
    labels, splits = [], []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            assert row["split"] in ("train", "test")
            labels.append(int(row["label"]))
            splits.append(row["split"] == "train")
    labels = np.asarray(labels, dtype=np.int32)
    is_train = np.asarray(splits, dtype=bool)
    assert len(labels) == 684677
    assert is_train.sum() == 521860
    assert (~is_train).sum() == 162817
    assert labels.min() == 0 and labels.max() == N_CLASSES - 1
    return labels, is_train


def prepare_training(digest, manifest_y, is_train):
    CACHE.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    xp, yp = CACHE / "train_images.npy", CACHE / "train_labels.npy"
    ready = CACHE / "ready.json"
    key = {"manifest_sha256": digest, "count": 521860, "size": 64,
           "format": "uint8_grayscale_v1"}
    if not (ready.exists() and xp.exists() and yp.exists()
            and json.loads(ready.read_text()) == key):
        paths = sorted(SOURCE.glob("train_*.npz"))
        assert len(paths) == 64, "需要 64 個訓練分片"
        x = np.lib.format.open_memmap(xp, mode="w+", dtype=np.uint8,
                                     shape=(521860, 64, 64, 1))
        y = np.lib.format.open_memmap(yp, mode="w+", dtype=np.int32,
                                     shape=(521860,))
        expected_ids = np.flatnonzero(is_train)
        offset = 0
        for i, path in enumerate(paths):
            local = LOCAL / path.name
            if not local.exists() or local.stat().st_size != path.stat().st_size:
                copy_saved(path, local)
            with np.load(local, allow_pickle=False) as z:
                n = len(z["labels"])
                assert z["images"].shape == (n, 64, 64, 1)
                assert z["images"].dtype == np.uint8
                assert np.array_equal(z["source_ids"], expected_ids[offset:offset+n])
                assert np.array_equal(z["labels"], manifest_y[expected_ids[offset:offset+n]])
                x[offset:offset+n], y[offset:offset+n] = z["images"], z["labels"]
                offset += n
            print(f"準備訓練資料：{i+1}/64", flush=True)
        assert offset == 521860
        x.flush()
        y.flush()
        del x, y
        save_json(ready, key)
    x, y = np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")
    assert x.shape == (521860, 64, 64, 1) and x.dtype == np.uint8
    assert np.array_equal(y, manifest_y[is_train])
    return x, y


def class_groups(labels):
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=N_CLASSES)
    assert (counts > 0).all() and (counts <= VIEWS).all()
    return np.split(order, np.cumsum(counts)[:-1])


def sample_epoch(groups, epoch):
    rng = np.random.default_rng(SEED + 1000 + epoch)
    blocks = []
    for group in groups:
        q, r = divmod(VIEWS, len(group))
        blocks.append(np.concatenate([np.tile(group, q), rng.permutation(group)[:r]]))
    plan = np.concatenate(blocks).astype(np.int32)
    rng.shuffle(plan)
    return plan


def make_model(kind):
    parts = [layers.Input(shape=(64, 64, 1)),
             layers.Rescaling(-1/255, offset=1, dtype="float32"),
             layers.RandomRotation(0.025, fill_mode="constant", fill_value=0, seed=SEED),
             layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0, seed=SEED+1),
             layers.RandomZoom(0.1, fill_mode="constant", fill_value=0, seed=SEED+2)]
    if kind != "ann":
        filters = [32, 64] if kind == "simple_cnn" else [32, 64, 128]
        for width in filters:
            parts.extend([layers.Conv2D(width, 3, padding="same", kernel_initializer="he_normal"),
                          layers.LeakyReLU(negative_slope=0.1), layers.MaxPooling2D()])
    parts.extend([layers.Flatten(),
                  layers.Dense(256 if kind == "ann" else 128, kernel_initializer="he_normal"),
                  layers.LeakyReLU(negative_slope=0.1), layers.Dropout(0.2),
                  layers.Dense(N_CLASSES, activation="softmax", dtype="float32")])
    model = tf.keras.Sequential(parts)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    return model


def train_dataset(x, y, plan):
    def generate():
        for start in range(0, len(plan), BATCH):
            idx = plan[start:start+BATCH]
            yield np.asarray(x[idx]), np.asarray(y[idx], dtype=np.int32)
    ds = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((None, 64, 64, 1), tf.uint8), tf.TensorSpec((None,), tf.int32)))
    return ds.apply(tf.data.experimental.assert_cardinality(math.ceil(len(plan)/BATCH))).prefetch(2)


def train_one(name, kind, target_epochs, x, y, groups, digest):
    run = ROOT / kind
    run.mkdir(parents=True, exist_ok=True)
    config = {"stage": "final_refit_all_training_originals_v1", "model": kind,
              "target_epochs": target_epochs, "manifest_sha256": digest,
              "training_originals": len(y), "classes": N_CLASSES, "size": 64,
              "views_per_class_per_epoch": VIEWS, "batch_size": BATCH,
              "seed": SEED, "tensorflow": tf.__version__, "precision": "mixed_float16",
              "input": "1_minus_pixel_div_255", "activation": "LeakyReLU_0.1",
              "hidden_units": 256 if kind == "ann" else 128,
              "conv_filters": [] if kind == "ann" else ([32,64] if kind == "simple_cnn" else [32,64,128]),
              "dropout": 0.2, "initializer": "he_normal_hidden_glorot_output",
              "augmentation": {"rotation": 0.025, "translation": 0.05, "zoom": 0.1,
                               "fill_mode": "constant", "fill_value": 0},
              "optimizer": "Adam", "learning_rate": 0.0003, "clipnorm": 1.0}
    cp = run / "config.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == config, "實驗設定改變，請停止檢查"
    else:
        save_json(cp, config)
    records = []
    for epoch in range(1, target_epochs+1):
        record_path = run / f"epoch_{epoch:03d}.json"
        model_path = run / f"epoch_{epoch:03d}.keras"
        if not (record_path.exists() and model_path.exists()):
            break
        record = json.loads(record_path.read_text())
        assert record["epoch"] == epoch and record["manifest_sha256"] == digest
        records.append(record)
    if len(records) == target_epochs:
        print(f"{name} 已完成 {target_epochs} 輪，跳過訓練。", flush=True)
        return
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(SEED)
    if records:
        model = tf.keras.models.load_model(run / records[-1]["model_file"])
        print(f"{name} 恢復第 {len(records)} 輪 checkpoint", flush=True)
    else:
        model = make_model(kind)
        print(f"{name} 由全新權重開始，使用全部原始訓練圖片。", flush=True)
    model.summary()
    with (run / "model_summary.txt").open("w") as f:
        model.summary(print_fn=lambda line, **kw: f.write(line+"\n"))
    for epoch in range(len(records), target_epochs):
        plan = sample_epoch(groups, epoch)
        assert np.all(np.bincount(y[plan], minlength=N_CLASSES) == VIEWS)
        started = time.perf_counter()
        history = model.fit(train_dataset(x,y,plan), initial_epoch=epoch, epochs=epoch+1,
                            verbose=1, callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        metrics = {k: float(v[-1]) for k,v in history.history.items()}
        assert metrics and all(np.isfinite(v) for v in metrics.values()), "訓練數值異常"
        elapsed = time.perf_counter()-started
        filename = f"epoch_{epoch+1:03d}.keras"
        temp = CACHE / f"final_{kind}.keras"
        model.save(temp)
        copy_saved(temp, run / filename)
        temp.unlink()
        record = {"epoch": epoch+1, "model_file": filename, "manifest_sha256": digest,
                  "train_seconds": elapsed, "model_params": int(model.count_params()), **metrics}
        save_json(run / f"epoch_{epoch+1:03d}.json", record)
        records.append(record)
        with (run / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"{name} 第 {epoch+1}/{target_epochs} 輪已保存，{elapsed/60:.1f} 分鐘。", flush=True)
        del plan
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def prepare_test(manifest_y, is_train):
    paths = sorted(SOURCE.glob("test_*.npz"))
    assert len(paths) == 20, "需要 20 個測試分片"
    expected_ids = np.flatnonzero(~is_train)
    offset, local_paths = 0, []
    for i,path in enumerate(paths):
        local = LOCAL / path.name
        if not local.exists() or local.stat().st_size != path.stat().st_size:
            copy_saved(path,local)
        with np.load(local, allow_pickle=False) as z:
            n = len(z["labels"])
            assert z["images"].shape == (n,64,64,1) and z["images"].dtype == np.uint8
            assert np.array_equal(z["source_ids"], expected_ids[offset:offset+n])
            assert np.array_equal(z["labels"],manifest_y[expected_ids[offset:offset+n]])
            offset += n
        local_paths.append(local)
        print(f"檢查測試分片：{i+1}/20", flush=True)
    assert offset == 162817
    return local_paths


def test_dataset(paths):
    def generate():
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                images, labels = z["images"], z["labels"]
            for start in range(0,len(labels),BATCH):
                yield images[start:start+BATCH], labels[start:start+BATCH].astype(np.int32)
    ds = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((None,64,64,1),tf.uint8),tf.TensorSpec((None,),tf.int32)))
    # Let the finite generator determine the batch count across all test shards.
    return ds.prefetch(2)


def main():
    assert tf.config.list_physical_devices("GPU"), "先選擇 T4 GPU"
    assert (PROJECT / "split_manifest.csv").is_file(), "先掛載 Google Drive"
    ROOT.mkdir(parents=True, exist_ok=True)
    mixed_precision.set_global_policy("mixed_float16")
    digest = hashlib.sha256((PROJECT/"split_manifest.csv").read_bytes()).hexdigest()
    assert json.loads((SOURCE/"config.json").read_text())["manifest_sha256"] == digest
    manifest_y, is_train = read_manifest(PROJECT/"split_manifest.csv")
    x,y = prepare_training(digest,manifest_y,is_train)
    groups = class_groups(y)
    # 完成全部訓練後，才開啟測試資料。
    for name,kind,epochs in SPECS:
        train_one(name,kind,epochs,x,y,groups,digest)
    del x,y,groups
    gc.collect()
    paths = prepare_test(manifest_y,is_train)
    results = []
    for name,kind,epochs in SPECS:
        run = ROOT/kind
        checkpoint = run/f"epoch_{epochs:03d}.keras"
        record = json.loads((run/f"epoch_{epochs:03d}.json").read_text())
        fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        output = run/"test_metrics.json"
        if output.exists():
            row = json.loads(output.read_text())
            assert row["manifest_sha256"] == digest and row["checkpoint_sha256"] == fingerprint
            assert row["test_images"] == 162817
            print(f"{name} 已有相同 checkpoint 嘅測試結果，直接重用。")
        else:
            tf.keras.backend.clear_session()
            gc.collect()
            model = tf.keras.models.load_model(checkpoint,compile=False)
            model.compile(loss="sparse_categorical_crossentropy", metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,name="top5_accuracy")],
                jit_compile=False)
            started = time.perf_counter()
            metrics = model.evaluate(test_dataset(paths),return_dict=True,verbose=1)
            assert all(np.isfinite(v) for v in metrics.values())
            times = [json.loads((run/f"epoch_{i:03d}.json").read_text())["train_seconds"]
                     for i in range(1,epochs+1)]
            row = {"model": name, "epochs": epochs, "model_params": record["model_params"],
                   "train_originals": 521860, "views_per_class_per_epoch": VIEWS,
                   "test_images": 162817, "test_accuracy": float(metrics["accuracy"]),
                   "test_top5_accuracy": float(metrics["top5_accuracy"]),
                   "test_loss": float(metrics["loss"]), "train_minutes": sum(times)/60,
                   "test_seconds": time.perf_counter()-started,
                   "manifest_sha256": digest, "checkpoint_sha256": fingerprint,
                   "checkpoint": str(checkpoint)}
            save_json(output,row)
            del model
        results.append(row)
        print(f"{name} TEST accuracy: {row['test_accuracy']:.2%}", flush=True)
    save_json(ROOT/"test_comparison.json",results)
    fields = ["model","epochs","model_params","test_images","test_accuracy",
              "test_top5_accuracy","test_loss","train_minutes","checkpoint"]
    with (ROOT/"test_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    text = ["## Final test results\n",
            "Architectures and epoch budgets were fixed using validation before test evaluation. "
            "Each model was initialized afresh and refitted on all 521,860 original training images. "
            "Each epoch used 200 augmented inputs per class across 13,065 classes. "
            "The held-out test set contained 162,817 images and received no augmentation.\n",
            "| Model | Epochs | Model parameters | Test top-1 accuracy | Test top-5 accuracy | Test loss |",
            "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        text.append(f"| {row['model']} | {row['epochs']} | {row['model_params']:,} | "
                    f"{row['test_accuracy']:.2%} | {row['test_top5_accuracy']:.2%} | {row['test_loss']:.4f} |")
    text.append("\nThe original split used the first 40 images in numeric filename order for training "
                "and the remainder for testing. For 43 classes with fewer than 41 images, an approximately "
                "80/20 split was used as a documented exception. This differs from the assignment's "
                "literal first-40 rule. Validation-stage results and final test results are reported separately. "
                "Training inputs were augmented online using rotation, translation and zoom.\n")
    (ROOT/"README_test_results.md").write_text("\n".join(text),encoding="utf-8")
    print("\n===== 最終 TEST 結果 =====")
    for row in results:
        print(f"{row['model']}: {row['test_accuracy']:.2%} (top-5: {row['test_top5_accuracy']:.2%})")
    print("已保存：",ROOT)
    print("CSV/JSON 嘅 accuracy 以 0–1 儲存；README 表格已轉為百分比。")
    print("模型設定已固定，之後唔好用 test 結果反覆調校模型。")


main()
