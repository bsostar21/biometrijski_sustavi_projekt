import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
THRESHOLD = 0.5
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = train_ds.class_names
print("Classes:", class_names)

def normalize(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.image.rgb_to_grayscale(x)
    x = tf.image.grayscale_to_rgb(x)
    return x, y

train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(normalize, num_parallel_calls=AUTOTUNE)
test_ds  = test_ds.map(normalize, num_parallel_calls=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

def augment(x, y):
    x = data_augmentation(x, training=True)
    return x, y

train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)


train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

acc = tf.keras.metrics.BinaryAccuracy(name="acc")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[acc],
)

print("\n===== TRAIN (FREEZE) =====")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[acc],
)

print("\n===== TRAIN (FINE-TUNE) =====")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
)

print("\n===== TEST EVALUATION =====")
model.evaluate(test_ds, verbose=2)

y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds], axis=0)
y_pred_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_pred_prob >= THRESHOLD).astype(int)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion matrix (TEST):")
print("Actual \\ Predicted")
print(f"{'':10s}{class_names[0]:>10s}{class_names[1]:>10s}")
for i, row in enumerate(cm):
    print(f"{class_names[i]:10s}{row[0]:10d}{row[1]:10d}")

print("\nClassification report (TEST):\n",
      classification_report(y_true, y_pred, target_names=class_names))


