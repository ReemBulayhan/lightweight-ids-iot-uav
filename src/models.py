"""SkyShield-IDS model definitions (TensorFlow/Keras 3).

Models A-E of the ablation study share a 1D-CNN backbone and differ only in Efficient Channel
Attention (ECA) and a skip branch. Model E is SkyShield-IDS; build_student() gives SkyShield-IDS-S
(every width divided by 4). kd_loss() is the distillation loss alpha*T^2*KL + (1-alpha)*CE with
T = 3 and alpha = 0.4, where y_true packs the label and the teacher probabilities.
The same code is used inside the notebooks.
"""
import keras
from keras import layers

N_CLASSES = 8
CLASS_NAMES = ['BENIGN', 'Botnet', 'BruteForce', 'DDoS', 'DoS', 'PortScan', 'Spoofing', 'WebAttack']
T_KD, ALPHA, WIDTH_DIV = 3.0, 0.4, 4


def eca(x):
    """Efficient Channel Attention with kernel size 3 and no bias (adds exactly 3 parameters)."""
    c = x.shape[-1]
    w = layers.GlobalAveragePooling1D()(x)
    w = layers.Reshape((c, 1))(w)
    w = layers.Conv1D(1, 3, padding='same', use_bias=False, activation='sigmoid')(w)
    w = layers.Reshape((1, c))(w)
    return layers.Multiply()([x, w])


def build(variant, n_features, n_classes=N_CLASSES):
    """A: CNN+GAP | B: +ECA | C: +skip | D: +ECA +skip | E: +2xECA +skip (SkyShield-IDS)."""
    inp = keras.Input((n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    if variant == 'E':
        x = eca(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    if variant in ('B', 'D', 'E'):
        x = eca(x)
    z = layers.GlobalAveragePooling1D()(x)
    if variant in ('C', 'D', 'E'):
        s = layers.Dense(128, activation='relu')(layers.Flatten()(inp))
        z = layers.Concatenate()([z, s])
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(n_classes, activation='softmax')(z)
    return keras.Model(inp, out, name=f'Model_{variant}')


def build_student(n_features, div=WIDTH_DIV, n_classes=N_CLASSES):
    """SkyShield-IDS-S: Model E with every width divided by div."""
    inp = keras.Input((n_features, 1))
    x = layers.Conv1D(64 // div, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = eca(x)
    x = layers.Conv1D(128 // div, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = eca(x)
    z = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(128 // div, activation='relu')(layers.Flatten()(inp))
    z = layers.Concatenate()([z, s])
    z = layers.Dense(64 // div, activation='relu')(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(n_classes, activation='softmax')(z)
    return keras.Model(inp, out, name=f'Student_E_div{div}')


def _kd(y_true, y_pred, alpha):
    """y_true = [label, teacher probabilities (8)]; y_pred = student probabilities."""
    lab = keras.ops.cast(y_true[:, 0], 'int32'); pt = y_true[:, 1:]
    ce = keras.losses.sparse_categorical_crossentropy(lab, y_pred)
    if alpha == 0:
        return ce
    ls = keras.ops.log(keras.ops.clip(y_pred, 1e-7, 1.0)); lt = keras.ops.log(keras.ops.clip(pt, 1e-7, 1.0))
    ps, pt_t = keras.ops.softmax(ls / T_KD, axis=-1), keras.ops.softmax(lt / T_KD, axis=-1)
    kl = keras.ops.sum(pt_t * (keras.ops.log(keras.ops.clip(pt_t, 1e-7, 1.0)) - keras.ops.log(keras.ops.clip(ps, 1e-7, 1.0))), axis=-1)
    return alpha * T_KD ** 2 * kl + (1 - alpha) * ce


@keras.saving.register_keras_serializable(package='kd')
def kd_loss(y_true, y_pred):
    return _kd(y_true, y_pred, ALPHA)


@keras.saving.register_keras_serializable(package='kd')
def ce_only_loss(y_true, y_pred):
    return _kd(y_true, y_pred, 0.0)


def class_weights(y, mode='standard'):
    """w_c = N / (K * N_c) over the K classes present; mode='sqrt' gives the softened weights (square root)."""
    import numpy as np
    c = np.bincount(y, minlength=N_CLASSES); present = np.flatnonzero(c)
    w = {int(k): float(len(y) / (len(present) * c[k])) for k in present}
    return {k: v ** 0.5 for k, v in w.items()} if mode == 'sqrt' else w
