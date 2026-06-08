"""
Cognitive Machine Learning for OT Security: A Kantian-Inspired Hybrid Framework
for Industrial Anomaly Detection Code
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef)
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# SABITLER & HİPERPARAMETRELER
# ==============================================================================
BENIGN_FILE = ' '
ATTACK_FILE = ' '

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

DAE_EPOCHS     = 60
DAE_BATCH      = 1024
DAE_LR         = 1e-3
DAE_PATIENCE   = 10
DAE_BOTTLENECK = 32

SEQ_LEN      = 10
LSTM_EPOCHS  = 60
LSTM_BATCH   = 512
LSTM_PATIENCE = 8
LSTM_L2      = 3e-3
LSTM_UNITS_1 = 24
LSTM_UNITS_2 = 12
NOISE_STD    = 0.10


# ==============================================================================
# KATMAN 1 — SENSİBİLİTY
# ==============================================================================
def sensibility_layer(benign_path, attack_path):
    print("=" * 60)
    print("KATMAN 1 — SENSİBİLİTY")
    print("=" * 60)

    df_b = pd.read_csv(benign_path, low_memory=False)
    df_a = pd.read_csv(attack_path, low_memory=False)
    df_b.columns = df_b.columns.str.strip().str.lower()
    df_a.columns = df_a.columns.str.strip().str.lower()

    identity_cols = {
        'src_ip', 'dst_ip', 'src_port', 'dst_port',
        'source ip', 'source port', 'destination ip', 'destination port',
        'flow id', 'timestamp', 'stream',
        'src_mac', 'dst_mac', 'eth_src_oui', 'eth_dst_oui',
        'device_mac', 'protocol'
    }
    numeric_b   = df_b.select_dtypes(include=[np.number]).columns.tolist()
    numeric_a   = df_a.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = sorted(
        set(c for c in numeric_b if c not in identity_cols) &
        set(c for c in numeric_a if c not in identity_cols)
    )
    if not common_cols:
        raise ValueError("Ortak özellik bulunamadı.")

    df_b = df_b[common_cols].copy(); df_b['label'] = 0
    df_a = df_a[common_cols].copy(); df_a['label'] = 1

    def chrono_split(df_cls):
        n  = len(df_cls)
        t1 = int(n * TRAIN_RATIO)
        t2 = int(n * (TRAIN_RATIO + VAL_RATIO))
        return df_cls.iloc[:t1].copy(), df_cls.iloc[t1:t2].copy(), df_cls.iloc[t2:].copy()

    b_train, b_val, b_test = chrono_split(df_b)
    a_train, a_val, a_test = chrono_split(df_a)

    def concat_2d(d0, d1):
        c = pd.concat([d0, d1], ignore_index=True)
        return c.drop(columns=['label']).values.astype(np.float64), c['label'].values

    X_tr_raw, y_tr = concat_2d(b_train, a_train)
    X_vl_raw, y_vl = concat_2d(b_val,   a_val)
    X_te_raw, y_te = concat_2d(b_test,  a_test)

    for arr in [X_tr_raw, X_vl_raw, X_te_raw]:
        arr[arr ==  np.inf] = np.nan
        arr[arr == -np.inf] = np.nan

    train_median = np.nanmedian(X_tr_raw, axis=0)
    X_tr_raw = np.where(np.isnan(X_tr_raw), train_median, X_tr_raw)
    X_vl_raw = np.where(np.isnan(X_vl_raw), train_median, X_vl_raw)
    X_te_raw = np.where(np.isnan(X_te_raw), train_median, X_te_raw)

    scaler = StandardScaler()
    scaler.fit(X_tr_raw)
    X_train_2d = scaler.transform(X_tr_raw).astype(np.float32)
    X_val_2d   = scaler.transform(X_vl_raw).astype(np.float32)
    X_test_2d  = scaler.transform(X_te_raw).astype(np.float32)

    print(f"\n  Toplam satır   : {len(df_b) + len(df_a)}")
    print(f"  Özellik sayısı : {len(common_cols)}")
    print(f"  Train          : {len(X_train_2d)} örnek")
    print(f"  Val            : {len(X_val_2d)} örnek")
    print(f"  Test           : {len(X_test_2d)} örnek")
    print(f"\n  → Sensibility tamamlandı")

    return (X_train_2d, X_val_2d, X_test_2d,
            y_tr, y_vl, y_te,
            b_train, b_val, b_test,
            a_train, a_val, a_test,
            scaler, train_median, common_cols)


# ==============================================================================
# KATMAN 2 — İMAGINATION (DAE)
# ==============================================================================
def build_imagination_model(input_dim):
    h1 = max(64, input_dim)
    h2 = max(32, input_dim // 2)

    ae_in   = Input(shape=(input_dim,), name='sensibility_output')
    x       = Dense(h1, activation='elu', name='imagination_enc1')(ae_in)
    x       = BatchNormalization()(x)
    x       = Dense(h2, activation='elu', name='imagination_enc2')(x)
    x       = BatchNormalization()(x)
    schema  = Dense(DAE_BOTTLENECK, activation='elu', name='kantian_schema')(x)
    x       = Dense(h2, activation='elu', name='imagination_dec1')(schema)
    x       = BatchNormalization()(x)
    x       = Dense(h1, activation='elu', name='imagination_dec2')(x)
    recon   = Dense(input_dim, activation='linear', name='reconstruction')(x)

    imagination  = Model(ae_in, recon,  name='Imagination_DAE')
    schema_model = Model(ae_in, schema, name='Kantian_Schema_Encoder')
    imagination.compile(optimizer=Adam(DAE_LR), loss='mae')
    return imagination, schema_model


def train_imagination(X_train_benign, X_val_benign):
    print("\n" + "=" * 60)
    print("IMAGINATION")
    print("=" * 60)
    print(f" benign ({len(X_train_benign)})")
    print(f"bottleneck) : {DAE_BOTTLENECK}")

    print(f"  Denoising: (std={NOISE_STD}), "
          f"clean")

    imagination, schema_model = build_imagination_model(X_train_benign.shape[1])

    
    _rng = np.random.default_rng(42)
    X_train_noisy = (X_train_benign
                     + _rng.normal(0, NOISE_STD, X_train_benign.shape)
                     ).astype(np.float32)
    X_val_noisy   = (X_val_benign
                     + _rng.normal(0, NOISE_STD, X_val_benign.shape)
                     ).astype(np.float32)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=DAE_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_imagination.keras', save_best_only=True,
                        monitor='val_loss', verbose=0)
    ]
    history = imagination.fit(
        X_train_noisy, X_train_benign,                 
        validation_data=(X_val_noisy, X_val_benign),    
        epochs=DAE_EPOCHS, batch_size=DAE_BATCH,
        callbacks=callbacks, verbose=1
    )
    return imagination, schema_model, history


def optimize_imagination_threshold(imagination, X_val_2d, y_val):
    print("\n  Optimization")
    recon       = imagination.predict(X_val_2d, batch_size=DAE_BATCH, verbose=0)
    anomaly_v   = np.mean(np.abs(X_val_2d - recon), axis=1)
    benign_anom = anomaly_v[y_val == 0]

    best_f1, best_thr = -1.0, float(np.percentile(benign_anom, 95))
    for pct in np.linspace(50, 99.9, 300):
        thr = np.percentile(benign_anom, pct)
        p   = (anomaly_v > thr).astype(int)
        if p.sum() == 0 or p.sum() == len(p): continue
        f1 = f1_score(y_val, p, average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr

    acc = np.mean((anomaly_v > best_thr).astype(int) == y_val) * 100
    print(f"  → Anomaly threshold: {best_thr:.6f}  Val F1=%{best_f1*100:.2f}  Val Acc=%{acc:.2f}")
    return best_thr


def build_schema_sequences(schema_model, imagination, df_cls0, df_cls1,
                            scaler, train_median, noisy=False, seed=42):
    rng = np.random.RandomState(seed)

    def process(df_cls):
        X = df_cls.drop(columns=['label']).values.astype(np.float64)
        X[X ==  np.inf] = np.nan; X[X == -np.inf] = np.nan
        X = np.where(np.isnan(X), train_median, X)
        X_clean = scaler.transform(X).astype(np.float32)
        X_proc  = X_clean.copy()
        if noisy:
            X_proc = X_proc + rng.normal(0, NOISE_STD, X_proc.shape).astype(np.float32)

        schema = schema_model.predict(X_proc,  batch_size=DAE_BATCH, verbose=0)
        recon  = imagination.predict(X_clean,  batch_size=DAE_BATCH, verbose=0)
        anomaly_full = np.mean(np.abs(X_clean - recon), axis=1)

        lbl = int(df_cls['label'].iloc[0])
        n   = len(schema)
        if n < SEQ_LEN:
            return (np.empty((0, SEQ_LEN, schema.shape[1]), np.float32),
                    np.empty(0, np.int64),
                    np.empty(0, np.float32))

        seqs    = np.zeros((n - SEQ_LEN + 1, SEQ_LEN, schema.shape[1]), np.float32)
        labels  = np.full(n - SEQ_LEN + 1, lbl, np.int64)
        anomaly = anomaly_full[SEQ_LEN - 1:]
        for i in range(len(seqs)):
            seqs[i] = schema[i : i + SEQ_LEN]
        return seqs, labels, anomaly

    S0, L0, A0 = process(df_cls0)
    S1, L1, A1 = process(df_cls1)
    seqs    = np.concatenate([S0, S1])
    labs    = np.concatenate([L0, L1])
    anomaly = np.concatenate([A0, A1])
    perm    = np.random.RandomState(seed).permutation(len(seqs))
    return seqs[perm], labs[perm], anomaly[perm]


# ==============================================================================
# KATMAN 3 — UNDERSTANDING (LSTM)
# ==============================================================================
def build_understanding_model(schema_dim):
    reg = l2(LSTM_L2)
    inp = Input(shape=(SEQ_LEN, schema_dim), name='imagination_schemas')
    x   = LSTM(LSTM_UNITS_1, return_sequences=True,
               kernel_regularizer=reg, name='understanding_1')(inp)
    x   = Dropout(0.5)(x)
    x   = LSTM(LSTM_UNITS_2, return_sequences=False,
               kernel_regularizer=reg, name='understanding_2')(x)
    x   = Dropout(0.4)(x)
    x   = Dense(12, activation='relu',
                kernel_regularizer=reg, name='understanding_dense')(x)
    out = Dense(1, activation='sigmoid', name='understanding_score')(x)

    model = Model(inp, out, name='Understanding_LSTM')
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auroc')]
    )
    return model


def train_understanding(X_train_seq, y_train, X_val_seq, y_val):
    print("\n" + "=" * 60)
    print("UNDERSTANDING")
    print("=" * 60)
    print(f"  Imagination shema {X_train_seq.shape}")

    cw = dict(enumerate(
        class_weight.compute_class_weight('balanced',
                                          classes=np.unique(y_train),
                                          y=y_train)
    ))
    model     = build_understanding_model(X_train_seq.shape[2])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=LSTM_PATIENCE,
                      min_delta=0.001, restore_best_weights=True,
                      mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_understanding.keras', save_best_only=True,
                        monitor='val_loss', mode='min', verbose=0)
    ]
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
        class_weight=cw, callbacks=callbacks, verbose=1
    )
    return model, history


# ==============================================================================
# LAYER 4 — REASON / COGNİTİVE
# ==============================================================================
def optimize_reason(understanding_model, X_val_seq, y_val, anomaly_val):
    print("\n" + "=" * 60)
    print("Layer 4 — REASON / COGNİTİVE (Kantian Reasoner)")
    print("=" * 60)

    understanding_scores = understanding_model.predict(
        X_val_seq, batch_size=LSTM_BATCH, verbose=0
    ).flatten()
    anomaly_norm = ((anomaly_val - anomaly_val.min()) /
                    (anomaly_val.max() - anomaly_val.min() + 1e-9))

    best_f1, best_w, best_thr = -1.0, (0.3, 0.7), 0.5
    for w_imag in np.arange(0.1, 0.7, 0.05):
        w_under = round(1.0 - w_imag, 2)
        reason_score = w_under * understanding_scores + w_imag * anomaly_norm
        for thr in np.linspace(0.01, 0.99, 500):
            preds = (reason_score >= thr).astype(int)
            if preds.sum() == 0 or preds.sum() == len(preds): continue
            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_w, best_thr = f1, (round(w_imag, 2), w_under), thr

    w_imag, w_under = best_w
    print(f"  → w_imagination={w_imag:.2f}  w_understanding={w_under:.2f}  "
          f"threshold={best_thr:.4f}  Val F1=%{best_f1*100:.2f}")
    return w_imag, w_under, best_thr, best_f1


def reason_predict(understanding_model, X_seq, anomaly,
                   w_imag, w_under, threshold):
    understanding_scores = understanding_model.predict(
        X_seq, batch_size=LSTM_BATCH, verbose=0
    ).flatten()
    anomaly_norm = ((anomaly - anomaly.min()) /
                    (anomaly.max() - anomaly.min() + 1e-9))
    reason_score = w_under * understanding_scores + w_imag * anomaly_norm
    judgement    = (reason_score >= threshold).astype(int)
    return judgement, reason_score, understanding_scores, anomaly


# ==============================================================================
# 5. ABLATION TABLE
# ==============================================================================
def compute_metrics(y_test, y_val,
                    final_judgement, reason_score,
                    understanding_scores, anomaly_test,
                    anomaly_threshold, w_imag, w_under, reason_threshold,
                    understanding_model, X_val_seq):
    print("\n" + "=" * 60)
    print("5. ABLATION TABLE")
    print("=" * 60)

    imagination_preds   = (anomaly_test > anomaly_threshold).astype(int)
    understanding_preds = (understanding_scores > 0.5).astype(int)

    imag_f1   = f1_score(y_test, imagination_preds,   average='macro', zero_division=0) * 100
    under_f1  = f1_score(y_test, understanding_preds, average='macro', zero_division=0) * 100
    reason_f1 = f1_score(y_test, final_judgement,     average='macro', zero_division=0) * 100

    imag_acc   = np.mean(imagination_preds   == y_test) * 100
    under_acc  = np.mean(understanding_preds == y_test) * 100
    reason_acc = np.mean(final_judgement     == y_test) * 100

    try:
        roc_auc = roc_auc_score(y_test, reason_score) * 100
    except Exception:
        roc_auc = float('nan')
    pr_auc    = average_precision_score(y_test, reason_score) * 100
    known_rec = (final_judgement[y_test==1]==1).mean()*100 if (y_test==1).sum()>0 else 0

    val_scores = understanding_model.predict(
        X_val_seq, batch_size=LSTM_BATCH, verbose=0
    ).flatten()
    val_f1 = f1_score(y_val, (val_scores > 0.5).astype(int),
                      average='macro', zero_division=0) * 100
    gap = abs(val_f1 - under_f1)

    print(f"\n{'─'*58}")
    print(f"  {'Kantian Katman':<35} {'Acc':>8}  {'F1 Macro':>9}")
    print(f"{'─'*58}")
    print(f"  {'Imagination  (DAE Anomaly)':<35} %{imag_acc:>6.2f}  %{imag_f1:>8.2f}")
    print(f"  {'Understanding(LSTM on Schemas)':<35} %{under_acc:>6.2f}  %{under_f1:>8.2f}")
    print(f"  {'★ Reason     (Kantian Reasoner)':<35} %{reason_acc:>6.2f}  %{reason_f1:>8.2f}")
    print(f"{'─'*58}")
    print(f"  Reason katkısı (vs Understanding)  : +%{reason_f1 - under_f1:.2f}")
    print(f"  Ağırlıklar: w_imagination={w_imag:.2f} / w_understanding={w_under:.2f}")
    print(f"{'─'*58}")
    print(f"\n  Known Attack Recognition : %{known_rec:.2f}")
    print(f"  ROC-AUC                  : %{roc_auc:.2f}")
    print(f"  PR-AUC                   : %{pr_auc:.2f}")
    print(f"  [GAP] Val Understanding  : %{val_f1:.2f}")
    print(f"  [GAP] Test Understanding : %{under_f1:.2f}")
    print(f"  [GAP] Fark               : %{gap:.2f}  "
          f"{'⚠ Overfitting' if gap > 5 else '✓ Tutarlı'}")
    print(f"\nDetaylı Classification Report (Reason/Cognitive):")
    print(classification_report(y_test, final_judgement, labels=[0, 1],
                                target_names=['Benign', 'Attack'], zero_division=0))

    return {
        'imag_f1':   imag_f1,   'imag_acc':   imag_acc,
        'under_f1':  under_f1,  'under_acc':  under_acc,
        'reason_f1': reason_f1, 'reason_acc': reason_acc,
        'roc_auc': roc_auc, 'pr_auc': pr_auc,
        'known_rec': known_rec,
        'val_f1': val_f1, 'test_f1': under_f1, 'gap': gap,
        'w_imag': w_imag, 'w_under': w_under
    }


# ==============================================================================
# 6. PRINTEABLE
# ==============================================================================
def plot_results(imag_history, under_history,
                 anomaly_test, y_test, final_judgement,
                 anomaly_threshold, metrics):

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(
        'SANNA v5.0 — Kantian Cognitive IDS · Sonuç Paneli\n'
        'Sensibility → Imagination → Understanding → Reason',
        fontsize=14, fontweight='bold', y=1.01
    )

    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(imag_history.history['loss'],     label='Train', color='royalblue')
    ax1.plot(imag_history.history['val_loss'], label='Val',   color='deepskyblue', ls='--')
    ax1.set_title('Imagination (DAE) — Rekonstrüksiyon Kaybı')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MAE'); ax1.legend(); ax1.grid(alpha=.3)

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(under_history.history['accuracy'],     label='Train', color='seagreen')
    ax2.plot(under_history.history['val_accuracy'], label='Val',   color='mediumseagreen', ls='--')
    ax2.set_title('Understanding (LSTM) — Doğruluk')
    ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(alpha=.3)

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(under_history.history['loss'],     label='Train', color='tomato')
    ax3.plot(under_history.history['val_loss'], label='Val',   color='salmon', ls='--')
    ax3.set_title('Understanding (LSTM) — Kayıp')
    ax3.set_xlabel('Epoch'); ax3.legend(); ax3.grid(alpha=.3)

    ax4 = fig.add_subplot(3, 3, 4)
    sns.histplot(anomaly_test[y_test==0], bins=60, color='steelblue',
                 alpha=.6, label='Benign', stat='density', ax=ax4)
    sns.histplot(anomaly_test[y_test==1], bins=60, color='crimson',
                 alpha=.6, label='Attack', stat='density', ax=ax4)
    ax4.axvline(anomaly_threshold, color='black', ls='--', lw=2,
                label=f'Eşik={anomaly_threshold:.4f}')
    ax4.set_title('Imagination — Anomali Sinyali Dağılımı')
    ax4.set_xlabel('Rekonstrüksiyon Hatası (MAE)'); ax4.legend(); ax4.grid(alpha=.3)

    ax5 = fig.add_subplot(3, 3, 5)
    sns.heatmap(confusion_matrix(y_test, final_judgement),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'], ax=ax5)
    ax5.set_title('Reason — Final Yargı (Confusion Matrix)')
    ax5.set_xlabel('Tahmin'); ax5.set_ylabel('Gerçek')

    ax6 = fig.add_subplot(3, 3, 6)
    names  = ['Imagination\n(DAE)', 'Understanding\n(LSTM)', '★ Reason\n(Cognitive)']
    scores = [metrics['imag_f1'], metrics['under_f1'], metrics['reason_f1']]
    colors = ['royalblue', 'seagreen', 'crimson']
    bars   = ax6.bar(names, scores, color=colors, width=.55)
    ax6.set_ylim(max(0, min(scores) - 15), 105)
    ax6.set_ylabel('F1 Macro (%)'); ax6.set_title('Kantian Ablation Study')
    for bar, val in zip(bars, scores):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5,
                 f'%{val:.1f}', ha='center', va='bottom', fontweight='bold')
    ax6.grid(axis='y', alpha=.3)

    ax7 = fig.add_subplot(3, 3, 7)
    delta = metrics['reason_f1'] - metrics['under_f1']
    ax7.bar(['Understanding\n(LSTM)', '★ Reason\n(Cognitive)'],
            [metrics['under_f1'], metrics['reason_f1']],
            color=['seagreen', 'crimson'], width=0.4)
    ax7.set_ylim(max(0, metrics['under_f1'] - 5), 105)
    ax7.set_ylabel('F1 Macro (%)'); ax7.set_title(f'Reason Katkısı: +%{delta:.2f}')
    for v, x in zip([metrics['under_f1'], metrics['reason_f1']], [0, 1]):
        ax7.text(x, v+0.3, f'%{v:.2f}', ha='center', fontweight='bold')
    ax7.grid(axis='y', alpha=.3)

    ax8 = fig.add_subplot(3, 3, 8)
    gv  = [metrics['val_f1'], metrics['test_f1']]
    b8  = ax8.bar(['Val F1', 'Test F1'], gv,
                  color=['steelblue', 'darkorange'], width=0.4)
    ax8.set_ylim(max(0, min(gv)-10), 105)
    ax8.set_ylabel('F1 Macro (%)'); ax8.set_title('Val / Test F1 Gap')
    for bar, val in zip(b8, gv):
        ax8.text(bar.get_x()+bar.get_width()/2, val+.3,
                 f'%{val:.2f}', ha='center', fontweight='bold')
    ax8.set_xlabel(f"Δ=%{metrics['gap']:.2f}  "
                   f"{'⚠ Overfitting' if metrics['gap']>5 else '✓ Tutarlı'}")
    ax8.grid(axis='y', alpha=.3)

    ax9 = fig.add_subplot(3, 3, 9)
    ak  = next((k for k in under_history.history if 'auroc' in k and 'val' not in k), None)
    vk  = next((k for k in under_history.history if 'val' in k and 'auroc' in k), None)
    if ak and vk:
        ax9.plot(under_history.history[ak], label='Train', color='purple')
        ax9.plot(under_history.history[vk], label='Val',   color='mediumpurple', ls='--')
        ax9.set_title('Understanding — AUROC')
        ax9.set_xlabel('Epoch'); ax9.legend(); ax9.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig('sanna_v5.0_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nGörsel kaydedildi: sanna_v5.0_results.png")



def compute_extended_metrics(y_true, y_pred, y_score=None, name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    acc = (TP + TN) / (TP + TN + FP + FN) * 100
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr = FP / (FP + TN) * 100 if (FP + TN) > 0 else 0
    fnr = FN / (FN + TP) * 100 if (FN + TP) > 0 else 0
    kar = TP / (TP + FN) * 100 if (FN + TP) > 0 else 0

    roc = roc_auc_score(y_true, y_score) * 100 if y_score is not None else 0
    pr  = average_precision_score(y_true, y_score) * 100 if y_score is not None else 0

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : %{acc:.2f}")
    print(f"  F1 Macro  : %{f1:.2f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  FPR       : %{fpr:.2f}")
    print(f"  FNR       : %{fnr:.2f}")
    print(f"  KAR       : %{kar:.2f}")
    print(f"  AUC-ROC   : %{roc:.2f}")
    print(f"  AUC-PR    : %{pr:.2f}")
    print(f"  CM → TN={TN}, FP={FP}, FN={FN}, TP={TP}")
    print(f"{'─'*55}")

    return {
        'name': name, 'acc': acc, 'f1': f1, 'mcc': mcc,
        'fpr': fpr, 'fnr': fnr, 'kar': kar, 'roc': roc, 'pr': pr,
        'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)
    }


# ==============================================================================
# ★ BASELINE MODEL (RF, XGB, IF, OC-SVM)
# ==============================================================================
def run_baselines(X_train_2d, X_test_2d, y_tr, y_te):
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.svm import OneClassSVM
    from xgboost import XGBClassifier

    print("\n" + "=" * 60)
    print("★ BASELINE MODELLERİ KARŞILAŞTIRMASI")
    print("=" * 60)

    X_train_b = X_train_2d[y_tr == 0]
    results   = []

    print("\n  1. Random Forest eğitiliyor...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_2d, y_tr)
    results.append(compute_extended_metrics(
        y_te, rf.predict(X_test_2d), rf.predict_proba(X_test_2d)[:, 1], "Random Forest"))

    print("\n  2. XGBoost eğitiliyor...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss',
                        n_jobs=-1, use_label_encoder=False)
    xgb.fit(X_train_2d, y_tr)
    results.append(compute_extended_metrics(
        y_te, xgb.predict(X_test_2d), xgb.predict_proba(X_test_2d)[:, 1], "XGBoost"))

    print("\n  3. Isolation Forest eğitiliyor (benign-only)...")
    attack_ratio = float((y_tr == 1).sum() / len(y_tr))
    contamination = min(attack_ratio, 0.499)
    print(f"     Attack oranı: %{attack_ratio*100:.1f} → contamination={contamination:.3f}")
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X_train_b)
    results.append(compute_extended_metrics(
        y_te, (iso.predict(X_test_2d) == -1).astype(int),
        -iso.score_samples(X_test_2d), "Isolation Forest"))

    print("\n  4. One-Class SVM eğitiliyor (benign-only, örnekleme)...")
    max_samples = min(5000, len(X_train_b))
    idx = np.random.RandomState(42).choice(len(X_train_b), max_samples, replace=False)
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    ocsvm.fit(X_train_b[idx])
    results.append(compute_extended_metrics(
        y_te, (ocsvm.predict(X_test_2d) == -1).astype(int),
        -ocsvm.decision_function(X_test_2d), "One-Class SVM"))

    return results


# ==============================================================================
# ★ LSTM-AE BASELINE
# ==============================================================================
def run_lstm_ae_baseline(X_train_2d, X_val_2d, X_test_2d, y_tr, y_vl, y_te):
    print("\n" + "=" * 60)
    print("★ LSTM-AE BASELINE")
    print("=" * 60)

    def make_sequences(X, y, seq_len=SEQ_LEN):
        seqs, labs = [], []
        for cls in [0, 1]:
            idx   = np.where(y == cls)[0]
            X_cls = X[idx]; n = len(X_cls)
            for i in range(n - seq_len + 1):
                seqs.append(X_cls[i:i+seq_len]); labs.append(cls)
        return np.array(seqs, np.float32), np.array(labs)

    X_tr_seq, y_tr_seq = make_sequences(X_train_2d, y_tr)
    X_vl_seq, y_vl_seq = make_sequences(X_val_2d,   y_vl)
    X_te_seq, y_te_seq = make_sequences(X_test_2d,  y_te)
    X_tr_b = X_tr_seq[y_tr_seq == 0]

    inp = Input(shape=(SEQ_LEN, X_train_2d.shape[1]))
    x   = LSTM(64, return_sequences=False, name='lstm_ae_enc')(inp)
    x   = tf.keras.layers.RepeatVector(SEQ_LEN)(x)
    x   = LSTM(64, return_sequences=True,  name='lstm_ae_dec')(x)
    out = tf.keras.layers.TimeDistributed(Dense(X_train_2d.shape[1]), name='lstm_ae_out')(x)
    lstm_ae = Model(inp, out, name='LSTM_AE_Baseline')
    lstm_ae.compile(optimizer=Adam(1e-3), loss='mae')

    lstm_ae.fit(X_tr_b, X_tr_b,
        validation_data=(X_vl_seq[y_vl_seq==0], X_vl_seq[y_vl_seq==0]),
        epochs=30, batch_size=512,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)

    recon_val = lstm_ae.predict(X_vl_seq, batch_size=512, verbose=0)
    err_val   = np.mean(np.abs(X_vl_seq - recon_val), axis=(1, 2))
    benign_e  = err_val[y_vl_seq == 0]

    best_f1, best_thr = -1, np.percentile(benign_e, 95)
    for pct in np.linspace(50, 99.9, 200):
        thr = np.percentile(benign_e, pct)
        f1  = f1_score(y_vl_seq, (err_val > thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr

    recon_te = lstm_ae.predict(X_te_seq, batch_size=512, verbose=0)
    err_te   = np.mean(np.abs(X_te_seq - recon_te), axis=(1, 2))
    return compute_extended_metrics(y_te_seq, (err_te > best_thr).astype(int), err_te, "LSTM-AE Baseline")


# ==============================================================================
# ★ CNN-LSTM BASELINE
# ==============================================================================
def run_cnn_lstm_baseline(X_train_2d, X_val_2d, X_test_2d, y_tr, y_vl, y_te):
    print("\n" + "=" * 60)
    print("★ CNN-LSTM BASELINE")
    print("=" * 60)

    def make_sequences(X, y, seq_len=SEQ_LEN):
        seqs, labs = [], []
        for cls in [0, 1]:
            idx   = np.where(y == cls)[0]
            X_cls = X[idx]; n = len(X_cls)
            for i in range(n - seq_len + 1):
                seqs.append(X_cls[i:i+seq_len]); labs.append(cls)
        return np.array(seqs, np.float32), np.array(labs)

    X_tr_seq, y_tr_seq = make_sequences(X_train_2d, y_tr)
    X_vl_seq, y_vl_seq = make_sequences(X_val_2d,   y_vl)
    X_te_seq, y_te_seq = make_sequences(X_test_2d,  y_te)

    feat_dim = X_train_2d.shape[1]
    inp = Input(shape=(SEQ_LEN, feat_dim), name='cnn_lstm_input')
    x   = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', name='cnn_lstm_conv1')(inp)
    x   = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same', name='cnn_lstm_conv2')(x)
    x   = LSTM(32, return_sequences=False, name='cnn_lstm_lstm')(x)
    x   = Dropout(0.3)(x)
    x   = Dense(16, activation='relu', name='cnn_lstm_dense')(x)
    out = Dense(1, activation='sigmoid', name='cnn_lstm_out')(x)

    cnn_lstm = Model(inp, out, name='CNN_LSTM_Baseline')
    cnn_lstm.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    cw = dict(enumerate(class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_tr_seq), y=y_tr_seq)))

    cnn_lstm.fit(X_tr_seq, y_tr_seq, validation_data=(X_vl_seq, y_vl_seq),
        epochs=30, batch_size=512, class_weight=cw,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)

    proba_te = cnn_lstm.predict(X_te_seq, batch_size=512, verbose=0).flatten()
    return compute_extended_metrics(y_te_seq, (proba_te >= 0.5).astype(int), proba_te, "CNN-LSTM Baseline")


# ==============================================================================
# ★ TRANSFORMER-ENCODER BASELINE
# ==============================================================================
def run_transformer_baseline(X_train_2d, X_val_2d, X_test_2d, y_tr, y_vl, y_te):
    
    print("\n" + "=" * 60)
    print("★ TRANSFORMER-ENCODER BASELINE")
    print("=" * 60)

    def make_sequences(X, y, seq_len=SEQ_LEN):
        seqs, labs = [], []
        for cls in [0, 1]:
            idx   = np.where(y == cls)[0]
            X_cls = X[idx]
            n     = len(X_cls)
            for i in range(n - seq_len + 1):
                seqs.append(X_cls[i:i+seq_len])
                labs.append(cls)
        return np.array(seqs, np.float32), np.array(labs)

    X_tr_seq, y_tr_seq = make_sequences(X_train_2d, y_tr)
    X_vl_seq, y_vl_seq = make_sequences(X_val_2d,   y_vl)
    X_te_seq, y_te_seq = make_sequences(X_test_2d,  y_te)

    feat_dim = X_train_2d.shape[1]
    d_model, n_heads = 64, 4

    inp = Input(shape=(SEQ_LEN, feat_dim), name='transformer_input')
    x   = Dense(d_model, name='transformer_proj')(inp)

    positions = tf.range(start=0, limit=SEQ_LEN, delta=1)
    pos_emb   = tf.keras.layers.Embedding(
        input_dim=SEQ_LEN,
        output_dim=d_model,
        name='pos_embedding'
    )(positions)
    x = x + pos_emb

    for blk in range(2):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            name=f'mha_{blk}'
        )(x, x)
        attn = Dropout(0.2, name=f'attn_dropout_{blk}')(attn)
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f'ln1_{blk}'
        )(x + attn)

        ff = Dense(128, activation='relu', name=f'ff1_{blk}')(x)
        ff = Dense(d_model, name=f'ff2_{blk}')(ff)
        ff = Dropout(0.2, name=f'ff_dropout_{blk}')(ff)
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f'ln2_{blk}'
        )(x + ff)

    x   = tf.keras.layers.GlobalAveragePooling1D(name='transformer_pool')(x)
    x   = Dropout(0.3, name='transformer_dropout')(x)
    x   = Dense(32, activation='relu', name='transformer_dense')(x)
    out = Dense(1, activation='sigmoid', name='transformer_out')(x)

    transformer = Model(inp, out, name='Transformer_IDS_Baseline')
    transformer.compile(
        optimizer=Adam(3e-4),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )

    cw = dict(enumerate(class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_tr_seq), y=y_tr_seq)))

    callbacks = [
        EarlyStopping(
            monitor='val_pr_auc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    transformer.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_vl_seq, y_vl_seq),
        epochs=60,
        batch_size=256,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    
    val_proba = transformer.predict(X_vl_seq, batch_size=512, verbose=0).flatten()
    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(0.05, 0.95, 181):
        pred_val = (val_proba >= thr).astype(int)
        f1 = f1_score(y_vl_seq, pred_val, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    proba_te = transformer.predict(X_te_seq, batch_size=512, verbose=0).flatten()
    print(f"  Transformer: {transformer.count_params():,}")
    print(f"  Transformer validation-optimal threshold: {best_thr:.3f}  Val Macro-F1=%{best_f1*100:.2f}")

    return compute_extended_metrics(
        y_te_seq,
        (proba_te >= best_thr).astype(int),
        proba_te,
        "Transformer-IDS Baseline"
    )

# ==============================================================================
# ★ KitNET (Kitsune) BASELINE
# ==============================================================================
class _KitNET_AE:
    
    def __init__(self, n_in, hidden_ratio=0.75, lr=0.1, seed=42):
        self.n_in = n_in
        self.n_hid = max(1, int(np.ceil(n_in * hidden_ratio)))
        rng = np.random.RandomState(seed)
        a = 1.0 / max(1, n_in)
        self.W  = rng.uniform(-a, a, (self.n_in, self.n_hid))
        self.hb = np.zeros(self.n_hid)
        self.vb = np.zeros(self.n_in)
        self.lr = lr
        self.nmax = np.full(n_in, -np.inf)
        self.nmin = np.full(n_in,  np.inf)

    def _sig(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _norm(self, x):
        self.nmax = np.maximum(self.nmax, x)
        self.nmin = np.minimum(self.nmin, x)
        return (x - self.nmin) / (self.nmax - self.nmin + 1e-13)

    def train(self, x):
        x = self._norm(x)
        h = self._sig(x @ self.W + self.hb)
        z = self._sig(h @ self.W.T + self.vb)
        e = x - z
        dz = e * z * (1 - z)
        dh = (dz @ self.W) * h * (1 - h)
        self.W  += self.lr * (np.outer(x, dh) + np.outer(dz, h))
        self.vb += self.lr * dz
        self.hb += self.lr * dh
        return np.sqrt(np.mean(e ** 2))

    def score(self, x):
        x = self._norm(x)
        h = self._sig(x @ self.W + self.hb)
        z = self._sig(h @ self.W.T + self.vb)
        return np.sqrt(np.mean((x - z) ** 2))


class _KitNET:
    
    def __init__(self, n_features, max_ae=10, hidden_ratio=0.75, lr=0.1, seed=42):
        self.n_features = n_features; self.max_ae = max_ae
        self.hidden_ratio = hidden_ratio; self.lr = lr; self.seed = seed
        self.groups = None; self.ens = None; self.out = None

    def build_feature_map(self, X_sample):
        
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        n = self.n_features
        C = np.corrcoef(X_sample.T); C = np.nan_to_num(C, nan=0.0)
        D = 1.0 - np.abs(C); np.fill_diagonal(D, 0.0); D = (D + D.T) / 2.0
        Z = linkage(squareform(D, checks=False), method="average")
        target = min(self.max_ae, max(2, n // 3))
        labels = fcluster(Z, t=target, criterion="maxclust")
        groups = [np.where(labels==c)[0] for c in np.unique(labels)]
        if len(groups) < 2:
            idx = np.arange(n)
            groups = [g for g in np.array_split(idx, min(self.max_ae, n))]
            print(f"  [KitNET]: fallback ({len(groups)})")
        self.groups = groups
        self.ens = [_KitNET_AE(len(g), self.hidden_ratio, self.lr, self.seed) for g in self.groups]
        self.out = _KitNET_AE(len(self.groups), self.hidden_ratio, self.lr, self.seed)

    def train(self, x):
        rmses = np.array([ae.train(x[g]) for ae, g in zip(self.ens, self.groups)])
        self.out.train(rmses)

    def score(self, x):
        rmses = np.array([ae.score(x[g]) for ae, g in zip(self.ens, self.groups)])
        return self.out.score(rmses)


def run_kitnet_baseline(X_train_2d, X_val_2d, X_test_2d, y_tr, y_vl, y_te, max_train=50000):
    
    print("\n" + "=" * 60)
    print("★ KitNET (Kitsune) BASELINE")
    print("=" * 60)

    benign = X_train_2d[y_tr == 0]
    if len(benign) > max_train:
        print(f"  Benign {len(benign):,} → {max_train:,}")
        idx = np.random.RandomState(42).choice(len(benign), max_train, replace=False)
        benign = benign[idx]

    kn = _KitNET(X_train_2d.shape[1], max_ae=10)
    fm_sample = benign[:min(10000, len(benign))]
    kn.build_feature_map(fm_sample)
    print(f"   {len(kn.groups)} ")
    print(f" ({len(benign):,} ")
    for i, x in enumerate(benign):
        kn.train(x)
        if (i + 1) % 20000 == 0:
            print(f"    {i+1:,}/{len(benign):,}")

    print("   (val benign 99. persentil)...")
    val_scores = np.array([kn.score(x) for x in X_val_2d])
    thr = np.percentile(val_scores[y_vl == 0], 99)

    print("  Test score...")
    te_scores = np.array([kn.score(x) for x in X_test_2d])
    return compute_extended_metrics(y_te, (te_scores >= thr).astype(int), te_scores, "KitNET")


# ==============================================================================
# ★ Multiple SEED
# ==============================================================================
def run_multi_seed(seeds=[42, 123, 7, 99, 0]):
    print("\n" + "=" * 60)
    
    print("=" * 60)

    all_results = []
    for seed in seeds:
        print(f"\n{'─'*45}  SEED = {seed}  {'─'*45}")
        tf.random.set_seed(seed); np.random.seed(seed)

        (X_tr, X_vl, X_te, y_tr, y_vl, y_te,
         b_tr, b_vl, b_te, a_tr, a_vl, a_te,
         sc, tm, _) = sensibility_layer(BENIGN_FILE, ATTACK_FILE)

        imag, sch, _ = train_imagination(X_tr[y_tr==0], X_vl[y_vl==0])
        athr = optimize_imagination_threshold(imag, X_vl, y_vl)

        X_tr_s, y_tr_s, _  = build_schema_sequences(sch, imag, b_tr, a_tr, sc, tm, noisy=True,  seed=seed)
        X_vl_s, y_vl_s, av = build_schema_sequences(sch, imag, b_vl, a_vl, sc, tm, noisy=False, seed=seed)
        X_te_s, y_te_s, at = build_schema_sequences(sch, imag, b_te, a_te, sc, tm, noisy=False, seed=seed)

        und, _ = train_understanding(X_tr_s, y_tr_s, X_vl_s, y_vl_s)
        wi, wu, rt, _ = optimize_reason(und, X_vl_s, y_vl_s, av)
        fj, rs, us, _ = reason_predict(und, X_te_s, at, wi, wu, rt)

        all_results.append(compute_extended_metrics(y_te_s, fj, rs, f"Seed {seed}"))

    print("\n" + "=" * 60)
    print("★ (mean ± std)")
    print("=" * 60)
    keys = ['acc', 'f1', 'mcc', 'fpr', 'fnr', 'kar', 'roc', 'pr']
    summary = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        m, s = np.mean(vals), np.std(vals)
        summary[k] = (m, s)
        unit = '' if k == 'mcc' else '%'
        print(f"  {k.upper():8s}: {m:.4f}{unit} ± {s:.4f}{unit}")
    return all_results, summary


# ==============================================================================
# ★ CPU LATENCY
# ==============================================================================
def measure_cpu_latency(understanding_model, X_test_seq, imagination, X_test_2d,
                         w_imag, w_under, reason_threshold):
    import time
    print("\n" + "=" * 60)
    print("★ CPU LATENCY ")
    print("=" * 60)
    

    n = len(X_test_seq)
    start = time.time()
    recon = imagination.predict(X_test_2d, batch_size=DAE_BATCH, verbose=0)
    dae_t = time.time() - start

    start = time.time()
    us = understanding_model.predict(X_test_seq, batch_size=LSTM_BATCH, verbose=0).flatten()
    lstm_t = time.time() - start

    anom = np.mean(np.abs(X_test_2d - recon), axis=1)
    start = time.time()
    an = (anom - anom.min()) / (anom.max() - anom.min() + 1e-9)
    an = an[:len(us)]
    rs = w_under * us + w_imag * an
    _  = (rs >= reason_threshold).astype(int)
    rsn_t = time.time() - start

    total_t = dae_t + lstm_t + rsn_t
    dae_ms, lstm_ms = dae_t/n*1000, lstm_t/n*1000
    rsn_ms, tot_ms  = rsn_t/n*1000, total_t/n*1000

    print(f"\n  {'step':<12} {'Total':>10} {'Sample/ms':>12}")
    print(f"  {'─'*36}")
    print(f"  {'DAE':<12} {dae_t:>8.2f}s  {dae_ms:>10.4f}ms")
    print(f"  {'LSTM':<12} {lstm_t:>8.2f}s  {lstm_ms:>10.4f}ms")
    print(f"  {'Reason':<12} {rsn_t:>8.4f}s  {rsn_ms:>10.6f}ms")
    print(f"  {'─'*36}")
    print(f"  {'Total':<12} {total_t:>8.2f}s  {tot_ms:>10.4f}ms")
    print(f"\n  Throughput: ~{1000/tot_ms:.0f} örnek/sn")
    return {'dae_ms': dae_ms, 'lstm_ms': lstm_ms, 'rsn_ms': rsn_ms, 'total_ms': tot_ms}


# ==============================================================================
# Kantian Pipeline
# ==============================================================================
if __name__ == "__main__":

    # ── LAYER 1: SENSİBİLİTY ─────────────────────────────────────────────────
    (X_train_2d, X_val_2d, X_test_2d,
     y_tr, y_vl, y_te,
     b_train, b_val, b_test,
     a_train, a_val, a_test,
     scaler, train_median, common_cols) = sensibility_layer(BENIGN_FILE, ATTACK_FILE)

    # ── LAYER 2: IMAGINATION ─────────────────────────────────────────────────
    X_train_b = X_train_2d[y_tr == 0]
    X_val_b   = X_val_2d  [y_vl == 0]
    imagination, schema_model, imag_history = train_imagination(X_train_b, X_val_b)

    anomaly_threshold = optimize_imagination_threshold(imagination, X_val_2d, y_vl)

    print("\nKantian shema (Imagination → Understanding)...")
    X_train_seq, y_train, _           = build_schema_sequences(
        schema_model, imagination, b_train, a_train, scaler, train_median, noisy=True)
    X_val_seq,   y_val,   anomaly_val  = build_schema_sequences(
        schema_model, imagination, b_val, a_val, scaler, train_median, noisy=False)
    X_test_seq,  y_test,  anomaly_test = build_schema_sequences(
        schema_model, imagination, b_test, a_test, scaler, train_median, noisy=False)

    print(f"\n{'─'*45}")
    for name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        u, c = np.unique(yy, return_counts=True); d = dict(zip(u, c))
        print(f"  {name:5s}: Benign={d.get(0,0):>7d}  Attack={d.get(1,0):>7d}")
    print(f" {X_train_seq.shape}")
    print(f"{'─'*45}")

    # ── LAYER 3: UNDERSTANDING ───────────────────────────────────────────────
    understanding_model, under_history = train_understanding(
        X_train_seq, y_train, X_val_seq, y_val)

    # ── LAYER 4: REASON / COGNİTİVE ─────────────────────────────────────────
    w_imag, w_under, reason_threshold, _ = optimize_reason(
        understanding_model, X_val_seq, y_val, anomaly_val)

    final_judgement, reason_score, understanding_scores, _ = reason_predict(
        understanding_model, X_test_seq, anomaly_test, w_imag, w_under, reason_threshold)

    # ── THRESHOLD SENSITIVITY GRAPH ──────────────────────
    _taus = np.linspace(0.0, 1.0, 200)
    _f1s, _fprs = [], []
    for _t in _taus:
        _pred = (reason_score >= _t).astype(int)
        _f1s.append(f1_score(y_test, _pred, average='macro', zero_division=0) * 100)
        _tn, _fp, _fn, _tp = confusion_matrix(y_test, _pred, labels=[0, 1]).ravel()
        _fprs.append(_fp / (_fp + _tn) * 100 if (_fp + _tn) > 0 else 0.0)

    _fig, _ax1 = plt.subplots(figsize=(6, 3.2))
    _ax1.plot(_taus, _f1s, color='#1f4e96', lw=2)
    _ax1.set_xlabel(r'Decision threshold $\tau_{\mathcal{R}}$', fontsize=11)
    _ax1.set_ylabel('Macro-F1 (%)', color='#1f4e96', fontsize=11)
    _ax1.set_ylim(0, 100); _ax1.tick_params(axis='y', labelcolor='#1f4e96'); _ax1.grid(alpha=0.3)
    _ax2 = _ax1.twinx()
    _ax2.plot(_taus, _fprs, color='#a01b1b', lw=2, ls='--')
    _ax2.set_ylabel('FPR (%)', color='#a01b1b', fontsize=11)
    _ax2.set_ylim(0, 100); _ax2.tick_params(axis='y', labelcolor='#a01b1b')
    _ax1.axvline(reason_threshold, color='green', ls=':', lw=1.5)
    _fig.tight_layout()
    plt.savefig('threshold_sensitivity.pdf', bbox_inches='tight')
    plt.savefig('threshold_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close(_fig)
    print(f"\u2713 threshold_sensitivity.pdf (tau*={reason_threshold:.4f})")

    # ── ABLATION ────────────────────────────────────────
    metrics = compute_metrics(
        y_test, y_val, final_judgement, reason_score,
        understanding_scores, anomaly_test,
        anomaly_threshold, w_imag, w_under, reason_threshold,
        understanding_model, X_val_seq)

    # ── Proposed Model ────────────────────────────
    extended = compute_extended_metrics(
        y_test, final_judgement, reason_score, "Proposed Model (Cognitive/Reason)")

    # ── BASELINE (RF, XGB, IF, OC-SVM) ────────────────────────────
    baseline_results = run_baselines(X_train_2d, X_test_2d, y_tr, y_te)

    # ── LSTM-AE BASELINE ────────────────────────────────────────────────────
    lstm_ae_result = run_lstm_ae_baseline(X_train_2d, X_val_2d, X_test_2d, y_tr, y_vl, y_te)

    # ── CNN-LSTM BASELINE ───────────────────────────────────────────────────
    cnn_lstm_result = run_cnn_lstm_baseline(X_train_2d, X_val_2d, X_test_2d, y_tr, y_vl, y_te)

    # ── TRANSFORMER BASELINE ──────────────────────
    transformer_result = run_transformer_baseline(
        X_train_2d, X_val_2d, X_test_2d, y_tr, y_vl, y_te)

    # ── KitNET BASELINE ─────────────────────────────────
    kitnet_result = run_kitnet_baseline(
        X_train_2d, X_val_2d, X_test_2d, y_tr, y_vl, y_te)

    # ── CPU LATENCY  ─────────────
    cpu_latency = measure_cpu_latency(
        understanding_model, X_test_seq, imagination, X_test_2d,
        w_imag, w_under, reason_threshold)

    # ── Multiple SEED  ───────────────────

    multi_results, summary = run_multi_seed(seeds=[42, 123, 7, 99, 0])

    # ── DRAWING ──────────────────────────────────────────────
    plot_results(imag_history, under_history, anomaly_test, y_test,
                 final_judgement, anomaly_threshold, metrics)
