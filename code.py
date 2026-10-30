
import os, sys, math, random, warnings, zipfile, copy, json, time, pickle, subprocess, glob, shutil, platform, hashlib
import logging
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def _pip_install(pairs):
    for mod, pkg in pairs:
        try:
            __import__(mod)
        except ImportError:
            print(f"[setup] installing {pkg} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


_pip_install([("openpyxl", "openpyxl"), ("xgboost", "xgboost"), ("lightgbm", "lightgbm"),
              ("catboost", "catboost"), ("shap", "shap")])

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from scipy import stats as sps
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, spearmanr, pearsonr

QUICK_TEST = False        
RESET_CHECKPOINTS = False  

CSV_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CSV_PATH = CSV_NAME
CSV_URLS = [   
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
]

N_FOLDS = 5
N_REPEATS = 5
VAL_FRACTION = 0.15
EPOCHS = 200
BATCH_SIZE = 64
PATIENCE = 20
FOCAL_GAMMA_DEFAULT = 2.0
LABEL_SMOOTH = 0.03
N_ENSEMBLE = 3
BASELINE_LR, BASELINE_WD = 3e-4, 5e-3

PSO_PARTICLES, PSO_ITERS, PSO_EPOCHS, PSO_PATIENCE = 8, 6, 60, 10
PSO_W, PSO_C1, PSO_C2 = 0.6, 1.4, 1.4
RERUN_PSO = False
PUBLISHED_CFG = {
    "d_token": 32, "n_heads": 4, "n_layers": 1,
    "dropout": 0.43087266123881035, "attn_dropout": 0.2855948465660245,
    "lr": 0.0011391406735339136, "weight_decay": 0.0001,
    "focal_gamma": 1.0, "mixup_alpha": 0.0,
}
PUBLISHED_PSO_ROWS = [
    (1, 1, 0.5918, 32, 4, 2, 0.000766),
    (1, 2, 0.525, 64, 8, 3, 0.000107),
    (1, 3, 0.5714, 24, 4, 1, 0.000596),
    (1, 4, 0.5205, 16, 4, 1, 0.000472),
    (1, 5, 0.587, 48, 4, 2, 0.000179),
    (1, 6, 0.5783, 64, 8, 2, 0.00102),
    (1, 7, 0.5897, 16, 2, 1, 0.000952),
    (1, 8, 0.5682, 24, 4, 1, 0.00244),
    (2, 1, 0.5918, 32, 4, 2, 0.000766),
    (2, 2, 0.5833, 32, 4, 1, 0.000803),
    (2, 3, 0.5714, 32, 4, 1, 0.000767),
    (2, 4, 0.5647, 16, 4, 1, 0.00063),
    (2, 5, 0.5238, 32, 4, 2, 0.000597),
    (2, 6, 0.6452, 32, 4, 1, 0.000839),
    (2, 7, 0.5676, 24, 4, 1, 0.000915),
    (2, 8, 0.6667, 32, 4, 2, 0.00186),
    (3, 1, 0.551, 32, 4, 2, 0.00225),
    (3, 2, 0.5867, 16, 2, 1, 0.003),
    (3, 3, 0.6392, 32, 4, 1, 0.00173),
    (3, 4, 0.6835, 32, 8, 2, 0.000991),
    (3, 5, 0.6933, 32, 4, 1, 0.000536),
    (3, 6, 0.5591, 16, 2, 1, 0.000808),
    (3, 7, 0.5974, 32, 4, 1, 0.00183),
    (3, 8, 0.65, 32, 8, 2, 0.00158),
    (4, 1, 0.5405, 24, 4, 1, 0.00069),
    (4, 2, 0.6301, 16, 2, 1, 0.003),
    (4, 3, 0.6598, 32, 4, 1, 0.000747),
    (4, 4, 0.7089, 32, 4, 2, 0.00111),
    (4, 5, 0.625, 24, 4, 1, 0.000502),
    (4, 6, 0.6575, 24, 4, 1, 0.000788),
    (4, 7, 0.5682, 32, 4, 2, 0.000813),
    (4, 8, 0.6567, 48, 4, 2, 0.00121),
    (5, 1, 0.5333, 32, 4, 2, 0.00061),
    (5, 2, 0.5789, 16, 2, 1, 0.003),
    (5, 3, 0.6957, 32, 4, 1, 0.000707),
    (5, 4, 0.6667, 32, 4, 2, 0.00119),
    (5, 5, 0.6667, 24, 4, 1, 0.000774),
    (5, 6, 0.6585, 32, 4, 2, 0.000805),
    (5, 7, 0.5435, 32, 4, 2, 0.000781),
    (5, 8, 0.6389, 48, 4, 2, 0.00173),
    (6, 1, 0.5789, 32, 8, 2, 0.00119),
    (6, 2, 0.6567, 16, 4, 1, 0.00262),
    (6, 3, 0.7179, 32, 4, 1, 0.00114),
    (6, 4, 0.6923, 32, 4, 2, 0.00115),
    (6, 5, 0.6579, 32, 4, 1, 0.000784),
    (6, 6, 0.6761, 32, 8, 2, 0.000842),
    (6, 7, 0.5714, 32, 4, 2, 0.00125),
    (6, 8, 0.6667, 32, 8, 2, 0.00202),
]

RUN_ABLATIONS = True
RUN_GBDT = True
GBDT_SEARCH_BUDGET = 48         
RUN_NESTED_PSO = True           
NESTED_PSO_REPEATS = 1          
NESTED_PSO_PARTICLES, NESTED_PSO_ITERS = 6, 4   
RUN_TRUE_SHAP = True
N_PERM_REPEATS = 5              
IG_STEPS = 32
BOOTSTRAP_RESAMPLES = 5000

if QUICK_TEST:
    N_FOLDS, N_REPEATS = 2, 1
    EPOCHS, PSO_EPOCHS = 4, 3
    PSO_PARTICLES, PSO_ITERS = 2, 2
    GBDT_SEARCH_BUDGET = 3
    NESTED_PSO_REPEATS = 1
    NESTED_PSO_PARTICLES, NESTED_PSO_ITERS = 2, 2
    N_PERM_REPEATS, IG_STEPS, BOOTSTRAP_RESAMPLES = 2, 8, 200
    PATIENCE = 2

N_WORKERS = int(os.environ.get("CAFTX_WORKERS", 2))
WORKER = None
if "--worker" in sys.argv:
    _i = sys.argv.index("--worker")
    WORKER = (int(sys.argv[_i + 1]), int(sys.argv[_i + 2]))
GBDT_THREADS = max(1, (os.cpu_count() or 2) // max(1, N_WORKERS))
torch.set_num_threads(GBDT_THREADS)

OUT_DIR = "output"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
LOG_DIR = os.path.join(OUT_DIR, "logs")
CKPT_DIR = "ckpt"
for _d in (FIG_DIR, TAB_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#3B3B3B", "axes.linewidth": 0.9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": True, "legend.framealpha": 0.9,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})
sns.set_style("whitegrid", {"font.family": "serif"})
PALETTE = ["#1B4F72", "#B03A2E", "#1E8449", "#B7950B", "#6C3483", "#117A65", "#CA6F1E", "#566573", "#7D3C98", "#2E86C1"]
LW_PROPOSED, LW_BASELINE = 3.0, 1.5


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s); st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _tex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("±", r"$\pm$"), ("≤", r"$\leq$"), ("≥", r"$\geq$"), ("Δ", r"$\Delta$"), ("α", r"$\alpha$"),
                 ("γ", r"$\gamma$"), ("µ", r"$\mu$"), ("→", r"$\rightarrow$")]:
        s = s.replace(a, b)
    return s


def _df_to_latex(df, name, float_format="%.4f", caption=None, bold_max_cols=None):
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else float_format % v
        return _tex_escape(v)

    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    idx_name = df.index.name or ""
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\begin{tabular}{%s}" % ("l" + "r" * len(cols)), r"\toprule",
             " & ".join([_tex_escape(idx_name)] + [_tex_escape(c) for c in cols]) + r" \\", r"\midrule"]
    best = {}
    if bold_max_cols:
        for c in bold_max_cols:
            if c in df.columns:
                try:
                    best[c] = pd.to_numeric(df[c]).idxmax()
                except Exception:
                    pass
    for idx, row in df.iterrows():
        idx_s = " / ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        cells = []
        for c, v in zip(df.columns, row.tolist()):
            s = fmt(v)
            if c in best and best[c] == idx:
                s = r"\textbf{%s}" % s
            cells.append(s)
        lines.append(" & ".join([_tex_escape(idx_s)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(r"\caption{%s}" % caption)
    lines.append(r"\end{table}")
    with open(os.path.join(TAB_DIR, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_table(df, name, caption=None, bold_max_cols=None, index=True):
    df = df.copy()
    df.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), float_format="%.4f", index=index, encoding="utf-8-sig")
    try:
        df.to_excel(os.path.join(TAB_DIR, f"{name}.xlsx"), engine="openpyxl", index=index)
    except Exception:
        pass
    try:
        d2 = df if index else df.set_index(df.columns[0])
        _df_to_latex(d2, name, caption=caption, bold_max_cols=bold_max_cols)
    except Exception:
        pass


def find_or_get_csv():
    cands = [CSV_PATH, os.path.join("/content", CSV_NAME), os.path.join("/content/drive/MyDrive", CSV_NAME),
             os.path.join(os.getcwd(), CSV_NAME)]
    for pat in ("*Employee-Attrition*.csv", "*emp_attrition*.csv", "*attrition*.csv"):
        cands += glob.glob(pat) + glob.glob(os.path.join("/content", pat)) + glob.glob(os.path.join("data", pat))
    for p in cands:
        if os.path.isfile(p):
            return p
    import urllib.request
    for url in CSV_URLS:
        try:
            print(f"[data] local CSV not found, downloading {url}")
            urllib.request.urlretrieve(url, CSV_NAME)
            if os.path.isfile(CSV_NAME) and os.path.getsize(CSV_NAME) > 100000:
                return CSV_NAME
        except Exception as e:
            print("[data] download failed:", e)
    try:
        from google.colab import files
        print("[data] please upload WA_Fn-UseC_-HR-Employee-Attrition.csv")
        up = files.upload()
        for k in up:
            return k
    except Exception:
        pass
    raise FileNotFoundError("Could not find the IBM HR Analytics CSV. Place it next to code.py.")


def load_ibm_hr_dataset():
    path = find_or_get_csv()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().replace("﻿", "") for c in df.columns]
    if not pd.api.types.is_numeric_dtype(df["Attrition"]):
        df["Attrition"] = df["Attrition"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    df["Attrition"] = df["Attrition"].astype(int)
    n_dupes = int(df.drop(columns=["EmployeeNumber"], errors="ignore").duplicated().sum())
    print(f"Loaded {path}: shape={df.shape}, positive rate={df['Attrition'].mean():.3f}, duplicate rows={n_dupes}")
    return df, "Attrition"


CATEGORICAL_COLS = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def preprocess(df, target_col):
    df = df.copy().dropna(subset=[target_col])
    y = df[target_col].values.astype(np.int64)
    X = df.drop(columns=[target_col])
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])
    X = X.drop(columns=[c for c in X.columns if X[c].nunique() <= 1])
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("missing")
    feature_names = num_cols + cat_cols
    return (X[num_cols].values.astype(np.float32), X[cat_cols].values, y, feature_names, num_cols, cat_cols)


def prepare_data(df, target_col):
    x_num_raw, x_cat_raw, y, feature_names, num_cols, cat_cols = preprocess(df, target_col)
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_cat_full = ordinal.fit_transform(x_cat_raw).astype(np.int64)
    cat_card = [len(c) for c in ordinal.categories_]
    return dict(x_num_raw=x_num_raw, x_cat_raw=x_cat_raw, x_cat_full=x_cat_full, y=y,
                feature_names=feature_names, num_cols=num_cols, cat_cols=cat_cols,
                n_num=x_num_raw.shape[1], cat_card=cat_card, categories=[list(c) for c in ordinal.categories_])


def sanity_check_no_leakage(tr_idx, va_idx, test_idx):
    s_tr, s_va, s_te = set(tr_idx.tolist()), set(va_idx.tolist()), set(test_idx.tolist())
    assert s_tr.isdisjoint(s_te), "LEAKAGE: train/test overlap."
    assert s_va.isdisjoint(s_te), "LEAKAGE: val/test overlap."
    assert s_tr.isdisjoint(s_va), "LEAKAGE: train/val overlap."


class FeatureTokenizer(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_token):
        super().__init__()
        self.n_num, self.d_token = n_num, d_token
        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.uniform_(self.num_weight, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
            nn.init.uniform_(self.num_bias, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cardinalities])
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x_num, x_cat, cat_alpha=None, cat_base=None):
        tokens = []
        b = x_num.shape[0] if self.n_num > 0 else x_cat.shape[0]
        if self.n_num > 0:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        for i, emb in enumerate(self.cat_embeddings):
            e = emb(x_cat[:, i])
            if cat_alpha is not None:
                e0 = emb(cat_base[i].expand(b))
                e = e0 + cat_alpha[:, i:i + 1] * (e - e0)
            tokens.append(e.unsqueeze(1))
        cls = self.cls.expand(b, -1, -1)
        return torch.cat([cls] + tokens, dim=1)


class GatedFeatureInteraction(nn.Module):
    def __init__(self, d_token, dropout):
        super().__init__()
        self.summary = nn.Linear(d_token, d_token)
        self.gate = nn.Sequential(nn.Linear(d_token * 2, d_token), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_token, d_token), nn.Sigmoid())

    def gate_tensor(self, tok):
        ctx = self.summary(tok.mean(dim=1, keepdim=True)).expand_as(tok)
        return self.gate(torch.cat([tok, ctx], dim=-1))

    def forward(self, tok):
        return tok * self.gate_tensor(tok)


class ContextFreeGate(nn.Module):          
    def __init__(self, d_token, dropout):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_token, d_token), nn.Sigmoid())

    def forward(self, tok):
        return tok * self.gate(tok)


class ScalarGate(nn.Module):               
    def __init__(self, d_token, dropout):
        super().__init__()
        self.summary = nn.Linear(d_token, d_token)
        self.gate = nn.Sequential(nn.Linear(d_token * 2, d_token), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_token, 1), nn.Sigmoid())

    def forward(self, tok):
        ctx = self.summary(tok.mean(dim=1, keepdim=True)).expand_as(tok)
        return tok * self.gate(torch.cat([tok, ctx], dim=-1))


class AdditiveInteraction(nn.Module):      
    def __init__(self, d_token, dropout):
        super().__init__()
        self.summary = nn.Linear(d_token, d_token)
        self.net = nn.Sequential(nn.Linear(d_token * 2, d_token), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_token, d_token))

    def forward(self, tok):
        ctx = self.summary(tok.mean(dim=1, keepdim=True)).expand_as(tok)
        return tok + self.net(torch.cat([tok, ctx], dim=-1))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_mult, dropout, attn_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(nn.Linear(d_token, d_token * ffn_mult), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_token * ffn_mult, d_token))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, attn_w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x, attn_w


CONST_GATE_VALUE = 0.5


class CA_FTX_Plus(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_token=32, n_heads=4, n_layers=2, ffn_mult=2,
                 dropout=0.25, attn_dropout=0.15, use_gate=True, gate_type="full"):
        super().__init__()
        if not use_gate:
            gate_type = "none"
        self.gate_type = gate_type
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_token)
        if gate_type == "full":
            self.gate = GatedFeatureInteraction(d_token, dropout)
        elif gate_type == "nocontext":
            self.gate = ContextFreeGate(d_token, dropout)
        elif gate_type == "scalar":
            self.gate = ScalarGate(d_token, dropout)
        elif gate_type == "additive":
            self.gate = AdditiveInteraction(d_token, dropout)
        else:
            self.gate = None
        self.token_dropout = nn.Dropout(dropout * 0.5)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(d_token, n_heads, ffn_mult, dropout, attn_dropout)
                                     for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_token, 1))
        self.gate_override = None   

    def forward(self, x_num, x_cat, return_attn=False, cat_alpha=None, cat_base=None):
        tok = self.tokenizer(x_num, x_cat, cat_alpha, cat_base)
        if self.gate_override is not None:
            tok = tok * self.gate_override
        elif self.gate_type == "const":
            tok = tok * CONST_GATE_VALUE
        elif self.gate is not None:
            tok = self.gate(tok)
        tok = self.token_dropout(tok)
        attn_maps = []
        for blk in self.blocks:
            tok, aw = blk(tok)
            attn_maps.append(aw)
        tok = self.final_norm(tok)
        cls_out = tok[:, 0]
        logit = self.head(cls_out).squeeze(-1)
        if return_attn:
            return logit, cls_out, attn_maps
        return logit, cls_out

    @torch.no_grad()
    def gate_diagnostics_tensors(self, x_num, x_cat):
        assert self.gate_type == "full"
        self.eval()
        tok = self.tokenizer(x_num, x_cat)
        return tok, self.gate.gate_tensor(tok)


class MixtureModel(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x_num, x_cat, return_attn=False, **kw):
        outs = [m(x_num, x_cat, **kw) for m in self.members]
        p = torch.stack([torch.sigmoid(o[0]) for o in outs]).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p), outs[0][1]


class MLPBaseline(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_emb=8, hidden=128, dropout=0.25):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, d_emb) for c in cat_cardinalities])
        in_dim = n_num + d_emb * len(cat_cardinalities)
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout))
        self.block1 = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x_num, x_cat, return_attn=False):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat([x_num] + embs, dim=1) if embs else x_num
        h = self.inp(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out(h).squeeze(-1), h


class PlainFeatureTransformer(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_token=32):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_token)
        self.block = TransformerEncoderBlock(d_token, n_heads=2, ffn_mult=2, dropout=0.1, attn_dropout=0.1)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_num, x_cat, return_attn=False):
        tok = self.tokenizer(x_num, x_cat)
        tok, _ = self.block(tok)
        cls_out = tok[:, 0]
        return self.head(cls_out).squeeze(-1), cls_out


class WideShallowMLP(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_emb=8, hidden=256, dropout=0.3):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, d_emb) for c in cat_cardinalities])
        in_dim = n_num + d_emb * len(cat_cardinalities)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x_num, x_cat, return_attn=False):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat([x_num] + embs, dim=1) if embs else x_num
        return self.net(x).squeeze(-1), x


BASELINE_REGISTRY = {
    "MLP (residual, regularized)": MLPBaseline,
    "Feature Transformer (plain, no reg.)": PlainFeatureTransformer,
    "Wide-Shallow MLP": WideShallowMLP,
}
NN_BASELINES = list(BASELINE_REGISTRY.keys())
GBDT_LIBS = ["XGBoost", "LightGBM", "CatBoost"]
GBDT_DEFAULT_NAMES = [f"{g} (default)" for g in GBDT_LIBS]
LR_NAME = "Logistic Regression"
PROPOSED_NAME = "CA-FTX+ (PSO-tuned, mixture)"
SINGLE_FULL = "CA-FTX+ single member (full)"
NOGATE_MIX = "CA-FTX+ w/o gate (mixture)"
FT_MIX = "Feature Transformer (plain, mixture x3)"
NESTED_NAME = "CA-FTX+ (nested PSO, mixture)"
ABLATION_SPECS = [
    ("w/o gate", dict(gate_type="none")),
    ("Constant gate (0.5)", dict(gate_type="const")),
    ("Context-free gate", dict(gate_type="nocontext")),
    ("Scalar (per-token) gate", dict(gate_type="scalar")),
    ("Additive interaction (no sigmoid gate)", dict(gate_type="additive")),
    ("w/o focal modulation (gamma=0)", dict(gamma=0.0)),
    ("w/o class weighting (alpha=0.5)", dict(alpha=0.5)),
    ("Default hyper-parameters (untuned)", dict(default_hp=True)),
]
ABLATION_NAMES = [n for n, _ in ABLATION_SPECS]
GATE_VARIANTS = ["w/o gate", "Constant gate (0.5)", "Context-free gate", "Scalar (per-token) gate",
                 "Additive interaction (no sigmoid gate)"]
MAIN_MODEL_NAMES = [PROPOSED_NAME] + NN_BASELINES + (GBDT_LIBS if RUN_GBDT else [])
SHORT = {PROPOSED_NAME: "CA-FTX+", "MLP (residual, regularized)": "Residual MLP",
         "Feature Transformer (plain, no reg.)": "FT-Transformer (plain)", "Wide-Shallow MLP": "Wide-Shallow MLP",
         SINGLE_FULL: "CA-FTX+ (single)", NOGATE_MIX: "CA-FTX+ w/o gate (mix.)", FT_MIX: "FT-Transformer (mix.)"}
MODEL_COLORS = {PROPOSED_NAME: PALETTE[0], "MLP (residual, regularized)": PALETTE[1],
                "Feature Transformer (plain, no reg.)": PALETTE[2], "Wide-Shallow MLP": PALETTE[3],
                "XGBoost": PALETTE[4], "LightGBM": PALETTE[6], "CatBoost": PALETTE[5]}


def sh(name):
    return SHORT.get(name, name)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=FOCAL_GAMMA_DEFAULT, label_smoothing=LABEL_SMOOTH):
        super().__init__()
        self.alpha, self.gamma, self.ls = alpha, gamma, label_smoothing

    def forward(self, logits, targets):
        targets = targets.float() * (1 - self.ls) + 0.5 * self.ls
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()


class TabDataset(Dataset):
    def __init__(self, x_num, x_cat, y):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class FastLoader:
    def __init__(self, xn, xc, y):
        self.xn = torch.tensor(xn, dtype=torch.float32, device=DEVICE)
        self.xc = torch.tensor(xc, dtype=torch.long, device=DEVICE)
        self.y = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        self.n = len(y)

    def __iter__(self):
        perm = torch.randperm(self.n, device=DEVICE)
        for i in range(0, self.n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:      
                continue
            yield self.xn[idx], self.xc[idx], self.y[idx]


def make_loader(xn, xc, y):
    return FastLoader(xn, xc, y)


def mixup_batch(x_num, y, alpha):
    if alpha <= 0:
        return x_num, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x_num.size(0), device=x_num.device)
    return lam * x_num + (1 - lam) * x_num[perm], y, perm, lam


_THR_GRID = np.linspace(0.05, 0.95, 181)


def find_best_threshold(y_true, probs):
    y1 = np.asarray(y_true) == 1
    pred = np.asarray(probs)[None, :] >= _THR_GRID[:, None]
    tp = (pred & y1[None, :]).sum(1).astype(float)
    fp = (pred & ~y1[None, :]).sum(1).astype(float)
    fn = (~pred & y1[None, :]).sum(1).astype(float)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    k = int(np.argmax(f1))
    return float(_THR_GRID[k]), float(f1[k])


def train_one_model(model, train_loader, val_x, epochs, lr, weight_decay, patience, alpha_focal, gamma_focal,
                    mixup_alpha, use_mixup, is_ca_ftx, label_smoothing=LABEL_SMOOTH):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=(DEVICE.type == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(10, epochs // 6))
    criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, label_smoothing=label_smoothing)
    val_xn, val_xc, val_y = val_x
    val_xn_t = torch.tensor(val_xn, dtype=torch.float32, device=DEVICE)
    val_xc_t = torch.tensor(val_xc, dtype=torch.long, device=DEVICE)
    best_auc, best_state, bad_epochs = -1, None, 0
    for epoch in range(epochs):
        model.train()
        for xn, xc, y in train_loader:
            xn, xc, y = xn.to(DEVICE), xc.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if use_mixup and mixup_alpha > 0 and is_ca_ftx:
                xn_m, y_a, perm, lam = mixup_batch(xn, y, mixup_alpha)
                logit, _ = model(xn_m, xc)
                y_b = y[perm] if perm is not None else y
                loss = lam * criterion(logit, y_a) + (1 - lam) * criterion(logit, y_b)
            else:
                logit, _ = model(xn, xc)
                loss = criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_xn_t, val_xc_t)[0]).cpu().numpy()
        try:
            val_auc = roc_auc_score(val_y, val_prob)
        except ValueError:
            val_auc = 0.5
        if val_auc > best_auc:
            best_auc, best_state, bad_epochs = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc


def predict_prob(model, x_num, x_cat):
    model.eval()
    with torch.no_grad():
        xn_t = torch.tensor(x_num, dtype=torch.float32, device=DEVICE)
        xc_t = torch.tensor(x_cat, dtype=torch.long, device=DEVICE)
        return torch.sigmoid(model(xn_t, xc_t)[0]).cpu().numpy()


def evaluate_from_prob(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    metrics = {
        "AUC_ROC": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "AUC_PR": average_precision_score(y, prob),
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "BalancedAcc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "Accuracy": accuracy_score(y, pred),
        "Threshold": float(threshold),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return metrics, pred


def ca_builder(n_num, cat_card, cfg, gate_type="full"):
    return lambda: CA_FTX_Plus(n_num, cat_card, d_token=cfg["d_token"], n_heads=cfg["n_heads"],
                               n_layers=cfg["n_layers"], dropout=cfg["dropout"],
                               attn_dropout=cfg["attn_dropout"], gate_type=gate_type)


def tr_kwargs_from_cfg(cfg, alpha_focal):
    return dict(epochs=EPOCHS, lr=cfg["lr"], weight_decay=cfg["weight_decay"], patience=PATIENCE,
                alpha_focal=alpha_focal, gamma_focal=cfg["focal_gamma"], mixup_alpha=cfg["mixup_alpha"],
                use_mixup=True, is_ca_ftx=True, label_smoothing=LABEL_SMOOTH)


def baseline_tr_kwargs(alpha_focal):
    return dict(epochs=EPOCHS, lr=BASELINE_LR, weight_decay=BASELINE_WD, patience=PATIENCE,
                alpha_focal=alpha_focal, gamma_focal=FOCAL_GAMMA_DEFAULT, mixup_alpha=0.0,
                use_mixup=False, is_ca_ftx=False, label_smoothing=LABEL_SMOOTH)


ARCH_CHOICES = [(16, 2, 1), (16, 4, 1), (24, 4, 1), (32, 4, 1), (32, 4, 2),
                (32, 8, 2), (48, 4, 2), (48, 8, 2), (64, 8, 2), (64, 8, 3)]
PSO_BOUNDS = {
    "arch_idx": (0, len(ARCH_CHOICES) - 1e-6),
    "dropout": (0.10, 0.50),
    "attn_dropout": (0.05, 0.30),
    "log_lr": (math.log10(1e-4), math.log10(3e-3)),
    "log_wd": (math.log10(1e-4), math.log10(2e-2)),
    "focal_gamma": (1.0, 3.0),
    "mixup_alpha": (0.0, 0.5),
}


def decode_particle(pos):
    arch = ARCH_CHOICES[int(np.clip(round(pos["arch_idx"]), 0, len(ARCH_CHOICES) - 1))]
    return dict(
        d_token=arch[0], n_heads=arch[1], n_layers=arch[2],
        dropout=float(np.clip(pos["dropout"], *PSO_BOUNDS["dropout"])),
        attn_dropout=float(np.clip(pos["attn_dropout"], *PSO_BOUNDS["attn_dropout"])),
        lr=float(10 ** np.clip(pos["log_lr"], *PSO_BOUNDS["log_lr"])),
        weight_decay=float(10 ** np.clip(pos["log_wd"], *PSO_BOUNDS["log_wd"])),
        focal_gamma=float(np.clip(pos["focal_gamma"], *PSO_BOUNDS["focal_gamma"])),
        mixup_alpha=float(np.clip(pos["mixup_alpha"], *PSO_BOUNDS["mixup_alpha"])),
    )


def pso_fitness(cfg, n_num, cat_card, tr_data, val_data, alpha_focal):
    xn_tr, xc_tr, y_tr = tr_data
    seed_everything(SEED)
    model = CA_FTX_Plus(n_num, cat_card, d_token=cfg["d_token"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                        dropout=cfg["dropout"], attn_dropout=cfg["attn_dropout"]).to(DEVICE)
    loader = make_loader(xn_tr, xc_tr, y_tr)
    model, _ = train_one_model(model, loader, val_data, PSO_EPOCHS, cfg["lr"], cfg["weight_decay"], PSO_PATIENCE,
                               alpha_focal, cfg["focal_gamma"], cfg["mixup_alpha"], use_mixup=True, is_ca_ftx=True)
    val_xn, val_xc, val_y = val_data
    return find_best_threshold(val_y, predict_prob(model, val_xn, val_xc))[1]


def run_pso_search(n_num, cat_card, tr_data, val_data, alpha_focal, rng, n_particles=None, n_iters=None, verbose=True):
    n_particles = n_particles or PSO_PARTICLES
    n_iters = n_iters or PSO_ITERS
    keys = list(PSO_BOUNDS.keys())
    swarm = []
    for _ in range(n_particles):
        pos = {k: float(rng.uniform(*PSO_BOUNDS[k])) for k in keys}
        swarm.append({"pos": pos, "vel": {k: 0.0 for k in keys}, "best_pos": dict(pos), "best_score": -np.inf})
    gbest_pos, gbest_score = None, -np.inf
    history, trace = [], []
    for it in range(n_iters):
        for p_i, particle in enumerate(swarm):
            cfg = decode_particle(particle["pos"])
            score = pso_fitness(cfg, n_num, cat_card, tr_data, val_data, alpha_focal)
            trace.append(dict(iteration=it + 1, particle=p_i + 1, val_F1=score, **cfg))
            if score > particle["best_score"]:
                particle["best_score"], particle["best_pos"] = score, dict(particle["pos"])
            if score > gbest_score:
                gbest_score, gbest_pos = score, dict(particle["pos"])
            if verbose:
                print(f"  PSO iter {it + 1}/{n_iters} particle {p_i + 1}/{n_particles}: val F1={score:.4f} "
                      f"arch={(cfg['d_token'], cfg['n_heads'], cfg['n_layers'])} lr={cfg['lr']:.2e}")
        for particle in swarm:
            for k in keys:
                r1, r2 = rng.rand(), rng.rand()
                particle["vel"][k] = (PSO_W * particle["vel"][k]
                                      + PSO_C1 * r1 * (particle["best_pos"][k] - particle["pos"][k])
                                      + PSO_C2 * r2 * (gbest_pos[k] - particle["pos"][k]))
                lo, hi = PSO_BOUNDS[k]
                particle["pos"][k] = float(np.clip(particle["pos"][k] + particle["vel"][k], lo, hi))
        history.append({"iteration": it + 1, "best_f1_so_far": gbest_score})
    return decode_particle(gbest_pos), gbest_score, pd.DataFrame(history), pd.DataFrame(trace)


def pso_on_indices(base_idx, D, rng_seed, verbose=True, n_particles=None, n_iters=None):
    y = D["y"]
    s_tr, s_val = train_test_split(base_idx, test_size=0.20, stratify=y[base_idx], random_state=SEED)
    sc = StandardScaler()
    xn_tr = sc.fit_transform(D["x_num_raw"][s_tr])
    xn_val = sc.transform(D["x_num_raw"][s_val])
    tr_data = (xn_tr, D["x_cat_full"][s_tr], y[s_tr])
    val_data = (xn_val, D["x_cat_full"][s_val], y[s_val])
    return run_pso_search(D["n_num"], D["cat_card"], tr_data, val_data, 1 - y[s_tr].mean(),
                          np.random.RandomState(rng_seed), n_particles=n_particles, n_iters=n_iters, verbose=verbose)


def global_search_split(D):
    y = D["y"]
    search_idx, _ = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=SEED)
    s_tr, s_val = train_test_split(search_idx, test_size=0.20, stratify=y[search_idx], random_state=SEED)
    return search_idx, s_tr, s_val


def get_pso_config(D):
    if not RERUN_PSO:
        rows = PUBLISHED_PSO_ROWS
        trace = pd.DataFrame(rows, columns=["iteration", "particle", "val_F1", "d_token", "n_heads", "n_layers", "lr"])
        its = sorted(trace.iteration.unique())
        hist = pd.DataFrame({"iteration": its})
        best, hv = -1, []
        for it in its:
            best = max(best, trace[trace.iteration == it].val_F1.max())
            hv.append(best)
        hist["best_f1_so_far"] = hv
        return dict(PUBLISHED_CFG), hist, trace, "published run (re-used)"
    cached = ckpt_load("pso_global")
    if cached is not None:
        return cached["cfg"], cached["hist"], cached["trace"], "re-run (cached)"
    print(f"\nPSO: {PSO_PARTICLES} particles x {PSO_ITERS} iterations")
    search_idx, _, _ = global_search_split(D)
    cfg, f1, hist, trace = pso_on_indices(search_idx, D, SEED)
    print(f" PSO done: inner-val F1={f1:.4f}, config={cfg}")
    ckpt_save("pso_global", dict(cfg=cfg, hist=hist, trace=trace))
    return cfg, hist, trace, "re-run"


GBDT_DEFAULT = {
    "XGBoost": dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=2, reg_lambda=5.0),
    "LightGBM": dict(num_leaves=8, min_child_samples=15, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0),
    "CatBoost": dict(depth=4, learning_rate=0.03, l2_leaf_reg=5.0),
}
GBDT_FIXED = dict(n_estimators=1000, early_stopping_rounds=50)


def sample_gbdt_params(lib, rng):
    lr = float(10 ** rng.uniform(-2, -0.7))
    if lib == "XGBoost":
        return dict(max_depth=int(rng.randint(2, 7)), learning_rate=lr, subsample=float(rng.uniform(0.6, 1.0)),
                    colsample_bytree=float(rng.uniform(0.5, 1.0)), min_child_weight=int(rng.randint(1, 11)),
                    reg_lambda=float(10 ** rng.uniform(-1, 1.7)))
    if lib == "LightGBM":
        return dict(num_leaves=int(rng.randint(4, 33)), min_child_samples=int(rng.randint(5, 41)), learning_rate=lr,
                    subsample=float(rng.uniform(0.6, 1.0)), colsample_bytree=float(rng.uniform(0.5, 1.0)),
                    reg_lambda=float(10 ** rng.uniform(-1, 1.7)))
    return dict(depth=int(rng.randint(3, 8)), learning_rate=lr, l2_leaf_reg=float(10 ** rng.uniform(0, 1.5)))


def make_df(xn, xc, D):
    df = pd.DataFrame(xn, columns=D["num_cols"])
    for j, c in enumerate(D["cat_cols"]):
        df[c] = xc[:, j].astype(np.int64)
    return df


def fit_gbdt(lib, params, Xtr, ytr, Xva, yva, cat_idx, seed):
    spw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    t0 = time.perf_counter()
    if lib == "XGBoost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=GBDT_FIXED["n_estimators"], scale_pos_weight=spw, eval_metric="auc",
                              early_stopping_rounds=GBDT_FIXED["early_stopping_rounds"], tree_method="hist",
                              random_state=seed, n_jobs=GBDT_THREADS, verbosity=0, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        n_trees = int(m.best_iteration) + 1
    elif lib == "LightGBM":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=GBDT_FIXED["n_estimators"], scale_pos_weight=spw, subsample_freq=1,
                               random_state=seed, n_jobs=GBDT_THREADS, verbose=-1, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(GBDT_FIXED["early_stopping_rounds"], verbose=False)])
        n_trees = int(m.best_iteration_ or GBDT_FIXED["n_estimators"])
    else:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=GBDT_FIXED["n_estimators"], scale_pos_weight=spw, eval_metric="AUC",
                               random_seed=seed, verbose=0, thread_count=GBDT_THREADS, allow_writing_files=False, **params)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=GBDT_FIXED["early_stopping_rounds"],
              cat_features=cat_idx)
        n_trees = int(m.get_best_iteration()) + 1
    return m, time.perf_counter() - t0, n_trees


def get_gbdt_config(D):
    if not RUN_GBDT:
        return None
    cached = ckpt_load("gbdt_cfg")
    if cached is not None:
        return cached

    _, s_tr, s_val = global_search_split(D)
    sc = StandardScaler().fit(D["x_num_raw"][s_tr])
    Xtr = make_df(D["x_num_raw"][s_tr], D["x_cat_full"][s_tr], D)
    Xva = make_df(D["x_num_raw"][s_val], D["x_cat_full"][s_val], D)
    ytr, yva = D["y"][s_tr], D["y"][s_val]
    cat_idx = list(range(D["n_num"], D["n_num"] + len(D["cat_cols"])))
    out = dict(tuned={}, best_val_F1={}, trace=[])
    for k, lib in enumerate(GBDT_LIBS):
        rng = np.random.RandomState(SEED + 17 * (k + 1))
        best_f1, best_params = -1, None
        for i in range(GBDT_SEARCH_BUDGET):
            params = sample_gbdt_params(lib, rng)
            try:
                m, _, _ = fit_gbdt(lib, params, Xtr, ytr, Xva, yva, cat_idx, SEED)
                f1 = find_best_threshold(yva, m.predict_proba(Xva)[:, 1])[1]
            except Exception as e:
                print("   search trial failed:", repr(e)[:120]); f1 = -1
            out["trace"].append(dict(library=lib, trial=i + 1, val_F1=f1, **params))
            if f1 > best_f1:
                best_f1, best_params = f1, params
        out["tuned"][lib], out["best_val_F1"][lib] = best_params, best_f1
        print(f"   {lib}: best inner-val F1={best_f1:.4f}  params={best_params}")
    ckpt_save("gbdt_cfg", out)
    return out


def baseline_cat_values(xc_tr):
    return np.array([np.bincount(xc_tr[:, j]).argmax() for j in range(xc_tr.shape[1])], dtype=np.int64)


def permutation_importance(model, xn, xc, y, n_repeats, seed):
    rng = np.random.RandomState(seed)
    base_auc = roc_auc_score(y, predict_prob(model, xn, xc))
    n_num, n_cat = xn.shape[1], xc.shape[1]
    drops = np.zeros(n_num + n_cat)
    for r in range(n_repeats):
        for j in range(n_num):
            xp = xn.copy(); xp[:, j] = rng.permutation(xp[:, j])
            drops[j] += base_auc - roc_auc_score(y, predict_prob(model, xp, xc))
        for j in range(n_cat):
            xp = xc.copy(); xp[:, j] = rng.permutation(xp[:, j])
            drops[n_num + j] += base_auc - roc_auc_score(y, predict_prob(model, xn, xp))
    return drops / n_repeats, base_auc


def occlusion_attributions(model, xn, xc, base_cat):
    base_prob = predict_prob(model, xn, xc)
    n_num, n_cat = xn.shape[1], xc.shape[1]
    attrs = np.zeros((xn.shape[0], n_num + n_cat), dtype=np.float32)
    for j in range(n_num):
        xo = xn.copy(); xo[:, j] = 0.0
        attrs[:, j] = base_prob - predict_prob(model, xo, xc)
    for j in range(n_cat):
        xo = xc.copy(); xo[:, j] = base_cat[j]
        attrs[:, n_num + j] = base_prob - predict_prob(model, xn, xo)
    return attrs


def integrated_gradients(model, xn, xc, base_cat, n_steps):
    model.eval()
    n, n_num, n_cat = xn.shape[0], xn.shape[1], xc.shape[1]
    xn_t = torch.tensor(xn, dtype=torch.float32, device=DEVICE)
    xc_t = torch.tensor(xc, dtype=torch.long, device=DEVICE)
    base_t = torch.tensor(base_cat, dtype=torch.long, device=DEVICE)
    acc_num = torch.zeros_like(xn_t)
    acc_cat = torch.zeros((n, n_cat), device=DEVICE)
    for step in range(1, n_steps + 1):
        a = step / n_steps
        xi = (a * xn_t).detach().requires_grad_(True)
        ca = torch.full((n, n_cat), a, device=DEVICE, requires_grad=True)
        logit, _ = model(xi, xc_t, cat_alpha=ca, cat_base=base_t)
        g_num, g_cat = torch.autograd.grad(torch.sigmoid(logit).sum(), [xi, ca])
        acc_num += g_num; acc_cat += g_cat
    ig = torch.cat([acc_num / n_steps * xn_t, acc_cat / n_steps], dim=1).detach().cpu().numpy()
    f_x = predict_prob(model, xn, xc)
    f_b = float(predict_prob(model, np.zeros((1, n_num), dtype=np.float32), base_cat[None, :])[0])
    comp_err = float(np.mean(np.abs(ig.sum(1) - (f_x - f_b))))
    return ig.astype(np.float32), f_b, comp_err


def true_shap_values(model, xn_tr, xc_tr, xn_te, xc_te, seed, n_explain=100, n_bg=20, n_perm=5):
    import shap
    n_num = xn_tr.shape[1]
    rng = np.random.RandomState(seed)
    bg = rng.choice(len(xn_tr), size=min(n_bg, len(xn_tr)), replace=False)
    sub = rng.choice(len(xn_te), size=min(n_explain, len(xn_te)), replace=False)
    Zbg = np.concatenate([xn_tr[bg], xc_tr[bg].astype(np.float32)], 1)
    Zte = np.concatenate([xn_te[sub], xc_te[sub].astype(np.float32)], 1)

    def f(Z):
        Z = np.asarray(Z)
        return predict_prob(model, Z[:, :n_num].astype(np.float32), np.rint(Z[:, n_num:]).astype(np.int64))

    masker = shap.maskers.Independent(Zbg, max_samples=n_bg)
    try:
        expl = shap.explainers.Permutation(f, masker, seed=seed)
    except TypeError:
        expl = shap.explainers.Permutation(f, masker)
    sv = expl(Zte, max_evals=2 * Zte.shape[1] * n_perm + 1, silent=True)
    return np.asarray(sv.values, dtype=np.float32), float(np.mean(sv.base_values)), sub


def _override_probs(m, xn_t, xc_t, override):
    m.gate_override = override
    try:
        m.eval()
        with torch.no_grad():
            return torch.sigmoid(m(xn_t, xc_t)[0]).cpu().numpy()
    finally:
        m.gate_override = None


def gate_analysis(members, xn_tr, xc_tr, xn_te, xc_te, y_te, seed):
    f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    i64 = lambda a: torch.tensor(a, dtype=torch.long, device=DEVICE)
    xn_te_t, xc_te_t, xn_tr_t, xc_tr_t = f32(xn_te), i64(xc_te), f32(xn_tr), i64(xc_tr)
    n_te = xn_te.shape[0]
    stats = {k: [] for k in ["mean_feat", "emp_sd", "dim_sd", "norm_ratio"]}
    hist = np.zeros(50, dtype=np.int64)
    scal = {k: [] for k in ["mean", "sd", "min", "max", "frac_045_055", "mean_abs_dev"]}
    modes = ["learned", "mean_element", "mean_token", "const_0.5", "shuffled"]
    mode_probs = {k: [] for k in modes}
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_te, generator=gen).to(DEVICE)
    for m in members:
        tok, g = m.gate_diagnostics_tensors(xn_te_t, xc_te_t)
        _, g_tr = m.gate_diagnostics_tensors(xn_tr_t, xc_tr_t)
        gf = g[:, 1:, :]
        stats["mean_feat"].append(gf.mean((0, 2)).cpu().numpy())
        stats["emp_sd"].append(gf.mean(2).std(0).cpu().numpy())
        stats["dim_sd"].append(gf.mean(0).std(1).cpu().numpy())
        ratio = ((tok * g).norm(dim=-1) / tok.norm(dim=-1).clamp_min(1e-8))[:, 1:].mean(0)
        stats["norm_ratio"].append(ratio.cpu().numpy())
        v = gf.flatten().cpu().numpy()
        hist += np.histogram(v, bins=50, range=(0, 1))[0]
        scal["mean"].append(float(v.mean())); scal["sd"].append(float(v.std()))
        scal["min"].append(float(v.min())); scal["max"].append(float(v.max()))
        scal["frac_045_055"].append(float(np.mean((v >= 0.45) & (v <= 0.55))))
        scal["mean_abs_dev"].append(float(np.mean(np.abs(v - 0.5))))
        overrides = {
            "learned": None,
            "mean_element": g_tr.mean(0, keepdim=True),
            "mean_token": g_tr.mean((0, 2), keepdim=True),
            "const_0.5": torch.full((1, 1, 1), 0.5, device=DEVICE),
            "shuffled": g[perm],
        }
        for k in modes:
            mode_probs[k].append(_override_probs(m, xn_te_t, xc_te_t, overrides[k]))
    base_p = np.mean(mode_probs["learned"], 0)
    interv = {}
    for k in modes:
        p = np.mean(mode_probs[k], 0)
        interv[k] = dict(auc=float(roc_auc_score(y_te, p)), mad=float(np.mean(np.abs(p - base_p))))
    return dict(mean_feat=np.mean(stats["mean_feat"], 0), emp_sd=np.mean(stats["emp_sd"], 0),
                dim_sd=np.mean(stats["dim_sd"], 0), norm_ratio=np.mean(stats["norm_ratio"], 0),
                hist=hist, scalars={k: float(np.mean(v)) for k, v in scal.items()}, interventions=interv)


def init_gate_histogram(D, best_cfg):
    sc = StandardScaler().fit(D["x_num_raw"])
    xn = torch.tensor(sc.transform(D["x_num_raw"]), dtype=torch.float32, device=DEVICE)
    xc = torch.tensor(D["x_cat_full"], dtype=torch.long, device=DEVICE)
    hist = np.zeros(50, dtype=np.int64); vals = []
    for s in range(N_ENSEMBLE):
        seed_everything(SEED + s)
        m = ca_builder(D["n_num"], D["cat_card"], best_cfg, "full")().to(DEVICE)
        _, g = m.gate_diagnostics_tensors(xn, xc)
        v = g[:, 1:, :].flatten().cpu().numpy()
        hist += np.histogram(v, bins=50, range=(0, 1))[0]; vals.append(v)
    v = np.concatenate(vals)
    return dict(hist=hist, mean=float(v.mean()), sd=float(v.std()))


def _ckpt_path(name):
    return os.path.join(CKPT_DIR, f"{name}.pkl")


def ckpt_save(name, obj):
    os.makedirs(CKPT_DIR, exist_ok=True)
    tmp = _ckpt_path(name) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, _ckpt_path(name))


def ckpt_load(name):
    p = _ckpt_path(name)
    if os.path.isfile(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def setup_checkpoint_dir(signature):
    sig_path = os.path.join(CKPT_DIR, "signature.txt")
    if RESET_CHECKPOINTS and WORKER is None and os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
    if os.path.isdir(CKPT_DIR):
        old = open(sig_path).read() if os.path.isfile(sig_path) else None
        if old != signature:
            print("[ckpt] settings changed since the cached run -> cache cleared")
            shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)
    open(sig_path, "w").write(signature)


def run_fold(fold_i, train_idx, test_idx, D, best_cfg, gbdt_cfg, n_splits_total):
    x_num_raw, x_cat_full, y = D["x_num_raw"], D["x_cat_full"], D["y"]
    n_num, cat_card = D["n_num"], D["cat_card"]
    seed_everything(SEED + fold_i)
    tr_idx, va_idx = train_test_split(train_idx, test_size=VAL_FRACTION, stratify=y[train_idx],
                                      random_state=SEED + fold_i)
    sanity_check_no_leakage(tr_idx, va_idx, test_idx)
    scaler = StandardScaler()
    xn_tr = scaler.fit_transform(x_num_raw[tr_idx])
    xn_va, xn_te = scaler.transform(x_num_raw[va_idx]), scaler.transform(x_num_raw[test_idx])
    xc_tr, xc_va, xc_te = x_cat_full[tr_idx], x_cat_full[va_idx], x_cat_full[test_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[test_idx]
    alpha_focal = 1 - y_tr.mean()
    loader = make_loader(xn_tr, xc_tr, y_tr)
    val3 = (xn_va, xc_va, y_va)

    R = dict(fold=fold_i, test_idx=test_idx.astype(np.int32), y_te=y_te.astype(np.int8),
             n_train=int(len(train_idx)), n_test=int(len(test_idx)), models={})

    def record(name, va_prob, te_prob, secs, **extra):
        thr, _ = find_best_threshold(y_va, va_prob)
        m, _ = evaluate_from_prob(y_te, te_prob, thr)
        m05, _ = evaluate_from_prob(y_te, te_prob, 0.5)
        R["models"][name] = dict(metrics=m, metrics05=m05, prob=np.asarray(te_prob, dtype=np.float32),
                                 thr=float(thr), secs=float(secs), **extra)

    def train_net(build, seed, kw):
        seed_everything(seed)
        t0 = time.perf_counter()
        model = build().to(DEVICE)
        model, _ = train_one_model(model, loader, val3, **kw)
        return model, time.perf_counter() - t0

    def probs(model):
        return predict_prob(model, xn_va, xc_va), predict_prob(model, xn_te, xc_te)

    
    kw_prop = tr_kwargs_from_cfg(best_cfg, alpha_focal)
    build_full = ca_builder(n_num, cat_card, best_cfg, "full")
    members, vps, tps, secs_l = [], [], [], []
    for m in range(N_ENSEMBLE):
        mod, s = train_net(build_full, SEED + fold_i * 100 + m, kw_prop)
        vp, tp = probs(mod)
        members.append(mod); vps.append(vp); tps.append(tp); secs_l.append(s)
    record(PROPOSED_NAME, np.mean(vps, 0), np.mean(tps, 0), sum(secs_l), member_probs=np.array(tps, dtype=np.float32))
    record(SINGLE_FULL, vps[0], tps[0], secs_l[0])

    if RUN_ABLATIONS:
        build_nog = ca_builder(n_num, cat_card, best_cfg, "none")
        nv, nt, ns = [], [], []
        for m in range(N_ENSEMBLE):
            mod, s = train_net(build_nog, SEED + fold_i * 100 + m, kw_prop)
            vp, tp = probs(mod); nv.append(vp); nt.append(tp); ns.append(s)
        record(NOGATE_MIX, np.mean(nv, 0), np.mean(nt, 0), sum(ns))
        record("w/o gate", nv[0], nt[0], ns[0])
        for name, spec in ABLATION_SPECS:
            if name == "w/o gate":
                continue
            kw, build = dict(kw_prop), build_full
            if "gate_type" in spec:
                build = ca_builder(n_num, cat_card, best_cfg, spec["gate_type"])
            if "gamma" in spec:
                kw["gamma_focal"] = spec["gamma"]
            if "ls" in spec:
                kw["label_smoothing"] = spec["ls"]
            if "alpha" in spec:
                kw["alpha_focal"] = spec["alpha"]
            if spec.get("default_hp"):
                build = lambda: CA_FTX_Plus(n_num, cat_card)
                kw = baseline_tr_kwargs(alpha_focal); kw["is_ca_ftx"] = True
            mod, s = train_net(build, SEED + fold_i * 100, kw)
            record(name, *probs(mod), s)

    base_models = {}
    for name, cls in BASELINE_REGISTRY.items():
        mod, s = train_net(lambda cls=cls: cls(n_num, cat_card), SEED + fold_i, baseline_tr_kwargs(alpha_focal))
        record(name, *probs(mod), s)
        base_models[name] = (mod, s)

    if RUN_ABLATIONS:   
        pname = "Feature Transformer (plain, no reg.)"
        mem_v, mem_t = [probs(base_models[pname][0])[0]], [probs(base_models[pname][0])[1]]
        s_tot = base_models[pname][1]
        for m in range(1, N_ENSEMBLE):
            mod, s = train_net(lambda: PlainFeatureTransformer(n_num, cat_card), SEED + fold_i + 7919 * m,
                               baseline_tr_kwargs(alpha_focal))
            vp, tp = probs(mod); mem_v.append(vp); mem_t.append(tp); s_tot += s
        record(FT_MIX, np.mean(mem_v, 0), np.mean(mem_t, 0), s_tot)

    if RUN_GBDT and gbdt_cfg is not None:
        Xtr, Xva, Xte = (make_df(x_num_raw[tr_idx], xc_tr, D), make_df(x_num_raw[va_idx], xc_va, D),
                         make_df(x_num_raw[test_idx], xc_te, D))
        cat_idx = list(range(n_num, n_num + len(D["cat_cols"])))
        for lib in GBDT_LIBS:
            for nm, params in [(lib, gbdt_cfg["tuned"][lib])]:
                try:
                    mod, secs, n_trees = fit_gbdt(lib, params, Xtr, y_tr, Xva, y_va, cat_idx, SEED + fold_i)
                    vp, tp = mod.predict_proba(Xva)[:, 1], mod.predict_proba(Xte)[:, 1]
                    t0 = time.perf_counter()
                    for _ in range(5):
                        mod.predict_proba(Xte.iloc[:64])
                    infer_ms = (time.perf_counter() - t0) / 5 * 1000
                    record(nm, vp, tp, secs, n_trees=n_trees, infer_ms=infer_ms)
                except Exception as e:
                    print(f"   [warn] {nm} failed in fold {fold_i}: {repr(e)[:150]}")
        oh = lambda xc: np.concatenate([np.eye(c)[xc[:, j]] for j, c in enumerate(cat_card)], 1)
        t0 = time.perf_counter()
        lr_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        lr_model.fit(np.concatenate([xn_tr, oh(xc_tr)], 1), y_tr)
        lr_p = lambda xn, xc: lr_model.predict_proba(np.concatenate([xn, oh(xc)], 1))[:, 1]
        record(LR_NAME, lr_p(xn_va, xc_va), lr_p(xn_te, xc_te), time.perf_counter() - t0)

    mix = MixtureModel(members).to(DEVICE).eval()
    base_cat = baseline_cat_values(xc_tr)
    perm_drop, base_auc = permutation_importance(mix, xn_te, xc_te, y_te, N_PERM_REPEATS, SEED + fold_i)
    occ = occlusion_attributions(mix, xn_te, xc_te, base_cat)
    ig, f_base, comp_err = integrated_gradients(mix, xn_te, xc_te, base_cat, IG_STEPS)
    R["explain"] = dict(perm=perm_drop.astype(np.float32), base_auc=float(base_auc), occ=occ, ig=ig,
                        f_base=f_base, ig_completeness_err=comp_err,
                        xn=xn_te.astype(np.float32), xc=xc_te.astype(np.int16))
    if RUN_TRUE_SHAP and fold_i == n_splits_total - 1:
        try:
            sv, sbase, sub = true_shap_values(mix, xn_tr, xc_tr, xn_te, xc_te, SEED)
            R["shap"] = dict(values=sv, base=sbase, rows=sub)
        except Exception as e:
            print("   [warn] SHAP failed:", repr(e)[:200])
    R["gate"] = gate_analysis(members, xn_tr, xc_tr, xn_te, xc_te, y_te, SEED + fold_i)

    if RUN_NESTED_PSO and fold_i < NESTED_PSO_REPEATS * N_FOLDS:
        cfg_n, f1_n, _, _ = pso_on_indices(train_idx, D, SEED + 1000 + fold_i, verbose=False,
                                           n_particles=NESTED_PSO_PARTICLES, n_iters=NESTED_PSO_ITERS)
        kw_n = tr_kwargs_from_cfg(cfg_n, alpha_focal)
        b_n = ca_builder(n_num, cat_card, cfg_n, "full")
        nv, nt, ns = [], [], 0
        for m in range(N_ENSEMBLE):
            mod, s = train_net(b_n, SEED + fold_i * 100 + m, kw_n)
            vp, tp = probs(mod); nv.append(vp); nt.append(tp); ns += s
        record(NESTED_NAME, np.mean(nv, 0), np.mean(nt, 0), ns, cfg=cfg_n, inner_val_f1=float(f1_n))
        print(f"   nested PSO (fold {fold_i + 1}): inner-val F1={f1_n:.4f} cfg arch="
              f"{(cfg_n['d_token'], cfg_n['n_heads'], cfg_n['n_layers'])} lr={cfg_n['lr']:.2e}")
    return R


def _spawn_workers(n_workers, n_total):
    procs = []
    for k in range(n_workers):
        lf = open(os.path.join(LOG_DIR, f"worker{k}.log"), "w")
        procs.append((subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(k), str(n_workers)],
                                       stdout=lf, stderr=subprocess.STDOUT), lf, k))
    t0 = time.time()
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(30)
        done = len(glob.glob(os.path.join(CKPT_DIR, "fold_*.pkl")))
        print(f"   ... {done}/{n_total} folds finished, {(time.time() - t0) / 60:.0f} min elapsed", flush=True)
    for p, lf, k in procs:
        lf.close()
        if p.returncode != 0:
            print(f"[warning] worker {k} exited with code {p.returncode}; last log lines:")
            print("".join(open(os.path.join(LOG_DIR, f"worker{k}.log")).readlines()[-8:]))


def run_all_folds(D, best_cfg, gbdt_cfg, shard=None):
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rskf.split(D["x_num_raw"], D["y"]))
    if shard is None and N_WORKERS > 1:
        try:
            todo = [i for i in range(len(splits)) if ckpt_load(f"fold_{i:02d}") is None]
            if todo:
                print(f"Running {len(todo)} folds on {N_WORKERS} parallel worker processes ...", flush=True)
                _spawn_workers(N_WORKERS, len(splits))
        except NameError:       
            print("[info] running sequentially (no __file__)")
    results, times = [], []
    for fold_i, (tr, te) in enumerate(splits):
        if shard is not None and fold_i % shard[1] != shard[0]:
            continue
        cached = ckpt_load(f"fold_{fold_i:02d}")
        if cached is not None:
            if shard is None:
                results.append(cached)
            continue
        t0 = time.perf_counter()
        R = run_fold(fold_i, tr, te, D, best_cfg, gbdt_cfg, len(splits))
        ckpt_save(f"fold_{fold_i:02d}", R)
        if shard is None:
            results.append(R)
        times.append(time.perf_counter() - t0)
        print(f"Fold {fold_i + 1}/{len(splits)} done in {times[-1] / 60:.1f} min | "
              f"{PROPOSED_NAME} F1={R['models'][PROPOSED_NAME]['metrics']['F1']:.4f} "
              f"AUC={R['models'][PROPOSED_NAME]['metrics']['AUC_ROC']:.4f}", flush=True)
    return results


METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC_ROC", "AUC_PR", "MCC", "BalancedAcc"]
PRETTY = {"AUC_ROC": "AUC-ROC", "AUC_PR": "AUC-PR", "BalancedAcc": "Bal. Acc."}


def pm(m):
    return PRETTY.get(m, m)


def M(results, name, metric, key="metrics"):
    return np.array([r["models"][name][key][metric] for r in results], dtype=float)


def names_in_all(results):
    s = set(results[0]["models"])
    for r in results[1:]:
        s &= set(r["models"])
    return s


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a - b, 0):
        return np.nan, np.nan
    try:
        s, p = wilcoxon(a, b)
        return float(s), float(p)
    except Exception:
        return np.nan, np.nan


def holm(p):
    p = np.asarray(p, float)
    q = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(q)
    adj = np.empty(len(q)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(q) - rank) * q[i]); adj[i] = min(1.0, run)
    adj[np.isnan(p)] = np.nan
    return adj


def nadeau_bengio_p(diff, n_test, n_train):
    diff = np.asarray(diff, float); k = len(diff)
    v = diff.var(ddof=1) if k > 1 else 0.0
    if v <= 0 or k < 3:
        return np.nan
    t = diff.mean() / math.sqrt((1.0 / k + n_test / n_train) * v)
    return float(2 * sps.t.sf(abs(t), k - 1))


def dz(diff):
    diff = np.asarray(diff, float)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12)) if len(diff) > 1 else np.nan


def boot_ci(vals, rng, B=None, ci=0.95):
    B = B or BOOTSTRAP_RESAMPLES
    vals = np.asarray(vals, float)
    idx = rng.randint(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(1)
    return np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])


def stars(p):
    if p is None or pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fmt_p(p):
    if pd.isna(p):
        return "n/a"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman_nemenyi(results, metric, names):
    vals = np.array([[r["models"][n]["metrics"][metric] for n in names] for r in results])
    nf, k = vals.shape
    ranks = np.array([rankdata(-vals[f, :]) for f in range(nf)])
    avg = pd.Series(ranks.mean(0), index=names)
    try:
        st, p = friedmanchisquare(*[vals[:, i] for i in range(k)])
    except Exception:
        st, p = np.nan, np.nan
    cd = NEMENYI_Q05.get(k, NEMENYI_Q05[10]) * math.sqrt(k * (k + 1) / (6.0 * nf))
    return avg, float(st), float(p), float(cd), nf, k


def table_main(results, main_names):
    rng = np.random.RandomState(SEED)
    mean_df = pd.DataFrame({n: {m: M(results, n, m).mean() for m in METRICS} for n in main_names}).T
    std_df = pd.DataFrame({n: {m: M(results, n, m).std(ddof=1) for m in METRICS} for n in main_names}).T
    save_table(mean_df, "T01a_mean_metrics")
    save_table(std_df, "T01b_std_metrics")
    cells = pd.DataFrame(index=main_names, columns=[pm(m) for m in METRICS], dtype=object)
    best = {m: mean_df[m].idxmax() for m in METRICS}
    for m in METRICS:
        pv = {n: wil(M(results, PROPOSED_NAME, m), M(results, n, m))[1] for n in main_names if n != PROPOSED_NAME}
        adj = dict(zip(pv.keys(), holm(list(pv.values()))))
        for n in main_names:
            s = "" if n == PROPOSED_NAME else stars(adj[n])
            cells.loc[n, pm(m)] = f"{mean_df.loc[n, m]:.4f} ± {std_df.loc[n, m]:.4f}{s}"
    cells.index.name = "Model"
    save_table(cells, "T01_main_results")
    lines = [r"\begin{table}[ht]", r"\centering", r"\small", r"\begin{tabular}{l" + "r" * len(METRICS) + "}", r"\toprule",
             " & ".join(["Model"] + [_tex_escape(pm(m)) for m in METRICS]) + r" \\", r"\midrule"]
    for n in main_names:
        row = []
        for m in METRICS:
            c = _tex_escape(cells.loc[n, pm(m)]).replace("*", r"$^{*}$")
            row.append(r"\textbf{%s}" % c if best[m] == n else c)
        lines.append(" & ".join([_tex_escape(n)] + row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Mean $\pm$ SD over 25 test folds (5$\times$5 repeated stratified CV). Stars: paired Wilcoxon test vs.\ "
              r"CA-FTX+, Holm-adjusted over all baselines ($^{*}$p$<$0.05, $^{**}$p$<$0.01, $^{***}$p$<$0.001). Best mean per column in bold.}",
              r"\end{table}"]
    open(os.path.join(TAB_DIR, "T01_main_results.tex"), "w", encoding="utf-8").write("\n".join(lines))
    return mean_df, std_df


def table_stats(results, main_names):
    n_te, n_tr = np.mean([r["n_test"] for r in results]), np.mean([r["n_train"] for r in results])
    rows = []
    for m in ["F1", "AUC_ROC", "AUC_PR", "MCC"]:
        sub = []
        for n in main_names:
            if n == PROPOSED_NAME:
                continue
            a, b = M(results, PROPOSED_NAME, m), M(results, n, m)
            st, p = wil(a, b)
            sub.append(dict(Metric=pm(m), Baseline=n, Proposed_mean=a.mean(), Baseline_mean=b.mean(),
                            Mean_diff=(a - b).mean(), Wilcoxon_stat=st, p_wilcoxon=p,
                            p_nadeau_bengio=nadeau_bengio_p(a - b, n_te, n_tr), Cohen_dz=dz(a - b)))
        adj = holm([s["p_wilcoxon"] for s in sub])
        adj_nb = holm([s["p_nadeau_bengio"] for s in sub])
        for s, a1, a2 in zip(sub, adj, adj_nb):
            s["p_wilcoxon_Holm"], s["p_NB_Holm"] = a1, a2
            s["Significant_Holm_p<0.05"] = bool(a1 < 0.05) if pd.notna(a1) else False
            s["Significant_NB_Holm_p<0.05"] = bool(a2 < 0.05) if pd.notna(a2) else False
            rows.append(s)
    df = pd.DataFrame(rows).set_index(["Metric", "Baseline"])
    save_table(df, "T02_statistical_tests")
    return df


def table_friedman(results, main_names):
    rank_rows, summ = {}, []
    for m in ["F1", "AUC_ROC"]:
        avg, st, p, cd, nf, k = friedman_nemenyi(results, m, main_names)
        rank_rows[pm(m)] = avg
        summ.append(dict(Metric=pm(m), Friedman_chi2=st, p_value=p, n_models=k, n_folds=nf, Nemenyi_CD=cd,
                         **{f"rank_{sh(n)}": avg[n] for n in main_names}))
    ranks = pd.DataFrame(rank_rows)
    ranks.index.name = "Model"
    save_table(ranks.sort_values(pm("F1")), "T03a_average_ranks")
    save_table(pd.DataFrame(summ).set_index("Metric"), "T03b_friedman_nemenyi")
    return {m: friedman_nemenyi(results, m, main_names) for m in ["F1", "AUC_ROC"]}


def table_bootstrap(results, main_names):
    rng = np.random.RandomState(SEED)
    cols = pd.MultiIndex.from_product([["F1", "AUC_ROC", "AUC_PR", "Precision", "Recall", "MCC"], ["mean", "CI_low", "CI_high"]])
    out = pd.DataFrame(index=main_names, columns=cols, dtype=float)
    for n in main_names:
        for m in ["F1", "AUC_ROC", "AUC_PR", "Precision", "Recall", "MCC"]:
            v = M(results, n, m); lo, hi = boot_ci(v, rng)
            out.loc[n, (m, "mean")], out.loc[n, (m, "CI_low")], out.loc[n, (m, "CI_high")] = v.mean(), lo, hi
    save_table(out, "T16_bootstrap_ci")
    return out


def table_ablation(results, names):
    n_te, n_tr = np.mean([r["n_test"] for r in results]), np.mean([r["n_train"] for r in results])
    rng = np.random.RandomState(SEED)
    rows = []

    def add_group(group, ref, variants):
        sub = []
        for v in [ref] + variants:
            if v not in names:
                continue
            d = dict(Group=group, Variant=v)
            for m in ["F1", "AUC_ROC", "AUC_PR", "Precision", "Recall", "MCC"]:
                x = M(results, v, m); d[pm(m)] = f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"
            if v != ref:
                for m in ["F1", "AUC_ROC"]:
                    diff = M(results, v, m) - M(results, ref, m)
                    lo, hi = boot_ci(diff, rng)
                    d[f"Δ{pm(m)}"], d[f"Δ{pm(m)}_CI_low"], d[f"Δ{pm(m)}_CI_high"] = diff.mean(), lo, hi
                    d[f"p_{pm(m)}"] = wil(M(results, v, m), M(results, ref, m))[1]
                    d[f"p_NB_{pm(m)}"] = nadeau_bengio_p(diff, n_te, n_tr)
                    d[f"dz_{pm(m)}"] = dz(diff)
            sub.append(d)
        for m in ["F1", "AUC_ROC"]:
            ps = [s.get(f"p_{pm(m)}", np.nan) for s in sub[1:]]
            adj = holm(ps) if ps else []
            for s, a in zip(sub[1:], adj):
                s[f"p_Holm_{pm(m)}"] = a
        rows.extend(sub)

    add_group("System level (reference = full CA-FTX+ mixture)", PROPOSED_NAME, [NOGATE_MIX, FT_MIX])
    add_group("Single network (reference = full CA-FTX+, member 1)", SINGLE_FULL, ABLATION_NAMES)
    df = pd.DataFrame(rows).set_index(["Group", "Variant"])
    save_table(df, "T04_detailed_ablation")
    return df


def table_ladder(results, names):
    steps = [("1. Plain FT-Transformer (untuned)", "Feature Transformer (plain, no reg.)"),
             ("2. + PSO-tuned architecture/training, no gate (single net)", "w/o gate"),
             ("3. + Gated Feature Interaction (single net)", SINGLE_FULL),
             ("4. + 3-member mixture  = CA-FTX+", PROPOSED_NAME)]
    steps = [(a, b) for a, b in steps if b in names]
    rows, prev = [], None
    for lab, n in steps:
        row = {"Step": lab, "Model": n}
        for m in ["F1", "AUC_ROC", "AUC_PR", "MCC"]:
            x = M(results, n, m); row[pm(m)] = x.mean()
            if prev is not None:
                row[f"Δ{pm(m)} vs previous step"] = (x - M(results, prev, m)).mean()
                row[f"p_{pm(m)}"] = wil(x, M(results, prev, m))[1]
        rows.append(row); prev = n
    df = pd.DataFrame(rows).set_index("Step")
    save_table(df, "T05_component_ladder")
    return df


def table_gate(results):
    G = [r["gate"] for r in results]
    modes = ["learned", "mean_element", "mean_token", "const_0.5", "shuffled"]
    label = {"learned": "Learned gates (as trained)", "mean_element": "Population-mean gate per token & dimension",
             "mean_token": "Population-mean gate per token (scalar)", "const_0.5": "Constant 0.5",
             "shuffled": "Gates shuffled across employees"}
    auc = {m: np.array([g["interventions"][m]["auc"] for g in G]) for m in modes}
    rows = []
    for m in modes:
        d = auc[m] - auc["learned"]
        rows.append({"Test-time gate": label[m], "AUC-ROC (mean ± SD)": f"{auc[m].mean():.4f} ± {auc[m].std(ddof=1):.4f}",
                     "ΔAUC vs learned": d.mean() if m != "learned" else np.nan,
                     "p (Wilcoxon)": wil(auc[m], auc["learned"])[1] if m != "learned" else np.nan,
                     "Mean |Δ probability|": np.mean([g["interventions"][m]["mad"] for g in G])})
    adj = holm([r["p (Wilcoxon)"] for r in rows[1:]])
    for r, a in zip(rows[1:], adj):
        r["p (Holm)"] = a
    df = pd.DataFrame(rows).set_index("Test-time gate")
    save_table(df, "T06_gate_interventions")
    return df, auc


def table_gate_stats(results, D):
    feats = D["feature_names"]
    G = [r["gate"] for r in results]
    df = pd.DataFrame({
        "Mean_gate": np.mean([g["mean_feat"] for g in G], 0),
        "SD_of_mean_gate_across_folds": np.std([g["mean_feat"] for g in G], 0, ddof=1) if len(G) > 1 else np.nan,
        "SD_across_employees": np.mean([g["emp_sd"] for g in G], 0),
        "SD_across_dimensions": np.mean([g["dim_sd"] for g in G], 0),
        "Token_norm_ratio": np.mean([g["norm_ratio"] for g in G], 0),
    }, index=feats)
    df.index.name = "Feature"
    save_table(df.sort_values("SD_across_employees", ascending=False), "T07a_gate_statistics_per_feature")
    sc = pd.DataFrame([g["scalars"] for g in G])
    summ = pd.DataFrame({"mean": sc.mean(), "sd_across_folds": sc.std(ddof=1) if len(sc) > 1 else np.nan})
    summ.index.name = "Statistic of raw element-wise gate values (test sets)"
    save_table(summ, "T07b_gate_value_summary")
    return df, summ


def table_hyperparameters(D, best_cfg, gbdt_cfg, comp_df, pso_source, pso_trace):
    soft = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}
    for lib in ["xgboost", "lightgbm", "catboost", "shap"]:
        try:
            soft[lib] = __import__(lib).__version__
        except Exception:
            soft[lib] = "not installed"
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    n_cat = len(D["cat_cols"]); T = 1 + D["n_num"] + n_cat
    R = []
    add = lambda g, i, v: R.append((g, i, v))
    add("Data & protocol", "Dataset", f"IBM HR Analytics Employee Attrition, {len(D['y'])} rows, {D['n_num'] + n_cat} model inputs ({D['n_num']} numeric, {n_cat} categorical); positives (Attrition=Yes) = {int(D['y'].sum())} ({D['y'].mean() * 100:.1f}%)")
    add("Data & protocol", "Dropped columns", ", ".join(DROP_COLS) + " (constant / identifier)")
    add("Data & protocol", "Outer evaluation", f"RepeatedStratifiedKFold, {N_FOLDS} folds x {N_REPEATS} repeats = {N_FOLDS * N_REPEATS} test folds, random_state={SEED}")
    add("Data & protocol", "Validation split", f"{int(VAL_FRACTION * 100)}% of each outer-training part (stratified, random_state={SEED}+fold); used for early stopping and threshold selection only")
    add("Data & protocol", "Numeric scaling", "StandardScaler fitted on the inner-training rows of each fold only")
    add("Data & protocol", "Categorical encoding", "OrdinalEncoder codes -> learned embedding index (category vocabulary of each column fixed from the data schema; label-free)")
    add("Data & protocol", "Decision threshold", "Chosen per model and fold on the validation split: grid 0.05-0.95 (step 0.005) maximising F1; applied unchanged to the test fold")
    add("Data & protocol", "Random seeds", f"global SEED={SEED}; fold f uses SEED+f; mixture member m of fold f uses SEED+100f+m; deterministic algorithms requested")
    add("CA-FTX+ architecture (selected)", "Token width d", best_cfg["d_token"])
    add("CA-FTX+ architecture (selected)", "Attention heads", best_cfg["n_heads"])
    add("CA-FTX+ architecture (selected)", "Encoder blocks", best_cfg["n_layers"])
    add("CA-FTX+ architecture (selected)", "Dropout (residual, FFN, gate MLP, head)", f"{best_cfg['dropout']:.4f}")
    add("CA-FTX+ architecture (selected)", "Token dropout (after gate)", f"{best_cfg['dropout'] * 0.5:.4f} (= dropout/2)")
    add("CA-FTX+ architecture (selected)", "Attention dropout", f"{best_cfg['attn_dropout']:.4f}")
    add("CA-FTX+ architecture (fixed)", "FFN width multiplier / activation", "2 x d, GELU")
    add("CA-FTX+ architecture (fixed)", "Normalisation", "pre-LayerNorm blocks + final LayerNorm")
    add("CA-FTX+ architecture (fixed)", "Tokeniser", f"numeric: x*w_j+b_j (U(-1/sqrt(d),1/sqrt(d))); categorical: nn.Embedding (N(0,0.02)); [CLS] token; sequence length {T}")
    add("CA-FTX+ architecture (fixed)", "Gated Feature Interaction", "c = Linear(d,d)(mean of tokens); g = sigmoid(Linear(d,d)(Dropout(GELU(Linear(2d,d)([t;c]))))); t' = t * g (element-wise, all tokens incl. [CLS])")
    add("CA-FTX+ architecture (fixed)", "Prediction head", "Linear(d,d) - GELU - Dropout - Linear(d,1) on the [CLS] output")
    if comp_df is not None and len(comp_df):
        for nm, r in comp_df.iterrows():
            if "[trees]" not in nm:
                add("Model size", f"{nm}: trainable parameters", int(r["Params_or_trees"]))
    add("Training (selected)", "Learning rate", f"{best_cfg['lr']:.6g}")
    add("Training (selected)", "Weight decay (AdamW)", f"{best_cfg['weight_decay']:.6g}")
    add("Training (selected)", "Focal gamma", f"{best_cfg['focal_gamma']:.4f}")
    add("Training (selected)", "Mixup alpha", f"{best_cfg['mixup_alpha']:.4f} (0 = off)")
    add("Training (fixed)", "Optimiser / scheduler", f"AdamW (betas 0.9/0.999); CosineAnnealingWarmRestarts T_0=max(10, epochs//6)={max(10, EPOCHS // 6)}, T_mult=1")
    add("Training (fixed)", "Epochs / batch size / patience", f"max {EPOCHS} / {BATCH_SIZE} / {PATIENCE} (early stopping on validation AUC-ROC, best epoch restored)")
    add("Training (fixed)", "Gradient clipping", "global norm 2.0")
    add("Training (fixed)", "Loss", f"focal loss, alpha = 1 - prevalence of the training rows, label smoothing eps={LABEL_SMOOTH} (target y(1-eps)+0.5eps)")
    add("Training (fixed)", "Mixture", f"{N_ENSEMBLE} networks, different seeds, probabilities averaged")
    add("PSO", "Swarm", f"{PSO_PARTICLES} particles x {PSO_ITERS} iterations = {PSO_PARTICLES * PSO_ITERS} evaluations; w={PSO_W}, c1={PSO_C1}, c2={PSO_C2}; positions clipped to bounds")
    add("PSO", "Particle encoding", f"{len(PSO_BOUNDS)} continuous coordinates decoded into 9 hyper-parameters; coordinate 'arch_idx' selects one of {len(ARCH_CHOICES)} (d, heads, blocks) triplets")
    add("PSO", "Fitness", f"validation-split F1 at the best threshold after a short training ({PSO_EPOCHS} epochs, patience {PSO_PATIENCE}, seed {SEED})")
    add("PSO", "Search data", "inner split: 20% of the rows held out, then 20% of the remaining 80% = inner validation (random_state=42). Search is run ONCE (not inside each outer fold)")
    add("PSO", "Source of the reported configuration", pso_source)
    add("PSO", "Nested sensitivity check", f"PSO repeated inside each outer training part of the first {NESTED_PSO_REPEATS * N_FOLDS} folds with a reduced budget ({NESTED_PSO_PARTICLES} particles x {NESTED_PSO_ITERS} iterations)")
    add("Neural baselines", "Residual MLP", "embeddings d=8 + 3 x (Linear(128) - BatchNorm - GELU - Dropout 0.25), two residual blocks; lr 3e-4, wd 5e-3, gamma 2, no mixup")
    add("Neural baselines", "Plain FT-Transformer", "d=32, 1 block, 2 heads, FFN x2, dropout 0.1, attention dropout 0.1, no gate, head Linear(d,1); lr 3e-4, wd 5e-3, gamma 2")
    add("Neural baselines", "Wide-Shallow MLP", "embeddings d=8, Linear(256) - BatchNorm - GELU - Dropout 0.3 - Linear(1); lr 3e-4, wd 5e-3, gamma 2")
    add("Neural baselines", "Shared training recipe", f"AdamW, same epochs/batch/patience/clipping/loss as CA-FTX+, no mixup, same validation-based threshold")
    if gbdt_cfg is not None:
        for lib in GBDT_LIBS:
            add("Tree ensembles", f"{lib} (tuned, selected)", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in gbdt_cfg["tuned"][lib].items()}))
        add("Tree ensembles", "Common settings", f"n_estimators={GBDT_FIXED['n_estimators']} with early stopping ({GBDT_FIXED['early_stopping_rounds']} rounds, validation AUC), class imbalance via scale_pos_weight=n_neg/n_pos, categorical codes as integers (CatBoost: native categorical), seed SEED+fold")
        add("Tree ensembles", "Tuning", f"random search, {GBDT_SEARCH_BUDGET} configurations per library (same budget as PSO), same inner split and objective (validation F1) as the PSO")
        add("Other baselines", "Logistic regression", "L2, C=1, class_weight='balanced', scaled numerics + one-hot categoricals")
    for k, v in soft.items():
        add("Software / hardware", k, v)
    add("Software / hardware", "Device", dev)
    df = pd.DataFrame(R, columns=["Group", "Hyper-parameter / setting", "Value"])
    save_table(df, "T08_hyperparameters_full", index=False)
    # search space
    rows = [("Architecture (d, heads, blocks)", "; ".join(str(a) for a in ARCH_CHOICES), "categorical (index)"),
            ("Dropout", f"{PSO_BOUNDS['dropout'][0]} - {PSO_BOUNDS['dropout'][1]}", "uniform"),
            ("Attention dropout", f"{PSO_BOUNDS['attn_dropout'][0]} - {PSO_BOUNDS['attn_dropout'][1]}", "uniform"),
            ("Learning rate", "1e-4 - 3e-3", "log10-uniform"),
            ("Weight decay", "1e-4 - 2e-2", "log10-uniform"),
            ("Focal gamma", f"{PSO_BOUNDS['focal_gamma'][0]} - {PSO_BOUNDS['focal_gamma'][1]}", "uniform"),
            ("Mixup alpha", f"{PSO_BOUNDS['mixup_alpha'][0]} - {PSO_BOUNDS['mixup_alpha'][1]}", "uniform")]
    sp = pd.DataFrame(rows, columns=["Hyper-parameter", "Range / choices", "Scale"])
    sel = {"Architecture (d, heads, blocks)": f"({best_cfg['d_token']}, {best_cfg['n_heads']}, {best_cfg['n_layers']})",
           "Dropout": f"{best_cfg['dropout']:.4f}", "Attention dropout": f"{best_cfg['attn_dropout']:.4f}",
           "Learning rate": f"{best_cfg['lr']:.3e}", "Weight decay": f"{best_cfg['weight_decay']:.3e}",
           "Focal gamma": f"{best_cfg['focal_gamma']:.3f}", "Mixup alpha": f"{best_cfg['mixup_alpha']:.3f}"}
    sp["Selected"] = sp["Hyper-parameter"].map(sel)
    save_table(sp, "T08b_pso_search_space", index=False)
    save_table(pso_trace.copy(), "T08c_pso_all_evaluations", index=False)
    if gbdt_cfg is not None:
        save_table(pd.DataFrame(gbdt_cfg["trace"]), "T08d_gbdt_random_search_trace", index=False)
    with open(os.path.join(TAB_DIR, "hyperparameters.json"), "w") as f:
        json.dump(dict(ca_ftx_plus=best_cfg, gbdt_tuned=(gbdt_cfg or {}).get("tuned"), gbdt_default=GBDT_DEFAULT,
                       protocol=dict(N_FOLDS=N_FOLDS, N_REPEATS=N_REPEATS, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE,
                                     PATIENCE=PATIENCE, VAL_FRACTION=VAL_FRACTION, LABEL_SMOOTH=LABEL_SMOOTH,
                                     N_ENSEMBLE=N_ENSEMBLE, SEED=SEED),
                       pso=dict(particles=PSO_PARTICLES, iters=PSO_ITERS, epochs=PSO_EPOCHS, patience=PSO_PATIENCE,
                                w=PSO_W, c1=PSO_C1, c2=PSO_C2), software=soft, device=dev), f, indent=2, default=str)
    return df


def complexity_table(D, best_cfg, results, main_names):
    n_num, cat_card = D["n_num"], D["cat_card"]
    rng = np.random.RandomState(SEED)
    xn = torch.tensor(rng.randn(64, n_num).astype(np.float32), device=DEVICE)
    xc = torch.tensor(np.stack([rng.randint(0, c, 64) for c in cat_card], 1).astype(np.int64), device=DEVICE)

    def timeit(model):
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                model(xn, xc)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                model(xn, xc)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 30 * 1000

    full = ca_builder(n_num, cat_card, best_cfg, "full")
    nog = ca_builder(n_num, cat_card, best_cfg, "none")
    builders = [("CA-FTX+ single network", full, PROPOSED_NAME), ("CA-FTX+ w/o gate (single network)", nog, "w/o gate" if "w/o gate" in names_in_all(results) else None)] + \
               [(n, (lambda c=c: c(n_num, cat_card)), n) for n, c in BASELINE_REGISTRY.items()]
    rows = []
    for name, b, rec in builders:
        seed_everything(SEED); mdl = b().to(DEVICE)
        ms = timeit(mdl)
        rows.append(dict(Model=name, Params_or_trees=sum(p.numel() for p in mdl.parameters() if p.requires_grad),
                         Inference_ms_per_batch64=ms, Inference_us_per_employee=ms * 1000 / 64,
                         Train_s_per_fold=np.mean([r["models"][rec]["secs"] / (N_ENSEMBLE if rec == PROPOSED_NAME else 1) for r in results]) if rec else np.nan,
                         Device=DEVICE.type.upper()))
    seed_everything(SEED)
    mix = MixtureModel([full().to(DEVICE) for _ in range(N_ENSEMBLE)])
    rows.append(dict(Model=PROPOSED_NAME + " [deployed model]", Params_or_trees=sum(p.numel() for p in mix.parameters()),
                     Inference_ms_per_batch64=timeit(mix), Inference_us_per_employee=timeit(mix) * 1000 / 64,
                     Train_s_per_fold=np.mean([r["models"][PROPOSED_NAME]["secs"] for r in results]), Device=DEVICE.type.upper()))
    for g in GBDT_LIBS:
        if g in names_in_all(results):
            rows.append(dict(Model=f"{g} (tuned) [trees]", Params_or_trees=float(np.median([r["models"][g]["n_trees"] for r in results])),
                             Inference_ms_per_batch64=np.median([r["models"][g]["infer_ms"] for r in results]),
                             Inference_us_per_employee=np.median([r["models"][g]["infer_ms"] for r in results]) * 1000 / 64,
                             Train_s_per_fold=np.mean([r["models"][g]["secs"] for r in results]), Device="CPU"))
    df = pd.DataFrame(rows).set_index("Model")
    save_table(df, "T09_efficiency")
    return df


def explain_tables(results, D):
    feats, n_num = D["feature_names"], D["n_num"]
    perm = np.array([r["explain"]["perm"] for r in results])
    occ_abs = np.array([np.abs(r["explain"]["occ"]).mean(0) for r in results])
    ig_abs = np.array([np.abs(r["explain"]["ig"]).mean(0) for r in results])
    ig_sgn = np.array([r["explain"]["ig"].mean(0) for r in results])
    imp = pd.DataFrame({"Perm_AUC_drop": perm.mean(0), "Perm_SD_across_folds": perm.std(0, ddof=1) if len(perm) > 1 else np.nan,
                        "Occlusion_mean_abs": occ_abs.mean(0), "IG_mean_abs": ig_abs.mean(0), "IG_signed_mean": ig_sgn.mean(0)},
                       index=feats)
    imp["Perm_rank"] = imp["Perm_AUC_drop"].rank(ascending=False)
    imp["Occlusion_rank"] = imp["Occlusion_mean_abs"].rank(ascending=False)
    imp["IG_rank"] = imp["IG_mean_abs"].rank(ascending=False)
    imp["Mean_rank"] = imp[["Perm_rank", "Occlusion_rank", "IG_rank"]].mean(1)
    last_shap = results[-1].get("shap")
    if last_shap is not None:
        sa = np.abs(last_shap["values"]).mean(0)
        imp["TrueSHAP_mean_abs_lastfold"] = sa
        imp["TrueSHAP_rank"] = pd.Series(sa, index=feats).rank(ascending=False)
    imp = imp.sort_values("Mean_rank")
    imp.index.name = "Feature"
    save_table(imp, "T10_feature_importance")
    pos = imp["Perm_AUC_drop"].clip(lower=0)
    summ = dict(top2_perm_share_of_positive=float(pos.nlargest(2).sum() / pos.sum()) if pos.sum() > 0 else np.nan,
                n_negative_perm=int((imp["Perm_AUC_drop"] < 0).sum()),
                ig_completeness_err_mean=float(np.mean([r["explain"]["ig_completeness_err"] for r in results])))
    rows = []
    pairs = [("Perm_AUC_drop", "Occlusion_mean_abs"), ("Perm_AUC_drop", "IG_mean_abs"), ("Occlusion_mean_abs", "IG_mean_abs")]
    if last_shap is not None:
        pairs += [("TrueSHAP_mean_abs_lastfold", "Occlusion_mean_abs"), ("TrueSHAP_mean_abs_lastfold", "IG_mean_abs"),
                  ("TrueSHAP_mean_abs_lastfold", "Perm_AUC_drop")]
    for a, b in pairs:
        rho, p = spearmanr(imp[a], imp[b]); r_, _ = pearsonr(imp[a], imp[b])
        ta, tb = set(imp[a].nlargest(10).index), set(imp[b].nlargest(10).index)
        rows.append({"Method A": a, "Method B": b, "Spearman rho": rho, "p": p, "Pearson r": r_, "Top-10 overlap": len(ta & tb)})
    ag = pd.DataFrame(rows).set_index(["Method A", "Method B"])
    save_table(ag, "T11_explanation_agreement")
    return imp, ag, summ


def table_confusion(results, main_names):
    rows = []
    for n in main_names:
        d = {"Model": n}
        for key, lab in [("metrics", "tuned thr"), ("metrics05", "thr 0.5")]:
            for c in ["TN", "FP", "FN", "TP"]:
                v = M(results, n, c, key); d[f"{c} ({lab})"] = f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"
        lm = results[-1]["models"][n]
        for lab, thr in [("tuned thr", lm["thr"]), ("thr 0.5", 0.5)]:
            tn, fp, fn, tp = confusion_matrix(results[-1]["y_te"], (lm["prob"] >= thr).astype(int), labels=[0, 1]).ravel()
            d[f"last fold {lab}: FP/FN"] = f"{fp}/{fn}"
        rows.append(d)
    df = pd.DataFrame(rows).set_index("Model")
    save_table(df, "T12_confusion_counts")
    return df


def table_extra(results, names):
    rows = []
    extras = [n for n in GBDT_DEFAULT_NAMES + [LR_NAME, FT_MIX, NOGATE_MIX] if n in names]
    for n in extras:
        d = {"Model": n}
        for m in ["F1", "AUC_ROC", "AUC_PR", "Precision", "Recall", "MCC"]:
            x = M(results, n, m); d[pm(m)] = f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"
        d["ΔF1 vs CA-FTX+"] = (M(results, n, "F1") - M(results, PROPOSED_NAME, "F1")).mean()
        d["ΔAUC vs CA-FTX+"] = (M(results, n, "AUC_ROC") - M(results, PROPOSED_NAME, "AUC_ROC")).mean()
        d["p (F1, Wilcoxon)"] = wil(M(results, n, "F1"), M(results, PROPOSED_NAME, "F1"))[1]
        d["p (AUC, Wilcoxon)"] = wil(M(results, n, "AUC_ROC"), M(results, PROPOSED_NAME, "AUC_ROC"))[1]
        rows.append(d)
    if rows:
        save_table(pd.DataFrame(rows).set_index("Model"), "T13_extra_baselines")


def table_threshold05(results, main_names):
    cells = pd.DataFrame(index=main_names, columns=[pm(m) for m in METRICS], dtype=object)
    for n in main_names:
        for m in METRICS:
            v = M(results, n, m, "metrics05"); cells.loc[n, pm(m)] = f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"
    cells.index.name = "Model"
    save_table(cells, "T15_metrics_at_fixed_threshold_0.5")


def table_nested(results):
    sub = [r for r in results if NESTED_NAME in r["models"]]
    if not sub:
        return None
    rows = []
    for r in sub:
        a, b = r["models"][PROPOSED_NAME]["metrics"], r["models"][NESTED_NAME]["metrics"]
        c = r["models"][NESTED_NAME]["cfg"]
        rows.append({"Fold": r["fold"] + 1, "F1 (single global PSO)": a["F1"], "F1 (nested PSO)": b["F1"],
                     "AUC-ROC (single global PSO)": a["AUC_ROC"], "AUC-ROC (nested PSO)": b["AUC_ROC"],
                     "Nested config (d,heads,blocks)": f"({c['d_token']},{c['n_heads']},{c['n_layers']})",
                     "Nested lr": c["lr"], "Nested dropout": c["dropout"], "Nested inner-val F1": r["models"][NESTED_NAME]["inner_val_f1"]})
    df = pd.DataFrame(rows).set_index("Fold")
    summ = pd.DataFrame([{"n_folds": len(sub),
                          "F1 single (mean)": df["F1 (single global PSO)"].mean(), "F1 nested (mean)": df["F1 (nested PSO)"].mean(),
                          "ΔF1 (nested - single)": (df["F1 (nested PSO)"] - df["F1 (single global PSO)"]).mean(),
                          "p (Wilcoxon, F1)": wil(df["F1 (nested PSO)"], df["F1 (single global PSO)"])[1],
                          "AUC single (mean)": df["AUC-ROC (single global PSO)"].mean(), "AUC nested (mean)": df["AUC-ROC (nested PSO)"].mean(),
                          "ΔAUC (nested - single)": (df["AUC-ROC (nested PSO)"] - df["AUC-ROC (single global PSO)"]).mean(),
                          "p (Wilcoxon, AUC)": wil(df["AUC-ROC (nested PSO)"], df["AUC-ROC (single global PSO)"])[1]}]).T
    summ.columns = ["value"]
    save_table(df, "T14a_nested_pso_per_fold")
    save_table(summ, "T14b_nested_pso_summary")
    return df, summ


def table_reproduction(results):
    pub = {PROPOSED_NAME: dict(Accuracy=.8646, Precision=.6133, Recall=.5661, F1=.5744, AUC_ROC=.8353, AUC_PR=.6422, MCC=.5050, BalancedAcc=.7440),
           "MLP (residual, regularized)": dict(Accuracy=.8366, Precision=.5152, Recall=.5302, F1=.5065, AUC_ROC=.8064, AUC_PR=.5654, MCC=.4209, BalancedAcc=.7129),
           "Feature Transformer (plain, no reg.)": dict(Accuracy=.8476, Precision=.5553, Recall=.5913, F1=.5576, AUC_ROC=.8260, AUC_PR=.6244, MCC=.4779, BalancedAcc=.7440),
           "Wide-Shallow MLP": dict(Accuracy=.8288, Precision=.4957, Recall=.5603, F1=.5122, AUC_ROC=.8091, AUC_PR=.5781, MCC=.4211, BalancedAcc=.7204)}
    rows = []
    for n, d in pub.items():
        for m, v in d.items():
            x = M(results, n, m).mean()
            rows.append(dict(Model=n, Metric=pm(m), Published_Table2=v, This_run=x, Difference=x - v))
    df = pd.DataFrame(rows).set_index(["Model", "Metric"])
    save_table(df, "T18_reproduction_check_vs_submitted_Table2")
    return df


def table_dataset(D, df):
    rows = []
    for c in D["num_cols"]:
        rows.append(dict(Feature=c, Type="numeric", Levels_or_range=f"{df[c].min()} to {df[c].max()}"))
    for j, c in enumerate(D["cat_cols"]):
        rows.append(dict(Feature=c, Type="categorical", Levels_or_range=f"{len(D['categories'][j])}: " + ", ".join(map(str, D["categories"][j]))))
    save_table(pd.DataFrame(rows), "T19_dataset_features", index=False)


def table_per_fold(results, names):
    rows = []
    for r in results:
        for n in names:
            d = {"Fold": r["fold"] + 1, "Model": n}; d.update({k: v for k, v in r["models"][n]["metrics"].items()})
            rows.append(d)
    save_table(pd.DataFrame(rows), "T17_per_fold_metrics", index=False)


def col(n):
    return MODEL_COLORS.get(n, "#7f8c8d")


def fig_dataset(df, target_col):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    counts = df[target_col].value_counts().sort_index()
    bars = axes[0].bar(["Stay", "Leave"], counts.values, color=[PALETTE[0], PALETTE[1]])
    for b, c in zip(bars, counts.values):
        axes[0].annotate(f"{c}\n({c / counts.sum() * 100:.1f}%)", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, counts.max() * 1.22); axes[0].set_title("(a) Class balance"); axes[0].set_ylabel("Employees")
    for ax, c, t in zip(axes[1:], ["YearsAtCompany", "MonthlyIncome", "Age"], ["(b) Tenure (years at company)", "(c) Monthly income", "(d) Age"]):
        for v, lab, cc in [(0, "Stay", PALETTE[0]), (1, "Leave", PALETTE[1])]:
            sns.kdeplot(df.loc[df[target_col] == v, c], ax=ax, label=lab, color=cc, fill=True, alpha=0.25, common_norm=False,
                        clip=(df[c].min(), df[c].max()))
        ax.set_title(t); ax.set_xlabel(c); ax.legend(fontsize=8)
    fig.tight_layout(); save_fig(fig, "fig01_dataset_overview")


def fig_pso(hist, trace):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if "val_F1" in trace:
        ax.scatter(trace["iteration"] + np.random.RandomState(0).uniform(-0.12, 0.12, len(trace)), trace["val_F1"],
                   s=18, color="#95a5a6", alpha=0.8, label="individual particle evaluations")
    ax.plot(hist["iteration"], hist["best_f1_so_far"], marker="o", color=PALETTE[4], lw=2.2, label="best so far")
    ax.set_xlabel("PSO iteration"); ax.set_ylabel("Inner-validation F1"); ax.set_title("PSO search progress")
    ax.set_xticks(hist["iteration"]); ax.legend(fontsize=8)
    fig.tight_layout(); save_fig(fig, "fig03_pso_convergence")


def fig_boxplots(results, main_names):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, m in zip(axes, ["F1", "AUC_ROC"]):
        d = pd.DataFrame([{"Model": sh(n), m: v} for n in main_names for v in M(results, n, m)])
        order = [sh(n) for n in main_names]
        pal = {sh(n): (col(n) if n == PROPOSED_NAME else "#C8CDD2") for n in main_names}
        sns.boxplot(data=d, x="Model", y=m, order=order, palette=pal, ax=ax)
        sns.stripplot(data=d, x="Model", y=m, order=order, color="black", alpha=0.35, size=2.5, ax=ax)
        ax.set_title(f"{pm(m)} over {len(results)} test folds"); ax.set_xlabel(""); ax.set_ylabel(pm(m))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); save_fig(fig, "fig04_cv_boxplots")


def _fold_curves(results, name):
    grid = np.linspace(0, 1, 101)
    tprs, precs = [], []
    for r in results:
        y, p = r["y_te"], r["models"][name]["prob"]
        fpr, tpr, _ = roc_curve(y, p); tprs.append(np.interp(grid, fpr, tpr))
        pr, rc, _ = precision_recall_curve(y, p); precs.append(np.interp(grid, rc[::-1], pr[::-1]))
    return grid, np.array(tprs), np.array(precs)


def fig_curves(results, main_names):
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for n in main_names:
        grid, tprs, precs = _fold_curves(results, n)
        lw = LW_PROPOSED if n == PROPOSED_NAME else LW_BASELINE
        auc = M(results, n, "AUC_ROC").mean(); ap = M(results, n, "AUC_PR").mean()
        axes[0].plot(grid, tprs.mean(0), color=col(n), lw=lw, label=f"{sh(n)} (AUC={auc:.3f})")
        axes[1].plot(grid, precs.mean(0), color=col(n), lw=lw, label=f"{sh(n)} (AP={ap:.3f})")
        if n == PROPOSED_NAME:
            axes[0].fill_between(grid, np.clip(tprs.mean(0) - tprs.std(0), 0, 1), np.clip(tprs.mean(0) + tprs.std(0), 0, 1), color=col(n), alpha=0.12)
            axes[1].fill_between(grid, np.clip(precs.mean(0) - precs.std(0), 0, 1), np.clip(precs.mean(0) + precs.std(0), 0, 1), color=col(n), alpha=0.12)
        y_all = np.concatenate([r["y_te"] for r in results]); p_all = np.concatenate([r["models"][n]["prob"] for r in results])
        fo, mp = calibration_curve(y_all, p_all, n_bins=8, strategy="quantile")
        axes[2].plot(mp, fo, marker="o", ms=4, color=col(n), lw=lw, label=sh(n))
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"(a) ROC (mean of {len(results)} folds, band = ±1 SD for CA-FTX+)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall (mean of folds)")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect"); axes[2].set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="(c) Calibration (pooled out-of-fold, quantile bins)")
    for a in axes:
        a.legend(fontsize=7)
    fig.tight_layout(); save_fig(fig, "fig05_roc_pr_calibration")


def fig_confusion(results, main_names, thr_mode="0.5"):
    r = results[-1]; n = len(main_names); ncols = 4; nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows)); axes = np.atleast_2d(axes).reshape(nrows, ncols)
    for i, nm in enumerate(main_names):
        ax = axes[i // ncols, i % ncols]; mm = r["models"][nm]
        thr = 0.5 if thr_mode == "0.5" else mm["thr"]
        cm = confusion_matrix(r["y_te"], (mm["prob"] >= thr).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
        ax.set_title(f"{sh(nm)} (thr={thr:.2f})", fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle(f"Confusion matrices, last test fold, threshold = {'0.5' if thr_mode == '0.5' else 'validation-tuned'}", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig06_confusion_matrices_thr0.5" if thr_mode == "0.5" else "figS_confusion_matrices_tuned_thr")


def fig_forest(boot, main_names):
    metrics = ["F1", "AUC_ROC", "AUC_PR", "Precision", "Recall", "MCC"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
    for ax, m in zip(axes.ravel(), metrics):
        sub = boot[m].loc[main_names].sort_values("mean"); y = np.arange(len(sub))
        ax.errorbar(sub["mean"], y, xerr=np.vstack([sub["mean"] - sub["CI_low"], sub["CI_high"] - sub["mean"]]), fmt="none", ecolor="black", capsize=3)
        for yi, n in enumerate(sub.index):
            ax.scatter(sub.loc[n, "mean"], yi, color=col(n), s=60, zorder=3)
        ax.axvline(boot[m].loc[PROPOSED_NAME, "mean"], color=PALETTE[0], ls="--", alpha=0.6)
        ax.set_yticks(y); ax.set_yticklabels([sh(n) for n in sub.index], fontsize=8); ax.set_title(pm(m))
    fig.suptitle("Mean and 95% bootstrap CI over folds (5,000 resamples)", fontweight="bold")
    fig.tight_layout(); save_fig(fig, "fig08_forest_plot")


def fig_cd(friedman_res, metric):
    avg, st, p, cd, nf, k = friedman_res
    avg = avg.sort_values(); fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot([1, k], [0, 0], color="black", lw=1.2)
    for r_ in range(1, k + 1):
        ax.plot([r_, r_], [-0.03, 0.03], color="black"); ax.text(r_, 0.07, str(r_), ha="center", fontsize=9)
    for i, (n, rk) in enumerate(avg.items()):
        side = 1 if i % 2 == 0 else -1; stem = 0.25 + 0.12 * (i // 2)
        ax.plot([rk, rk], [0, side * stem], color=col(n), lw=1.2)
        ax.text(rk, side * stem + 0.03 * side, f"{sh(n)} ({rk:.2f})", ha="center", va="bottom" if side > 0 else "top", fontsize=8, color=col(n))
    cd_y = -max(0.25 + 0.12 * (k // 2), 0.5) - 0.15
    ax.plot([1, 1 + cd], [cd_y, cd_y], color="black", lw=1.5); ax.text(1 + cd / 2, cd_y - 0.05, f"CD = {cd:.2f}", ha="center", fontsize=9)
    vals, y_c, i = avg.values, max(0.25 + 0.12 * (k // 2), 0.5) + 0.15, 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            ax.plot([vals[i], vals[j]], [y_c, y_c], color="gray", lw=4, alpha=0.6); y_c += 0.12
        i = j + 1
    ax.set_ylim(cd_y - 0.2, y_c + 0.15); ax.axis("off")
    ax.set_title(f"Critical-difference diagram, {pm(metric)} (Friedman chi2={st:.2f}, p={fmt_p(p)}, {nf} folds, {k} models)")
    fig.tight_layout(); save_fig(fig, f"figS_cd_diagram_{metric}")


def fig_ablation(results, abl_df):
    rng = np.random.RandomState(SEED)
    items = [(NOGATE_MIX, PROPOSED_NAME, "System: w/o gate (mixture) vs CA-FTX+ mixture"),
             (FT_MIX, PROPOSED_NAME, "System: plain FT-Transformer mixture vs CA-FTX+ mixture")] + \
            [(v, SINGLE_FULL, f"Single net: {v}") for v in ABLATION_NAMES]
    names = names_in_all(results)
    items = [it for it in items if it[0] in names and it[1] in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(items) + 2.2), sharey=True)
    for ax, m in zip(axes, ["F1", "AUC_ROC"]):
        for yi, (v, ref, lab) in enumerate(items[::-1]):
            d = M(results, v, m) - M(results, ref, m); lo, hi = boot_ci(d, rng)
            grp = "System level (reference = full CA-FTX+ mixture)" if ref == PROPOSED_NAME else "Single network (reference = full CA-FTX+, member 1)"
            try:
                padj = abl_df.loc[(grp, v), f"p_Holm_{pm(m)}"]
            except Exception:
                padj = np.nan
            c = PALETTE[1] if (pd.notna(padj) and padj < 0.05) else "#7f8c8d"
            ax.errorbar(d.mean(), yi, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", color=c, capsize=3, ms=6)
            ax.text(hi, yi + 0.18, "" if pd.isna(padj) else f"p_Holm={fmt_p(padj)}", fontsize=6.5, color=c, ha="left")
        ax.axvline(0, color="black", lw=1); ax.set_xlabel(f"Δ{pm(m)} (variant − reference), mean ± 95% CI")
        ax.set_yticks(range(len(items))); ax.set_yticklabels([it[2] for it in items[::-1]], fontsize=8); ax.set_title(pm(m))
    fig.suptitle("Detailed ablation (red = significant after Holm correction; negative = variant worse than the full model)", fontweight="bold")
    fig.tight_layout(); save_fig(fig, "fig_ablation")


def fig_gate(results, D, init_hist, auc_modes):
    G = [r["gate"] for r in results]; feats = D["feature_names"]
    hist = np.sum([g["hist"] for g in G], 0).astype(float); hist /= hist.sum()
    centers = (np.arange(50) + 0.5) / 50
    fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(centers, hist / (1 / 50), width=1 / 50, color=PALETTE[0], alpha=0.75, label="trained (all test folds)")
    if init_hist is not None:
        h0 = init_hist["hist"].astype(float); h0 /= h0.sum()
        ax.step(np.r_[0, centers + 0.01], np.r_[h0[0], h0] / (1 / 50), where="post", color=PALETTE[1], lw=2, label="untrained network (initialisation)")
    ax.axvline(0.5, color="black", ls="--", lw=1); ax.set_xlim(0, 1)
    ax.set_xlabel("Raw element-wise gate value"); ax.set_ylabel("Density"); ax.set_title("(a) Distribution of all gate values"); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    mf, es, ds = (np.mean([g[k] for g in G], 0) for k in ["mean_feat", "emp_sd", "dim_sd"])
    ax.scatter(mf, es, s=40 + 900 * ds, color=PALETTE[2], alpha=0.7, edgecolor="black")
    for j in np.argsort(-es)[:6]:
        ax.annotate(feats[j], (mf[j], es[j]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel("Mean gate value of the feature (averaged over employees and dimensions)")
    ax.set_ylabel("SD across employees"); ax.set_title("(b) Same mean (~0.5), different per-employee variability\n(marker size ∝ SD across embedding dimensions)")
    ax = fig.add_subplot(gs[1, 0])
    labs = {"mean_element": "mean gate\nper token & dim.", "mean_token": "mean gate\nper token", "const_0.5": "constant\n0.5", "shuffled": "shuffled across\nemployees"}
    data = [(auc_modes[k] - auc_modes["learned"]) for k in labs]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True)
    ax.set_xticklabels(list(labs.values()))
    for b in bp["boxes"]:
        b.set_facecolor("#AED6F1")
    for i, d in enumerate(data):
        ax.scatter(np.random.RandomState(i).normal(i + 1, 0.05, len(d)), d, s=10, color="black", alpha=0.45)
        p = wil(auc_modes[list(labs)[i]], auc_modes["learned"])[1]
        ax.text(i + 1, ax.get_ylim()[1], f"p={fmt_p(p)}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", lw=1); ax.set_ylabel("ΔAUC-ROC vs learned gates"); ax.set_title("(c) Replacing the learned gates at test time", pad=18)
    ax = fig.add_subplot(gs[1, 1])
    nr = np.mean([g["norm_ratio"] for g in G], 0); o = np.argsort(nr)
    ax.bar(range(len(nr)), nr[o], color=PALETTE[3], edgecolor="black")
    ax.set_xticks(range(len(nr))); ax.set_xticklabels([feats[j] for j in o], rotation=90, fontsize=6.5)
    ax.set_ylabel("‖gated token‖ / ‖token‖"); ax.set_title("(d) How much the gate shrinks each feature token")
    fig.suptitle("Gate analysis: why a gate whose values are close to 0.5 still helps", fontweight="bold")
    fig.tight_layout(); save_fig(fig, "fig_gate_analysis")


def _assemble_first_repeat(results, D):
    n, F = len(D["y"]), len(D["feature_names"])
    A = dict(occ=np.zeros((n, F)), ig=np.zeros((n, F)), feat=np.zeros((n, F)), prob=np.zeros(n), thr=np.zeros(n))
    f_base = []
    for r in results[:N_FOLDS]:
        i = r["test_idx"]; e = r["explain"]
        A["occ"][i], A["ig"][i] = e["occ"], e["ig"]
        A["feat"][i] = np.concatenate([e["xn"], e["xc"].astype(np.float32)], 1)
        A["prob"][i], A["thr"][i] = r["models"][PROPOSED_NAME]["prob"], r["models"][PROPOSED_NAME]["thr"]; f_base.append(e["f_base"])
    A["f_base"] = float(np.mean(f_base))
    return A


def fig_importance(imp, top_k=15):
    top = imp.head(top_k).iloc[::-1]; y = np.arange(len(top))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    axes[0].barh(y, top["Perm_AUC_drop"], xerr=top["Perm_SD_across_folds"], color=PALETTE[2], edgecolor="black", capsize=2)
    axes[0].set_xlabel("AUC-ROC drop when permuted (mean ± SD over folds)"); axes[0].set_title("(a) Permutation importance")
    axes[1].barh(y, top["Occlusion_mean_abs"], color=PALETTE[0], edgecolor="black"); axes[1].set_xlabel("Mean |attribution|"); axes[1].set_title("(b) Occlusion ('SHAP-style')")
    axes[2].barh(y, top["IG_mean_abs"], color=PALETTE[4], edgecolor="black"); axes[2].set_xlabel("Mean |attribution|"); axes[2].set_title("(c) Integrated Gradients")
    axes[0].set_yticks(y); axes[0].set_yticklabels(top.index)
    fig.suptitle(f"Top-{top_k} features by mean rank across the three methods (all out-of-fold test folds, 3-member mixture)", fontweight="bold")
    fig.tight_layout(); save_fig(fig, "fig09_importance_three_methods")


def fig_beeswarm(A, D, imp, top_k=12):
    feats = D["feature_names"]; top = list(imp.sort_values("IG_mean_abs", ascending=False).head(top_k).index)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, key, ttl in [(axes[0], "ig", "(a) Integrated Gradients"), (axes[1], "occ", "(b) Occlusion ('SHAP-style')")]:
        for row, f in enumerate(top):
            j = feats.index(f); v = rankdata(A["feat"][:, j]) / len(A["feat"])
            jit = (np.random.RandomState(SEED + row).rand(len(v)) - 0.5) * 0.6
            sc = ax.scatter(A[key][:, j], row + jit, c=v, cmap="coolwarm", s=7, alpha=0.7)
        ax.axvline(0, color="black", lw=1); ax.set_yticks(range(len(top))); ax.set_yticklabels(top); ax.set_xlabel("Attribution (change in predicted probability)"); ax.set_title(ttl)
    fig.colorbar(sc, ax=axes, label="Feature value (low → high)", shrink=0.8)
    fig.suptitle("Beeswarm of out-of-fold attributions (first CV repeat: every employee once)", fontweight="bold")
    save_fig(fig, "fig10_beeswarm")


def fig_agreement(imp):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (a, b, la, lb) in zip(axes, [("Occlusion_mean_abs", "IG_mean_abs", "Occlusion", "IG"),
                                          ("Perm_AUC_drop", "IG_mean_abs", "Permutation (AUC drop)", "IG"),
                                          ("Perm_AUC_drop", "Occlusion_mean_abs", "Permutation (AUC drop)", "Occlusion")]):
        ax.scatter(imp[a], imp[b], color=PALETTE[0], s=30)
        for f in imp.sort_values("Mean_rank").head(6).index:
            ax.annotate(f, (imp.loc[f, a], imp.loc[f, b]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rho, _ = spearmanr(imp[a], imp[b]); ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_title(f"Spearman ρ = {rho:.3f}")
    fig.suptitle("Agreement between explanation methods (one point per feature)", fontweight="bold")
    fig.tight_layout(); save_fig(fig, "fig11_explanation_agreement")


def fig_waterfall(A, D, results, top_k=8):
    y = D["y"]; feats = D["feature_names"]; n_num = D["n_num"]
    pred = (A["prob"] >= A["thr"]).astype(int); ok = pred == y
    cases = []
    if (ok & (y == 1)).any():
        cases.append((np.where(ok & (y == 1))[0][np.argmax(A["prob"][ok & (y == 1)])], "Correctly predicted: Leave"))
    if (ok & (y == 0)).any():
        cases.append((np.where(ok & (y == 0))[0][np.argmin(A["prob"][ok & (y == 0)])], "Correctly predicted: Stay"))
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(8.5 * len(cases), 5.5)); axes = np.atleast_1d(axes)
    for ax, (i, ttl) in zip(axes, cases):
        a = pd.Series(A["ig"][i], index=range(len(feats))); order = a.abs().sort_values(ascending=False).index[:top_k]
        rest = a.drop(order).sum(); labels, vals = [], []
        for j in order[::-1]:
            raw = D["x_num_raw"][i, j] if j < n_num else D["x_cat_raw"][i, j - n_num]
            raw = f"{raw:.4g}" if j < n_num else str(raw)
            labels.append(f"{feats[j]} = {raw}"); vals.append(a[j])
        labels = ["all other features"] + labels; vals = [rest] + vals
        cum = A["f_base"]
        for k, (lb, v) in enumerate(zip(labels, vals)):
            ax.barh(k, v, left=cum if v >= 0 else cum + v, color=PALETTE[1] if v >= 0 else PALETTE[5], edgecolor="black"); cum += v
        ax.axvline(A["f_base"], color="gray", ls="--", label=f"baseline prediction ({A['f_base']:.3f})")
        ax.axvline(A["prob"][i], color="black", label=f"model output ({A['prob'][i]:.3f})")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8); ax.set_title(ttl + f"   (employee #{i})", fontsize=10); ax.legend(fontsize=7)
    fig.suptitle("Local Integrated-Gradients explanations (baseline + sum of attributions = model output)", fontweight="bold")
    fig.tight_layout(); save_fig(fig, "fig12_local_waterfall")


def fig_true_shap(results, imp, D):
    sh_ = results[-1].get("shap")
    if sh_ is None:
        return
    feats = D["feature_names"]
    sa = pd.Series(np.abs(sh_["values"]).mean(0), index=feats).sort_values(ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6)); ax.barh(sa.index, sa.values, color=PALETTE[5], edgecolor="black")
    ax.set_xlabel("Mean |SHAP value| (permutation-sampling Shapley, last test fold)"); ax.set_title("True SHAP values for the CA-FTX+ mixture")
    fig.tight_layout(); save_fig(fig, "figS_true_shap")


def fig_nested(nested_df):
    if nested_df is None:
        return
    df = nested_df[0]; fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, a, b, t in [(axes[0], "F1 (single global PSO)", "F1 (nested PSO)", "F1"), (axes[1], "AUC-ROC (single global PSO)", "AUC-ROC (nested PSO)", "AUC-ROC")]:
        for _, r in df.iterrows():
            ax.plot([0, 1], [r[a], r[b]], color="#95a5a6", marker="o")
        ax.plot([0, 1], [df[a].mean(), df[b].mean()], color=PALETTE[1], lw=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["single global PSO\n(as submitted)", "nested PSO\n(inside each fold)"]); ax.set_title(t)
    fig.suptitle("Sensitivity to where the hyper-parameter search is run", fontweight="bold")
    fig.tight_layout(); save_fig(fig, "figS_nested_pso")


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if pd.isna(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    return o


FAILED = []


def safe(label, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        import traceback
        print(f"[WARNING] '{label}' failed: {repr(e)[:200]}")
        FAILED.append(f"{label}: {repr(e)[:300]}")
        traceback.print_exc(limit=3)
        return None


README = """# output.zip -- what is inside (reviewer comment -> file)

figures/ (PNG 300 dpi + PDF)           tables/ (CSV + XLSX + LaTeX)
--------------------------------------------------------------------------------
R1.1 tree ensembles       tables/T01_main_results, T02_statistical_tests, T03a/b, T13_extra_baselines, T08d
                          figures/fig04_cv_boxplots, fig05_roc_pr_calibration, fig08_forest_plot, figS_cd_diagram_*
R1.2 gate values ~0.5     figures/fig_gate_analysis ; tables/T06_gate_interventions, T07a/b_gate_*
R2.1 novelty (evidence)   tables/T05_component_ladder, T04_detailed_ablation ; figures/fig_ablation
R2.2 detailed ablation    tables/T04_detailed_ablation, T05_component_ladder ; figures/fig_ablation
R2.3 hyper-parameters     tables/T08_hyperparameters_full, T08b_pso_search_space, T08c_pso_all_evaluations,
                          T08d_gbdt_random_search_trace, hyperparameters.json, T19_dataset_features
R2.4 concise results      tables/T01_main_results (replaces old Table 2), T02_statistical_tests (replaces old Tables 3+5),
                          T03b (old Table 4), T09_efficiency (old Table 6); figures: fig04, fig05 (replaces old Figs 5+7), fig06, fig08
Explainability            figures/fig09_importance_three_methods, fig10_beeswarm, fig11_explanation_agreement,
                          fig12_local_waterfall, figS_true_shap ; tables/T10_feature_importance, T11_explanation_agreement
Other figures             fig01_dataset_overview (old Fig.1), fig03_pso_convergence (old Fig.3)
Checks I added            tables/T14a/b_nested_pso_*, figures/figS_nested_pso ; T18_reproduction_check_vs_submitted_Table2 ;
                          T15_metrics_at_fixed_threshold_0.5 ; T12_confusion_counts ; T16_bootstrap_ci ; T17_per_fold_metrics
results_summary.json                   every headline number in one machine-readable file
logs/run_log.txt, logs/warnings.txt    full console log and any step that failed
"""


def main():
    orig_stdout = sys.stdout
    log_f = open(os.path.join(LOG_DIR, "run_log.txt"), "a", encoding="utf-8")
    if WORKER is None:
        sys.stdout = _Tee(orig_stdout, log_f)
    t_start = time.time()
    try:
        seed_everything(SEED)
        print(f"Using device: {DEVICE} | QUICK_TEST={QUICK_TEST}")
        df, target_col = load_ibm_hr_dataset()
        D = prepare_data(df, target_col)
        print(f"Inputs: {D['n_num']} numeric + {len(D['cat_cols'])} categorical | cardinalities={D['cat_card']}")

        sig = json.dumps(dict(v=4, SEED=SEED, F=N_FOLDS, R=N_REPEATS, E=EPOCHS, P=PATIENCE, VF=VAL_FRACTION, NE=N_ENSEMBLE,
                              AB=RUN_ABLATIONS, GB=RUN_GBDT, GBB=GBDT_SEARCH_BUDGET, NP=RUN_NESTED_PSO, NR=NESTED_PSO_REPEATS,
                              SH=RUN_TRUE_SHAP, PR=N_PERM_REPEATS, IG=IG_STEPS, Q=QUICK_TEST, RP=RERUN_PSO,
                              PUB=PUBLISHED_CFG, PSO=(PSO_PARTICLES, PSO_ITERS, PSO_EPOCHS)), sort_keys=True, default=str)
        setup_checkpoint_dir(hashlib.md5(sig.encode()).hexdigest())

        best_cfg, pso_hist, pso_trace, pso_source = get_pso_config(D)
        print("CA-FTX+ configuration used:", json.dumps(best_cfg), "| source:", pso_source)
        gbdt_cfg = get_gbdt_config(D)
        if WORKER is not None:
            run_all_folds(D, best_cfg, gbdt_cfg, shard=WORKER)
            return

        results = run_all_folds(D, best_cfg, gbdt_cfg)
        names = names_in_all(results)
        main_names = [n for n in MAIN_MODEL_NAMES if n in names]
        print("\nModels evaluated in every fold:", sorted(names))

        mean_df, std_df = safe("T01", table_main, results, main_names)
        print("\nMAIN RESULTS")
        print(mean_df.round(4).to_string())
        stats_df = safe("T02", table_stats, results, main_names)
        fried = safe("T03", table_friedman, results, main_names)
        boot = safe("T16", table_bootstrap, results, main_names)
        abl_df = safe("T04", table_ablation, results, names) if RUN_ABLATIONS else None
        safe("T05", table_ladder, results, names) if RUN_ABLATIONS else None
        gate_tab = safe("T06", table_gate, results)
        safe("T07", table_gate_stats, results, D)
        comp_df = safe("T09", complexity_table, D, best_cfg, results, main_names)
        safe("T08", table_hyperparameters, D, best_cfg, gbdt_cfg, comp_df if comp_df is not None else pd.DataFrame(), pso_source, pso_trace)
        exp = safe("T10/T11", explain_tables, results, D)
        safe("T12", table_confusion, results, main_names)
        safe("T13", table_extra, results, names)
        safe("T15", table_threshold05, results, main_names)
        nested = safe("T14", table_nested, results) if RUN_NESTED_PSO else None
        repro = safe("T18", table_reproduction, results)
        safe("T19", table_dataset, D, df)
        safe("T17", table_per_fold, results, [n for n in main_names + ([SINGLE_FULL, NOGATE_MIX, FT_MIX] + ABLATION_NAMES if RUN_ABLATIONS else []) if n in names])

        safe("fig01", fig_dataset, df, target_col)
        safe("fig03", fig_pso, pso_hist, pso_trace)
        safe("fig04", fig_boxplots, results, main_names)
        safe("fig05", fig_curves, results, main_names)
        safe("fig06", fig_confusion, results, main_names, "0.5")
        safe("figS_conf", fig_confusion, results, main_names, "tuned")
        if boot is not None:
            safe("fig08", fig_forest, boot, main_names)
        if fried:
            for m in ["F1", "AUC_ROC"]:
                safe(f"cd_{m}", fig_cd, fried[m], m)
        if RUN_ABLATIONS and abl_df is not None:
            safe("fig_ablation", fig_ablation, results, abl_df)
        if gate_tab is not None:
            init_hist = safe("init_gate", init_gate_histogram, D, best_cfg)
            safe("fig_gate", fig_gate, results, D, init_hist, gate_tab[1])
        if exp is not None:
            imp = exp[0]
            A = safe("assemble", _assemble_first_repeat, results, D)
            safe("fig09", fig_importance, imp)
            safe("fig11", fig_agreement, imp)
            if A is not None:
                safe("fig10", fig_beeswarm, A, D, imp)
                safe("fig12", fig_waterfall, A, D, results)
            safe("figS_shap", fig_true_shap, results, imp, D)
        if nested:
            safe("figS_nested", fig_nested, nested)

        summ = dict(config=best_cfg, pso_source=pso_source, n_folds=len(results), device=str(DEVICE),
                    mean_metrics={n: {m: float(M(results, n, m).mean()) for m in METRICS} for n in sorted(names)},
                    std_metrics={n: {m: float(M(results, n, m).std(ddof=1)) for m in METRICS} for n in sorted(names)},
                    best_model_by_mean_F1=max(main_names, key=lambda n: M(results, n, "F1").mean()),
                    best_model_by_mean_AUC=max(main_names, key=lambda n: M(results, n, "AUC_ROC").mean()),
                    explain_summary=(exp[2] if exp else None), failed_steps=FAILED,
                    runtime_minutes=(time.time() - t_start) / 60)
        if gate_tab is not None:
            summ["gate_interventions_auc"] = {k: float(np.mean(v)) for k, v in gate_tab[1].items()}
        if nested:
            summ["nested_pso"] = nested[1]["value"].to_dict()
        json.dump(to_py(summ), open(os.path.join(TAB_DIR, "..", "results_summary.json"), "w"), indent=2)
        open(os.path.join(OUT_DIR, "README_outputs.md"), "w", encoding="utf-8").write(README)
        with open(os.path.join(LOG_DIR, "warnings.txt"), "w") as f:
            f.write("\n".join(FAILED) if FAILED else "no failed steps")

        proposed_vs_best_warning(results, main_names)
        sys.stdout.flush()
        zip_path = "output.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files_ in os.walk(OUT_DIR):
                for fn in files_:
                    full = os.path.join(root, fn); zf.write(full, os.path.relpath(full, "."))
        print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Wrote {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)."
              + (f"  {len(FAILED)} step(s) failed -- see logs/warnings.txt" if FAILED else ""))
    finally:
        sys.stdout = orig_stdout
        log_f.close()
    try:
        from google.colab import files
        files.download("output.zip")
    except Exception:
        pass


def proposed_vs_best_warning(results, main_names):
    f1 = {n: M(results, n, "F1").mean() for n in main_names}
    best = max(f1, key=f1.get)
    if best != PROPOSED_NAME:
        print(f"\n*** NOTE: on mean F1, '{best}' ({f1[best]:.4f}) is ahead of {PROPOSED_NAME} ({f1[PROPOSED_NAME]:.4f}). "
              "The manuscript text must describe this honestly (see T02 for significance).")
    return False


if __name__ == "__main__":
    main()
