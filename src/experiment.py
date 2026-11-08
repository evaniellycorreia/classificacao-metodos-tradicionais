"""Mini-projeto: classificação com métodos tradicionais I (SVM, regressão logística, k-NN)."""
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.io import arff
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from torch_models import TorchLinearClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
OUT_DIR = ROOT / "results"
SEED = 42
SVM_N = 10_000  # libsvm é O(n^2~n^3): SVC usa subamostra estratificada no CDC
KNN_TUNE_N = 30_000
EVAL_TRAIN_N = 30_000  # teto de pontos para métricas no treino
BOOT = 1000
BOUNDARY_FIT_N = 5_000

MODEL_ORDER = ["SVM linear", "SVM RBF", "Reg. logística", "k-NN"]
TORCH_ORDER = ["SVM linear (torch)", "Reg. logística (torch)"]


def load_cdc():
    data, _ = arff.loadarff(DATA_DIR / "cdc_diabetes.arff")
    df = pd.DataFrame(data)
    y = df.pop("Diabetes_binary").str.decode("utf-8").astype(int).to_numpy()
    prep = lambda: StandardScaler()
    return df, y, prep, ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Income"]


def load_telco():
    df = pd.read_csv(DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)  # 11 linhas com tenure=0
    y = (df.pop("Churn") == "Yes").astype(int).to_numpy()
    df = df.drop(columns="customerID")
    cat = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]
    num = [c for c in df.columns if c not in cat]

    def prep():
        return ColumnTransformer(
            [
                ("num", StandardScaler(), num),
                ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat),
            ]
        )

    return df, y, prep, ["tenure", "MonthlyCharges", "TotalCharges"]


def model_specs():
    return {
        "SVM linear": dict(
            est=lambda: SVC(kernel="linear", class_weight="balanced", cache_size=1000),
            grid={"C": [0.001, 0.01, 0.1, 1]},
            tune_n=SVM_N,
            fit_n=SVM_N,
        ),
        "SVM RBF": dict(
            est=lambda: SVC(kernel="rbf", class_weight="balanced", cache_size=1000),
            grid={"C": [0.1, 1, 10], "gamma": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]},
            tune_n=SVM_N,
            fit_n=SVM_N,
        ),
        "Reg. logística": dict(
            est=lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
            grid={"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
            tune_n=None,
            fit_n=None,
        ),
        "k-NN": dict(
            est=lambda: KNeighborsClassifier(),
            grid={"n_neighbors": [1, 3, 5, 11, 21, 31, 51, 101]},
            tune_n=KNN_TUNE_N,
            fit_n=None,
        ),
    }


def subset(X, y, n, seed=SEED):
    if n is None or len(y) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


def metrics(y, pred):
    return dict(
        acc=accuracy_score(y, pred),
        bacc=balanced_accuracy_score(y, pred),
        prec=precision_score(y, pred, zero_division=0),
        rec=recall_score(y, pred),
        f1=f1_score(y, pred),
        cm=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    )


def bootstrap_f1(y, pred, rng):
    y = y.astype(bool)
    pred = pred.astype(bool)
    scores = np.empty(BOOT)
    for i in range(BOOT):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], pred[idx]
        tp = np.sum(yb & pb)
        scores[i] = 2 * tp / (2 * tp + np.sum(~yb & pb) + np.sum(yb & ~pb))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def mcnemar_p(y, pred_a, pred_b):
    ok_a, ok_b = pred_a == y, pred_b == y
    b, c = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


def describe(df, cols):
    stats = df[cols].agg(["mean", "var", "min", "max"]).T
    return {c: {k: float(v) for k, v in row.items()} for c, row in stats.iterrows()}


def evaluate(pipe, Xfit, yfit, Xte, yte, rng):
    Xe, ye = subset(Xfit, yfit, EVAL_TRAIN_N, seed=1)
    pred_te = pipe.predict(Xte)
    return dict(
        test=metrics(yte, pred_te),
        train=metrics(ye, pipe.predict(Xe)),
        f1_ci=bootstrap_f1(yte, pred_te, rng),
    ), pred_te


def boundary_data(Xtr, ytr, Xall, yall, prep_factory, best_params, specs, rng):
    prep = prep_factory().fit(Xtr)
    pca = PCA(n_components=2, random_state=SEED).fit(prep.transform(Xtr))
    Ztr, Zall = pca.transform(prep.transform(Xtr)), pca.transform(prep.transform(Xall))
    Zfit, yfit = subset(Ztr, ytr, BOUNDARY_FIT_N)
    lo, hi = np.percentile(Zall, 0.5, axis=0), np.percentile(Zall, 99.5, axis=0)
    pad = 0.1 * (hi - lo)
    gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 250), np.linspace(lo[1] - pad[1], hi[1] + pad[1], 250))
    grid = np.c_[gx.ravel(), gy.ravel()]
    regions = {}
    for name in MODEL_ORDER:
        model = specs[name]["est"]().set_params(**best_params[name])
        regions[name] = model.fit(Zfit, yfit).predict(grid).reshape(gx.shape)
    pts = np.concatenate([rng.choice(np.flatnonzero(yall == k), 500, replace=False) for k in (0, 1)])
    return dict(gx=gx, gy=gy, regions=regions, pts=Zall[pts], labels=yall[pts], var=pca.explained_variance_ratio_.sum())


def run_dataset(name, X, y, prep_factory, desc_cols):
    print(f"\n=== {name}: {X.shape} ===", flush=True)
    rng = np.random.default_rng(SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv_k = 5 if len(ytr) < 20_000 else 3
    cv = StratifiedKFold(cv_k, shuffle=True, random_state=SEED)
    specs = model_specs()
    out = dict(
        n=int(len(y)),
        n_features=int(X.shape[1]),
        n_train=int(len(ytr)),
        n_test=int(len(yte)),
        pos_rate=float(y.mean()),
        majority_acc=float(max(y.mean(), 1 - y.mean())),
        cv_folds=cv_k,
        describe=describe(X, desc_cols),
        models={},
    )
    preds, best_params = {}, {}

    for mname in MODEL_ORDER:
        spec = specs[mname]
        t0 = time.time()
        Xt, yt = subset(Xtr, ytr, spec["tune_n"])
        pipe = Pipeline([("prep", prep_factory()), ("clf", spec["est"]())])
        gs = GridSearchCV(pipe, {f"clf__{k}": v for k, v in spec["grid"].items()}, scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(Xt, yt)
        best = {k.split("__")[1]: v for k, v in gs.best_params_.items()}
        best_params[mname] = best
        Xf, yf = subset(Xtr, ytr, spec["fit_n"])
        final = Pipeline([("prep", prep_factory()), ("clf", spec["est"]().set_params(**best))]).fit(Xf, yf)
        res, preds[mname] = evaluate(final, Xf, yf, Xte, yte, rng)
        res.update(best_params=best, cv_f1=float(gs.best_score_), n_fit=int(len(yf)), seconds=time.time() - t0)
        out["models"][mname] = res
        print(f"{mname}: {best} cvF1={gs.best_score_:.3f} testF1={res['test']['f1']:.3f} ({res['seconds']:.0f}s)", flush=True)

    # Versões em torch usam o C escolhido pelo GridSearchCV de sklearn e o treino completo.
    for tname, loss, ref in [("SVM linear (torch)", "hinge", "SVM linear"), ("Reg. logística (torch)", "logistic", "Reg. logística")]:
        t0 = time.time()
        pipe = Pipeline([("prep", prep_factory()), ("clf", TorchLinearClassifier(loss=loss, C=best_params[ref]["C"]))]).fit(Xtr, ytr)
        res, preds[tname] = evaluate(pipe, Xtr, ytr, Xte, yte, rng)
        res.update(best_params=dict(C=best_params[ref]["C"]), n_fit=int(len(ytr)), seconds=time.time() - t0,
                   agreement_with_sklearn=float(np.mean(preds[tname] == preds[ref])))
        out["models"][tname] = res
        print(f"{tname}: testF1={res['test']['f1']:.3f} agree={res['agreement_with_sklearn']:.3f}", flush=True)

    best_name = max(MODEL_ORDER, key=lambda m: out["models"][m]["test"]["f1"])
    out["best_model"] = best_name
    for m in MODEL_ORDER + TORCH_ORDER:
        out["models"][m]["mcnemar_p_vs_best"] = None if m == best_name else mcnemar_p(yte, preds[best_name], preds[m])
    out["mcnemar_pairs"] = {
        f"{a}|{b}": mcnemar_p(yte, preds[a], preds[b]) for i, a in enumerate(MODEL_ORDER) for b in MODEL_ORDER[i + 1 :]
    }

    boundary = boundary_data(Xtr, ytr, X, y, prep_factory, best_params, specs, rng)
    out["pca_var_2d"] = float(boundary["var"])
    return out, boundary


def plot_boundaries(boundaries, path):
    fig, axes = plt.subplots(len(boundaries), 4, figsize=(15, 3.6 * len(boundaries)), squeeze=False)
    cmap = ListedColormap(["#9ecae1", "#fcae91"])
    for row, (dname, b) in zip(axes, boundaries.items()):
        for ax, mname in zip(row, MODEL_ORDER):
            ax.contourf(b["gx"], b["gy"], b["regions"][mname], levels=[-0.5, 0.5, 1.5], cmap=cmap)
            for k, color in ((0, "#08519c"), (1, "#cb181d")):
                sel = b["labels"] == k
                ax.scatter(b["pts"][sel, 0], b["pts"][sel, 1], s=5, c=color, alpha=0.6, label=f"classe {k}")
            ax.set_xlim(b["gx"].min(), b["gx"].max())
            ax.set_ylim(b["gy"].min(), b["gy"].max())
            ax.set_title(f"{dname}: {mname}", fontsize=10)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
    axes[0][0].legend(loc="upper right", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_panel(results, path):
    names = MODEL_ORDER + TORCH_ORDER
    metric_keys = [("acc", "Acurácia"), ("prec", "Precisão"), ("rec", "Revocação"), ("f1", "F1")]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.3, 1, 1])
    for r, (dname, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, 3 * r : 3 * r + 3])
        width = 0.8 / len(metric_keys)
        for j, (mk, label) in enumerate(metric_keys):
            ax.bar(np.arange(len(names)) + j * width, [res["models"][m]["test"][mk] for m in names], width, label=label)
        ax.set_xticks(np.arange(len(names)) + 0.3)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{dname}: métricas no teste")
        if r == 0:
            ax.legend(fontsize=8, ncol=4, loc="upper center")
        for c, m in enumerate(names):
            cm = np.array(res["models"][m]["test"]["cm"])
            cax = fig.add_subplot(gs[1 + r, c])
            cax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(cm):
                cax.text(j, i, f"{v}", ha="center", va="center", fontsize=8, color="black")
            cax.set_xticks([0, 1])
            cax.set_yticks([0, 1])
            cax.set_xticklabels(["0", "1"], fontsize=7)
            cax.set_yticklabels(["0", "1"], fontsize=7)
            cax.set_title(f"{dname}\n{m}", fontsize=7)
            cax.set_xlabel("previsto", fontsize=7)
            cax.set_ylabel("real", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def fmt_stat(x):
    return fmt(x, 2) if abs(x) < 1000 else f"{x:.0f}"


def fmt_param(k, v):
    sym = {"C": "C", "gamma": r"\gamma", "n_neighbors": "k"}[k]
    return f"${sym}={v:g}$".replace(".", "{,}")


def write_tables(results):
    names = MODEL_ORDER + TORCH_ORDER
    cdc, telco = results["CDC Diabetes"], results["Telco Churn"]

    rows = []
    for k, (dname, res) in enumerate(results.items()):
        if k:
            rows.append(r"\midrule")
        for i, (col, s) in enumerate(res["describe"].items()):
            label = dname if i == 0 else ""
            rows.append(f"{label} & {col.replace('_', r'\_')} & {fmt_stat(s['mean'])} & {fmt_stat(s['var'])} & {fmt_stat(s['min'])} & {fmt_stat(s['max'])} \\\\")
    (OUT_DIR / "tab_desc.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for m in names:
        cells = [m]
        for res in (cdc, telco):
            t = res["models"][m]["test"]
            cells += [fmt(t[k]) for k in ("acc", "prec", "rec", "f1")] + [fmt(res["models"][m]["train"]["f1"])]
        rows.append(" & ".join(cells) + r" \\")
    (OUT_DIR / "tab_metrics.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for m in MODEL_ORDER:
        cells = [m]
        for res in (cdc, telco):
            r = res["models"][m]
            cells += [", ".join(fmt_param(k, v) for k, v in r["best_params"].items()), fmt(r["cv_f1"])]
        rows.append(" & ".join(cells) + r" \\")
    (OUT_DIR / "tab_params.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def p_text(p):
        return "--" if p is None else ("$<10^{-4}$" if p < 1e-4 else fmt(p, 4))

    rows = []
    for m in names:
        cells = [m]
        for res in (cdc, telco):
            lo, hi = res["models"][m]["f1_ci"]
            cells += [f"[{fmt(lo)}; {fmt(hi)}]", p_text(res["models"][m]["mcnemar_p_vs_best"])]
        rows.append(" & ".join(cells) + r" \\")
    (OUT_DIR / "tab_signif.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results, boundaries = {}, {}
    for name, loader in (("CDC Diabetes", load_cdc), ("Telco Churn", load_telco)):
        X, y, prep_factory, cols = loader()
        results[name], boundaries[name] = run_dataset(name, X, y, prep_factory, cols)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_boundaries(boundaries, OUT_DIR / "boundaries.png")
    plot_panel(results, OUT_DIR / "panel.png")
    write_tables(results)
    print("\nOK: resultados em", OUT_DIR)


if __name__ == "__main__":
    main()
