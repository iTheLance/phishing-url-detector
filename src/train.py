"""Train and evaluate the phishing URL models.

Runs three experiments:
  A. Naive: full URL text, random split. Shows the dataset shortcut.
  B. Fair: hostname only, duplicates removed, split by registered domain.
  C. External: the model from B scored on URLs it has never seen from other sources
     (today's OpenPhish feed and a sample of the Tranco top-sites list).

Usage:  python src/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from urlfeatures import FEATURE_NAMES, features, registered_domain, to_host  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
SEED = 42
FIG.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)


def scores(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "false_positive_rate": float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())),
    }
    if len(set(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, prob)
    return {k: round(float(v), 4) for k, v in out.items()}


def char_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0),
    )


def feature_matrix(hosts):
    return pd.DataFrame([features(h) for h in hosts], columns=FEATURE_NAMES)


print("Loading PhiUSIIL ...")
raw = pd.read_csv(DATA / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"])
raw["is_phishing"] = 1 - raw["label"]  # in PhiUSIIL, label 1 = legitimate
results = {"dataset_rows": len(raw), "phishing_rows": int(raw.is_phishing.sum())}

# ---------------- A. Naive ----------------
print("A. naive full-URL model ...")
rule_pred = (~raw.URL.str.startswith("https://www.")).astype(int)
results["A_one_line_rule"] = scores(raw.is_phishing.values, rule_pred.values.astype(float))
results["A_share_legit_starting_https_www"] = round(float(raw[raw.is_phishing == 0].URL.str.startswith("https://www.").mean()), 4)
a_tr, a_te = train_test_split(raw, test_size=0.2, random_state=SEED, stratify=raw.is_phishing)
naive = char_model().fit(a_tr.URL, a_tr.is_phishing)
results["A_naive_char_model"] = scores(a_te.is_phishing.values, naive.predict_proba(a_te.URL)[:, 1])

# ---------------- B. Fair ----------------
print("B. fair hostname model ...")
df = raw.assign(host=raw.URL.map(to_host))
df = df[df.host != ""]
conflict = df.groupby("host").is_phishing.nunique()
df = df[~df.host.isin(conflict[conflict > 1].index)].drop_duplicates("host")
df["reg"] = df.host.map(registered_domain)
results["B_unique_hosts"] = len(df)
results["B_unique_phishing_hosts"] = int(df.is_phishing.sum())
tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=df.reg))
tr, te = df.iloc[tr_idx], df.iloc[te_idx]
assert not set(tr.reg) & set(te.reg), "registered domain leaked across split"

X_tr, X_te = feature_matrix(tr.host), feature_matrix(te.host)
hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, random_state=SEED).fit(X_tr, tr.is_phishing)
chars = char_model().fit(tr.host, tr.is_phishing)
p_hgb = hgb.predict_proba(X_te)[:, 1]
p_chr = chars.predict_proba(te.host)[:, 1]
p_ens = (p_hgb + p_chr) / 2
results["B_handcrafted_gbm"] = scores(te.is_phishing.values, p_hgb)
results["B_char_ngram_logreg"] = scores(te.is_phishing.values, p_chr)
results["B_ensemble"] = scores(te.is_phishing.values, p_ens)

# ---------------- C. External ----------------
print("C. external test ...")
feed = sorted((DATA / "external").glob("openphish_feed_*.txt"))[-1]  # newest snapshot
results["C_openphish_snapshot"] = feed.name
phish = pd.Series(feed.read_text(encoding="utf-8").split())
phish_hosts = phish.map(to_host).drop_duplicates()
tranco = pd.read_csv(DATA / "external" / "top-1m.csv", header=None, names=["rank", "domain"])
pool = tranco[(tranco["rank"] > 10_000) & (tranco["rank"] <= 500_000)].domain.map(to_host)
seen = set(df.host)
pool = pool[~pool.isin(seen) & ~pool.isin(set(phish_hosts))]
legit_hosts = pool.sample(3000, random_state=SEED)
phish_hosts = phish_hosts[~phish_hosts.isin(seen)]
ext = pd.DataFrame({"host": pd.concat([phish_hosts, legit_hosts]),
                    "is_phishing": [1] * len(phish_hosts) + [0] * len(legit_hosts)})
e_hgb = hgb.predict_proba(feature_matrix(ext.host))[:, 1]
e_chr = chars.predict_proba(ext.host)[:, 1]
results["C_external_counts"] = {"phishing": len(phish_hosts), "legitimate": len(legit_hosts)}
results["C_handcrafted_gbm"] = scores(ext.is_phishing.values, e_hgb)
results["C_char_ngram_logreg"] = scores(ext.is_phishing.values, e_chr)
results["C_ensemble"] = scores(ext.is_phishing.values, (e_hgb + e_chr) / 2)

# ---------------- D. Improvement ----------------
# The external false-positive rate is high because PhiUSIIL's legitimate sites are a narrow sample.
# Add more legitimate domains from a Tranco range that does not overlap the external test (ranks 500k-1M),
# then pick the decision threshold on the B test set (target: at most 2% false positives) and freeze it.
print("D. retrain with broader legitimate data ...")
extra = tranco[tranco["rank"] > 500_000].domain.map(to_host)
extra = extra[~extra.isin(seen) & ~extra.isin(set(ext.host))].drop_duplicates().sample(40_000, random_state=SEED)
extra = extra[~extra.map(registered_domain).isin(set(te.reg))]
tr_d = pd.concat([tr[["host", "is_phishing"]], pd.DataFrame({"host": extra, "is_phishing": 0})])
hgb_d = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, random_state=SEED).fit(
    feature_matrix(tr_d.host), tr_d.is_phishing)
chars_d = char_model().fit(tr_d.host, tr_d.is_phishing)


def ens(hosts, h=hgb_d, c=chars_d):
    return (h.predict_proba(feature_matrix(hosts))[:, 1] + c.predict_proba(hosts)[:, 1]) / 2


p_d_te = ens(te.host)
legit_scores = np.sort(p_d_te[te.is_phishing.values == 0])
threshold = float(legit_scores[int(np.ceil(0.98 * len(legit_scores))) - 1])
p_d_ext = ens(ext.host)
results["D_extra_legit_training_domains"] = len(extra)
results["D_threshold_for_2pct_fpr_on_B"] = round(threshold, 4)
results["D_B_heldout_at_0.5"] = scores(te.is_phishing.values, p_d_te)
results["D_B_heldout_at_threshold"] = scores(te.is_phishing.values, p_d_te, threshold)
results["D_C_external_at_0.5"] = scores(ext.is_phishing.values, p_d_ext)
results["D_C_external_at_threshold"] = scores(ext.is_phishing.values, p_d_ext, threshold)
ext.assign(p_phishing_B=((e_hgb + e_chr) / 2).round(4), p_phishing_D=p_d_ext.round(4)).to_csv(
    ROOT / "reports" / "external_predictions.csv", index=False)

# ---------------- Figures ----------------
print("Figures ...")
labels = ["One-line rule\n(full URL)", "Naive model\n(full URL)", "Fair model\n(held-out domains)",
          "External test\n(first model)", "External test\n(improved)"]
vals = [results["A_one_line_rule"]["f1"], results["A_naive_char_model"]["f1"], results["B_ensemble"]["f1"],
        results["C_ensemble"]["f1"], results["D_C_external_at_0.5"]["f1"]]
fig, ax = plt.subplots(figsize=(9, 4.4))
bars = ax.bar(labels, vals, color=["#b0b0b0", "#b0b0b0", "#2a6f97", "#c8553d", "#3a7d44"])
ax.bar_label(bars, fmt="%.3f")
ax.set_ylim(0, 1.08)
ax.set_ylabel("F1 score (phishing class)")
ax.set_title("High scores on this dataset come from a shortcut, not from learning phishing")
fig.tight_layout()
fig.savefig(FIG / "01_shortcut_vs_fair.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
ConfusionMatrixDisplay.from_predictions(te.is_phishing, (p_ens >= 0.5).astype(int), display_labels=["legit", "phish"],
                                        ax=axes[0], colorbar=False)
axes[0].set_title("B. Held-out domains")
ConfusionMatrixDisplay.from_predictions(ext.is_phishing, (p_d_ext >= 0.5).astype(int),
                                        display_labels=["legit", "phish"], ax=axes[1], colorbar=False)
axes[1].set_title("D. External, improved model")
fig.tight_layout()
fig.savefig(FIG / "02_confusion_matrices.png", dpi=150)

rows = np.random.default_rng(SEED).choice(len(X_te), size=min(5000, len(X_te)), replace=False)
imp = permutation_importance(hgb, X_te.iloc[rows], te.is_phishing.values[rows],
                             n_repeats=5, random_state=SEED, scoring="f1")
order = np.argsort(imp.importances_mean)
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(np.array(FEATURE_NAMES)[order], imp.importances_mean[order], xerr=imp.importances_std[order], color="#2a6f97")
ax.set_xlabel("Drop in F1 when the feature is shuffled")
ax.set_title("What the hand-crafted model relies on")
fig.tight_layout()
fig.savefig(FIG / "03_feature_importance.png", dpi=150)
results["B_feature_importance"] = {FEATURE_NAMES[i]: round(float(imp.importances_mean[i]), 4) for i in order[::-1]}

joblib.dump({"hgb": hgb_d, "chars": chars_d, "feature_names": FEATURE_NAMES, "threshold": 0.5},
            MODELS / "phishing_url_model.joblib", compress=3)
(ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
