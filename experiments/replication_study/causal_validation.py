"""Causal validation: graduated test-time ablation + attribution analysis.

Measures how the detector's mean injection probability changes when we zero
out the top-K features by various rankings (intent-discriminators,
coefficient magnitude, random). If intent features are causally important,
ablating them should reduce B/E mean probabilities faster than ablating
random features.

Also computes per-feature attribution contributions to the logit difference
between injection and benign prompts.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

feats_A = np.load('experiments/replication_study/activations/A_benign_identity.npy')
feats_B = np.load('experiments/replication_study/activations/B_injection_identity.npy')
feats_D = np.load('experiments/replication_study/activations/D_benign_command.npy')
feats_E = np.load('experiments/replication_study/activations/E_injection_command.npy')

with open('data/processed/iris_dataset_balanced.json') as f:
    ds = json.load(f)
labels_orig = np.array([d['label'] for d in ds])
feats_orig = np.load('checkpoints/feature_matrix.npy')

with open('experiments/replication_study/results/replication_results.json') as f:
    r_id = json.load(f)
with open('experiments/replication_study/results/command_category_results.json') as f:
    r_cmd = json.load(f)

X_train, X_test, y_train, y_test = train_test_split(
    feats_orig, labels_orig, test_size=0.2, random_state=42, stratify=labels_orig)

np.random.seed(42)
A_perm = np.random.permutation(len(feats_A))
B_perm = np.random.permutation(len(feats_B))
D_perm = np.random.permutation(len(feats_D))
E_perm = np.random.permutation(len(feats_E))

X_uni = np.concatenate([X_train,
                        feats_A[A_perm[:20]], feats_D[D_perm[:20]],
                        feats_B[B_perm[:20]], feats_E[E_perm[:20]]])
y_uni = np.concatenate([y_train, np.zeros(40, dtype=int), np.ones(40, dtype=int)])

clf_uni = LogisticRegression(max_iter=2000, C=1.0)
clf_uni.fit(X_uni, y_uni)


def probs(clf, feats):
    return clf.predict_proba(feats)[:, 1]


def eval_ablation(feats_to_zero):
    def mask(f):
        f2 = f.copy()
        f2[:, feats_to_zero] = 0
        return f2
    return {
        "B_mean": float(probs(clf_uni, mask(feats_B)).mean()),
        "E_mean": float(probs(clf_uni, mask(feats_E)).mean()),
        "A_mean": float(probs(clf_uni, mask(feats_A)).mean()),
        "D_mean": float(probs(clf_uni, mask(feats_D)).mean()),
    }


intent_id = r_id['top_intent_discriminators_strict']
intent_cmd = r_cmd['top_intent_discriminators']
intent_all = list(dict.fromkeys(intent_id + intent_cmd))

top_by_coef = np.argsort(np.abs(clf_uni.coef_[0]))[::-1].tolist()
rng = np.random.default_rng(0)

print("Graduated test-time ablation — mean probabilities after zeroing top-K features")
print("=" * 100)
print(f"{'K':>5} | {'intent_ranked':<30} | {'by_coef_magnitude':<30} | {'random (avg 3)':<20}")
print(f"{'':>5} | {'B_mean  A_mean  E_mean  D_mean':<30} | {'B_mean  A_mean  E_mean  D_mean':<30} | {'B_mean  A_mean'}")
print("-" * 100)

rows = []
for K in [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]:
    if K == 0:
        r0 = eval_ablation([])
        print(f"{K:>5} | B={r0['B_mean']:.3f} A={r0['A_mean']:.3f} E={r0['E_mean']:.3f} D={r0['D_mean']:.3f}   "
              f"| (same)                         | (same)")
        rows.append({"K": 0, "intent": r0, "by_coef": r0,
                     "random_B": r0['B_mean'], "random_A": r0['A_mean']})
        continue
    # Intent ranked: use intent_all first, then pad with top_by_coef for larger K
    if K <= len(intent_all):
        feats_to_zero = intent_all[:K]
    else:
        extra = [f for f in top_by_coef if f not in intent_all][:K-len(intent_all)]
        feats_to_zero = intent_all + extra
    ab_intent = eval_ablation(feats_to_zero)

    ab_coef = eval_ablation(top_by_coef[:K])

    rand_Bs, rand_As, rand_Es, rand_Ds = [], [], [], []
    for _ in range(3):
        ridx = rng.choice(10240, size=K, replace=False).tolist()
        r = eval_ablation(ridx)
        rand_Bs.append(r['B_mean'])
        rand_As.append(r['A_mean'])
        rand_Es.append(r['E_mean'])
        rand_Ds.append(r['D_mean'])
    rB, rA, rE, rD = np.mean(rand_Bs), np.mean(rand_As), np.mean(rand_Es), np.mean(rand_Ds)

    print(f"{K:>5} | B={ab_intent['B_mean']:.3f} A={ab_intent['A_mean']:.3f} E={ab_intent['E_mean']:.3f} D={ab_intent['D_mean']:.3f}   "
          f"| B={ab_coef['B_mean']:.3f} A={ab_coef['A_mean']:.3f} E={ab_coef['E_mean']:.3f} D={ab_coef['D_mean']:.3f}   "
          f"| B={rB:.3f} A={rA:.3f}")
    rows.append({"K": K, "intent": ab_intent, "by_coef": ab_coef,
                 "random_B": rB, "random_A": rA, "random_E": rE, "random_D": rD})

with open('experiments/replication_study/results/ablation_curve.json', 'w') as f:
    json.dump({"rows": rows}, f, indent=2)

print()
print("=" * 80)
print("ATTRIBUTION ANALYSIS (logit-diff contribution)")
print("=" * 80)

mean_B = feats_B.mean(0)
mean_A = feats_A.mean(0)
mean_E = feats_E.mean(0)
mean_D = feats_D.mean(0)
coefs = clf_uni.coef_[0]

contribution_id = coefs * (mean_B - mean_A)
contribution_cmd = coefs * (mean_E - mean_D)

top_contrib_id = np.argsort(contribution_id)[::-1][:10]
top_contrib_cmd = np.argsort(contribution_cmd)[::-1][:10]

print("\nTop-10 features driving 'injection > benign' logit (IDENTITY B-A):")
print(f"  {'feat':>6} {'coef':>9} {'muA':>6} {'muB':>6} {'contrib':>9}")
for f in top_contrib_id:
    print(f"  {f:>6}  {coefs[f]:+.4f}  {mean_A[f]:5.2f}  {mean_B[f]:5.2f}  {contribution_id[f]:+.4f}")

print("\nTop-10 features driving 'injection > benign' logit (COMMAND E-D):")
print(f"  {'feat':>6} {'coef':>9} {'muD':>6} {'muE':>6} {'contrib':>9}")
for f in top_contrib_cmd:
    print(f"  {f:>6}  {coefs[f]:+.4f}  {mean_D[f]:5.2f}  {mean_E[f]:5.2f}  {contribution_cmd[f]:+.4f}")

print("\nTop identity contributors — category tags:")
for f in top_contrib_id:
    tags = []
    if f in r_id['top_intent_discriminators_strict'][:20]: tags.append("ID-intent")
    if f in r_id['top_fp_overlap_features'][:20]: tags.append("ID-overlap")
    if f in r_cmd['top_intent_discriminators'][:20]: tags.append("CMD-intent")
    if f in r_cmd['top_fp_overlap_features'][:20]: tags.append("CMD-overlap")
    print(f"  feat {f}: {', '.join(tags) if tags else '(uncategorized)'}")

print("\nTop command contributors — category tags:")
for f in top_contrib_cmd:
    tags = []
    if f in r_id['top_intent_discriminators_strict'][:20]: tags.append("ID-intent")
    if f in r_id['top_fp_overlap_features'][:20]: tags.append("ID-overlap")
    if f in r_cmd['top_intent_discriminators'][:20]: tags.append("CMD-intent")
    if f in r_cmd['top_fp_overlap_features'][:20]: tags.append("CMD-overlap")
    print(f"  feat {f}: {', '.join(tags) if tags else '(uncategorized)'}")

with open('experiments/replication_study/results/attribution.json', 'w') as f:
    json.dump({
        "top_contributors_identity": [
            {"feat": int(i), "coef": float(coefs[i]),
             "muA": float(mean_A[i]), "muB": float(mean_B[i]),
             "contribution": float(contribution_id[i])}
            for i in top_contrib_id
        ],
        "top_contributors_command": [
            {"feat": int(i), "coef": float(coefs[i]),
             "muD": float(mean_D[i]), "muE": float(mean_E[i]),
             "contribution": float(contribution_cmd[i])}
            for i in top_contrib_cmd
        ],
    }, f, indent=2)
