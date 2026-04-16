"""
Stretch 5A-S2 — Learning Curves Diagnostic
Module 5 Week A | Honors Track

Diagnoses bias vs. variance for logistic regression models on the
telecom churn dataset using sklearn's learning_curve utility.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.dummy import DummyClassifier

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service", "payment_method"]

TRAIN_SIZES = np.linspace(0.10, 1.0, 9)  # 9 sizes: 10% -> 100%

PALETTE = {
    r"LR — weak regularization (C=10)": ("#E05252", "#F4A0A0"),
    r"LR — default regularization (C=1)": ("#4A90D9", "#9AC4F0"),
    r"LR — strong regularization (C=0.01)": ("#50C878", "#A8E8BE"),
    "Dummy — most_frequent": ("#AAAAAA", "#DDDDDD"),
    "Dummy — stratified": ("#CC88FF", "#EEC4FF"),
}


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features
        and y is a Series of the target (churned).
    """
    df = pd.read_csv(filepath)
    X = df.drop(columns=["customer_id", "churned"])
    y = df["churned"]
    return (X, y)


def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def define_models():
    """Define the 5 model configurations to compare.

    Two dummy baselines are included to teach two different lessons:
    most_frequent demonstrates the accuracy inflation problem on imbalanced
    data; stratified shows what random guessing in proportion to class
    frequencies looks like, so F1 carries meaningful signal when comparing.

    Returns:
        Dictionary mapping model name to a fitted-ready Pipeline.
    """
    models = {
        r"LR — weak regularization (C=10)": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=10, max_iter=1000, random_state=42, class_weight="balanced"
                    ),
                ),
            ]
        ),
        r"LR — default regularization (C=1)": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=1, max_iter=1000, random_state=42, class_weight="balanced"
                    ),
                ),
            ]
        ),
        r"LR — strong regularization (C=0.01)": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=0.01, max_iter=1000, random_state=42, class_weight="balanced"
                    ),
                ),
            ]
        ),
        "Dummy — most_frequent": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
            ]
        ),
        "Dummy — stratified": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", DummyClassifier(strategy="stratified", random_state=42)),
            ]
        ),
    }
    return models


def compute_learning_curves(models, X, y, train_sizes=TRAIN_SIZES, n_splits=5):
    """Run sklearn learning_curve for every model in the dict.

    Args:
        models:       Dict of {label: pipeline} from define_models().
        X:            Feature DataFrame.
        y:            Target Series.
        train_sizes:  Array of fractional training sizes.
        n_splits:     Number of stratified CV folds.

    Returns:
        Dict mapping each model label to a results dict with keys:
        'sizes', 'train_mean', 'train_std', 'val_mean', 'val_std'.
    """
    # Macro F1 avoids accuracy inflation on the imbalanced churn dataset.
    # A majority-class classifier would score ~86% accuracy but ~46% macro F1,
    # making the baseline clearly visible in the plot.
    scorer = make_scorer(f1_score, average="macro", zero_division=0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for label, pipeline in models.items():
        print(f"  Computing: {label} ...")
        sizes_abs, train_scores, val_scores = learning_curve(
            pipeline,
            X,
            y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            shuffle=True,
            random_state=42,
        )
        results[label] = {
            "sizes": sizes_abs,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "val_mean": val_scores.mean(axis=1),
            "val_std": val_scores.std(axis=1),
        }
        gap = results[label]["train_mean"][-1] - results[label]["val_mean"][-1]
        print(
            f"    train F1={results[label]['train_mean'][-1]:.3f} | "
            f"val F1={results[label]['val_mean'][-1]:.3f} | "
            f"gap={gap:.3f}"
        )
    return results


def _style_ax(ax, bg, grid, text):
    """Apply shared dark-theme styling to an Axes object."""
    ax.set_facecolor(bg)
    ax.tick_params(colors=text, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid)
    ax.grid(True, color=grid, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_ylim(0.25, 1.02)


def plot_learning_curves(results, output_path="learning_curves.png"):
    """Render and save the full learning curve figure.

    Top row: one subplot per LR model showing training vs. validation
    with +/- 1 SD shaded bands and gap annotation.
    Bottom-left: overlay of all LR train + val curves for side-by-side comparison.
    Bottom-right: validation-only panel including dummy baselines so the reader
    can see how far above random the LR models land.

    Args:
        results:     Output of compute_learning_curves().
        output_path: Filepath for the saved PNG.
    """
    # ── Style constants ───────────────────────────────────────────────────────
    BG = "#0F1117"
    AXIS_BG = "#181C27"
    GRID_COL = "#2A2F42"
    TEXT_COL = "#D4D8E8"
    TITLE = "#FFFFFF"
    VAL_COL = "#F5C842"
    VAL_FILL = "#F5E4A0"

    lr_labels = [k for k in results if k.startswith("LR")]
    all_labels = list(results.keys())

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.88,
        bottom=0.09,
    )

    # ── Top row: one subplot per LR model ────────────────────────────────────
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, label in zip(top_axes, lr_labels):
        res = results[label]
        train_col, train_fill = PALETTE[label]
        sizes = res["sizes"]

        # Shaded +/- 1 SD bands
        ax.fill_between(
            sizes,
            res["train_mean"] - res["train_std"],
            res["train_mean"] + res["train_std"],
            alpha=0.22,
            color=train_fill,
        )
        ax.fill_between(
            sizes,
            res["val_mean"] - res["val_std"],
            res["val_mean"] + res["val_std"],
            alpha=0.22,
            color=VAL_FILL,
        )

        # Mean lines
        ax.plot(
            sizes,
            res["train_mean"],
            "o-",
            color=train_col,
            lw=2.2,
            markersize=5,
            label="Training F1",
            zorder=5,
        )
        ax.plot(
            sizes,
            res["val_mean"],
            "s--",
            color=VAL_COL,
            lw=2.2,
            markersize=5,
            label="Validation F1",
            zorder=5,
        )

        # Annotate final gap
        gap = res["train_mean"][-1] - res["val_mean"][-1]
        mid_y = (res["train_mean"][-1] + res["val_mean"][-1]) / 2
        ax.annotate(
            f"gap={gap:.3f}",
            xy=(sizes[-1], mid_y),
            xytext=(-58, 0),
            textcoords="offset points",
            fontsize=8,
            color="#FFCC44",
            arrowprops=dict(arrowstyle="-", color="#FFCC44", lw=0.8),
        )

        _style_ax(ax, AXIS_BG, GRID_COL, TEXT_COL)
        short = label.split("(")[1].rstrip(")")
        ax.set_title(short, color=TITLE, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Training set size", color=TEXT_COL, fontsize=8)
        ax.set_ylabel("Macro F1 score", color=TEXT_COL, fontsize=8)
        ax.legend(
            fontsize=7.5,
            facecolor=AXIS_BG,
            edgecolor=GRID_COL,
            labelcolor=TEXT_COL,
            loc="lower right",
        )

    # ── Bottom-left: LR models — train + validation overlay ──────────────────
    ax_overlay = fig.add_subplot(gs[1, :2])

    for label in lr_labels:
        res = results[label]
        col, fill = PALETTE[label]
        short = label.split("(")[1].rstrip(")")
        ax_overlay.plot(
            res["sizes"],
            res["train_mean"],
            "o-",
            color=col,
            lw=2,
            markersize=4,
            label=f"Train {short}",
        )
        ax_overlay.plot(
            res["sizes"],
            res["val_mean"],
            "s--",
            color=col,
            lw=1.6,
            alpha=0.75,
            label=f"Val   {short}",
        )
        ax_overlay.fill_between(
            res["sizes"],
            res["val_mean"] - res["val_std"],
            res["val_mean"] + res["val_std"],
            alpha=0.12,
            color=col,
        )

    _style_ax(ax_overlay, AXIS_BG, GRID_COL, TEXT_COL)
    ax_overlay.set_title(
        "LR Models — Train & Validation Overlay",
        color=TITLE,
        fontsize=10,
        fontweight="bold",
        pad=6,
    )
    ax_overlay.set_xlabel("Training set size", color=TEXT_COL, fontsize=9)
    ax_overlay.set_ylabel("Macro F1", color=TEXT_COL, fontsize=9)
    ax_overlay.legend(
        fontsize=7, facecolor=AXIS_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, ncol=2
    )

    # ── Bottom-right: validation only — all models including dummies ──────────
    ax_all = fig.add_subplot(gs[1, 2])

    for label in all_labels:
        res = results[label]
        col, fill = PALETTE[label]
        ls = "--" if label.startswith("Dummy") else "-"
        raw = label.replace("LR — ", "").replace("Dummy — ", "")
        short = raw.split("(")[1].rstrip(")") if "(" in raw else raw
        ax_all.plot(
            res["sizes"],
            res["val_mean"],
            marker="s",
            ls=ls,
            color=col,
            lw=1.8,
            markersize=4,
            label=short,
        )
        ax_all.fill_between(
            res["sizes"],
            res["val_mean"] - res["val_std"],
            res["val_mean"] + res["val_std"],
            alpha=0.10,
            color=col,
        )

    _style_ax(ax_all, AXIS_BG, GRID_COL, TEXT_COL)
    ax_all.set_title(
        "Validation F1 — All Models\n(dashed = dummy baselines)",
        color=TITLE,
        fontsize=9.5,
        fontweight="bold",
        pad=6,
    )
    ax_all.set_xlabel("Training set size", color=TEXT_COL, fontsize=9)
    ax_all.set_ylabel("Macro F1", color=TEXT_COL, fontsize=9)
    ax_all.legend(
        fontsize=7, facecolor=AXIS_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL
    )

    # ── Master title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Learning Curves — Logistic Regression on Telecom Churn\n"
        "Stratified 5-Fold CV  |  Scoring: Macro F1  |  Shaded = +/- 1 SD across folds",
        color=TITLE,
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    plt.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    print(f"\nPlot saved -> {output_path}")
    plt.close()


def print_summary(results):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 72)
    print(f"{'Model':<42} {'Train F1':>9} {'Val F1':>8} {'Gap':>7}")
    print("=" * 72)
    for label, res in results.items():
        gap = res["train_mean"][-1] - res["val_mean"][-1]
        print(
            f"{label:<42} "
            f"{res['train_mean'][-1]:>9.3f} "
            f"{res['val_mean'][-1]:>8.3f} "
            f"{gap:>7.3f}"
        )
    print("=" * 72)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data ...")
    X, y = load_and_prepare("data/telecom_churn.csv")
    print(f"  {X.shape[0]} samples | {X.shape[1]} features")
    print(f"  Churn rate: {y.mean():.1%}  -> using Macro F1 (not accuracy)\n")

    print("Defining models ...")
    models = define_models()

    print("Computing learning curves ...")
    results = compute_learning_curves(models, X, y)

    print_summary(results)

    print("\nRendering plot ...")
    plot_learning_curves(results, output_path="learning_curves.png")
"""
Bias–Variance Tradeoff Analysis via Learning Curves in an Imbalanced Dataset:

The learning curves provide a clear diagnostic of the model’s behavior
and indicate that the logistic regression model is primarily affected 
by **high bias (underfitting)** rather than high variance. 
This conclusion is based on two key observations. First, both the training
and validation F1 scores converge to relatively low values (approximately 0.54–0.57),
indicating that the model is unable to achieve high performance even on the training
data. Second, the gap between the training and validation curves is consistently
small across all training set sizes, which suggests that the model is not overfitting
and is generalizing similarly on both seen and unseen data.

At smaller training sizes, the training F1 score is initially higher, which is expected
due to mild overfitting when the model has access to limited data. However, as the training 
set size increases, the training score decreases and stabilizes, while the validation score 
increases slightly and then plateaus. This convergence behavior reflects the model 
transitioning from memorization to generalization. The key observation is that both 
curves stabilize at a relatively low performance level, reinforcing the conclusion that 
the model lacks sufficient capacity to capture the underlying patterns in the data.

Another important aspect of the learning curves is the behavior of the validation score as 
more data is added. While there is a slight improvement in validation performance at the 
beginning, the curve quickly flattens, indicating diminishing returns from additional data. 
This plateau suggests that the model has already learned as much as it can from the available
feature representation, and further increasing the dataset size is unlikely to result in 
significant performance gains. In other words, the limitation is not due to insufficient 
data, but rather due to the simplicity of the model.

The comparison between different regularization strengths (C values) further supports this 
interpretation. Models with stronger regularization (e.g., C=0.01) exhibit even lower 
training and validation scores, which is characteristic of increased bias. On the other 
hand, reducing regularization (e.g., C=10) slightly increases training performance but does 
not significantly improve validation performance, indicating that simply relaxing 
regularization is not sufficient to overcome the model’s limitations. 
The consistent convergence across all configurations highlights that the issue is structural 
rather than parametric.

Additionally, when compared to the dummy baselines, the logistic regression models outperform 
both the “most frequent” and “stratified” classifiers, confirming that the model is learning 
meaningful patterns beyond random or trivial predictions. However, the performance margin is 
relatively modest, suggesting that while the model captures some signal, it fails to fully 
exploit the available information in the dataset.

Given these observations, increasing model complexity is likely to be beneficial. 
Logistic regression is inherently a linear model, which restricts it to learning 
linear decision boundaries. If the relationship between features and the target 
variable is non-linear, the model will systematically underfit. One potential 
improvement is to introduce polynomial features, which allow the model to capture non-linear 
interactions between variables. Alternatively, more flexible models such as decision trees, 
random forests, or gradient boosting methods can be used, as they are better suited for 
modeling complex, non-linear relationships.

In conclusion, the learning curves strongly indicate that the model is limited by 
high bias. Collecting more data is unlikely to significantly improve performance, 
as the validation curve has already plateaued. The most effective next step is to 
increase model capacity through feature engineering or by adopting more expressive models, 
which can better capture the underlying structure of the telecom churn dataset and improve 
predictive performance.

"""
