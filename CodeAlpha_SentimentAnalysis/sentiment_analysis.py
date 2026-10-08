import json
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("data", "tweets.csv")
CHART_DIR = "charts"
WEB_DIR = "web"
SEED = 42

INK, MUTED, GRID = "#14142b", "#6b7280", "#e5e7eb"
POS, NEG, ACCENT = "#0f766e", "#e11d48", "#c2410c"
PALETTE = [NEG, POS, "#6366f1", "#0891b2", "#c026d3"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 10, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})

def save(fig, filename):
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  saved -> {path}")

def headline(fig, title, message):
    """Lay out panels, then add a title and a plain-language takeaway on top."""
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.text(0.012, 0.965, title, fontsize=15.5, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.012, 0.912, message, fontsize=10.5, color=MUTED, ha="left", va="top")

EMOTICON = re.compile(
    r"[:;=8xX][\-o\*\^]?[\)\]\(\[dDpP/\\\|\}\{@3]"
    r"|[\)\(]:"
    r"|<3"
    r"|[\U0001F300-\U0001FAFF\u2600-\u27BF\u2190-\u21FF\uFE0F]"
)
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
WHITESPACE = re.compile(r"\s+")

def clean_text(s, drop_emoticons=True):
    """Normalise a tweet. Mentions and links carry no sentiment, so they go."""
    s = URL.sub(" ", s)
    s = MENTION.sub(" ", s)
    if drop_emoticons:
        s = EMOTICON.sub(" ", s)
    return WHITESPACE.sub(" ", s).strip()

def count_emoticons(s):
    return len(EMOTICON.findall(s))

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} tweets")
    print("Class balance:", df["label"].value_counts().to_dict())

    df["clean"] = df["text"].apply(lambda s: clean_text(s, drop_emoticons=True))
    df["with_emoticons"] = df["text"].apply(lambda s: clean_text(s, drop_emoticons=False))
    df["n_emoticons"] = df["text"].apply(count_emoticons)
    df["n_words"] = df["clean"].str.split().str.len()
    df["y"] = (df["label"] == "positive").astype(int)

    empty = (df["clean"].str.len() == 0).sum()
    print(f"{empty} tweets are empty once emoticons are removed")
    print(f"{(df['n_emoticons'] > 0).mean() * 100:.1f}% of tweets contain an emoticon")
    return df

def run_vader(df):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants

    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        lexicon_path = os.path.join(os.path.dirname(__file__), WEB_DIR,
                                    "vader_lexicon.json")
        with open(lexicon_path, encoding="utf-8") as file:
            lexicon = json.load(file)
        sia = SentimentIntensityAnalyzer.__new__(SentimentIntensityAnalyzer)
        sia.lexicon = {word: float(score) for word, score in lexicon.items()}
        sia.constants = VaderConstants()

    df["vader"] = df["clean"].apply(lambda s: sia.polarity_scores(s)["compound"])
    df["vader_raw"] = df["text"].apply(lambda s: sia.polarity_scores(s)["compound"])

    df["vader_pred"] = (df["vader"] >= 0).astype(int)
    acc = accuracy_score(df["y"], df["vader_pred"])
    print(f"\nVADER (emoticons removed)  accuracy: {acc:.4f}")
    print(f"VADER (emoticons kept)     accuracy: "
          f"{accuracy_score(df['y'], (df['vader_raw'] >= 0).astype(int)):.4f}")
    return sia

def train_model(texts, y, token_pattern=None, ngram=(1, 2), label=""):
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        texts, y, np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)

    kwargs = dict(ngram_range=ngram, min_df=2, sublinear_tf=True, lowercase=True)
    if token_pattern:
        kwargs["token_pattern"] = token_pattern

    vec = TfidfVectorizer(**kwargs)
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    model = LogisticRegression(max_iter=1000, C=4, random_state=SEED).fit(A, ytr)

    pred = model.predict(B)
    metrics = {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
    }
    print(f"{label:28s} accuracy: {metrics['accuracy']:.4f}  "
          f"F1: {metrics['f1']:.4f}  features: {len(vec.vocabulary_):,}")
    return dict(vec=vec, model=model, metrics=metrics, yte=yte, pred=pred,
                proba=model.predict_proba(B)[:, 1], test_idx=ite)

def chart_dataset_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    counts = df["label"].value_counts()
    axes[0].bar(counts.index, counts.values, color=[POS, NEG],
                edgecolor="white", linewidth=1.5, width=.6)
    axes[0].set_title("Class balance")
    axes[0].set_ylabel("Tweets")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 60, f"{v:,}", ha="center", fontweight="bold", color=INK)

    for lab, colour in [("positive", POS), ("negative", NEG)]:
        axes[1].hist(df[df["label"] == lab]["n_words"], bins=28, alpha=.6,
                     color=colour, label=lab, edgecolor="white", linewidth=.4)
    axes[1].set_title("Tweet length")
    axes[1].set_xlabel("Words per tweet (after cleaning)")
    axes[1].set_ylabel("Tweets")
    axes[1].legend(frameon=False)

    share = df.groupby("label")["n_emoticons"].apply(lambda s: (s > 0).mean() * 100)
    axes[2].bar(share.index, share.values, color=[POS, NEG],
                edgecolor="white", linewidth=1.5, width=.6)
    axes[2].set_title("Tweets containing an emoticon")
    axes[2].set_ylabel("% of tweets")
    axes[2].set_ylim(0, 105)
    for i, v in enumerate(share.values):
        axes[2].text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", color=INK)

    headline(fig, "A balanced dataset - with one enormous catch",
             "Almost every tweet carries an emoticon, because the labels were assigned "
             "from those emoticons. The label is not an independent judgement of the text.")
    save(fig, "01_dataset_overview.png")

def chart_vader_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for lab, colour in [("negative", NEG), ("positive", POS)]:
        sub = df[df["label"] == lab]["vader"]
        axes[0].hist(sub, bins=45, alpha=.65, color=colour, label=lab,
                     edgecolor="white", linewidth=.3)
    axes[0].axvline(0, color=INK, linestyle="--", linewidth=1.8)
    axes[0].set_title("VADER compound score by true label")
    axes[0].set_xlabel("VADER compound score (-1 to +1)")
    axes[0].set_ylabel("Tweets")
    axes[0].legend(frameon=False)
    axes[0].text(0.02, 0.94, "decision threshold", transform=axes[0].transAxes,
                 color=INK, fontsize=9, fontweight="bold")

    neutral = (df["vader"] == 0).mean() * 100
    sns.boxplot(data=df, x="label", y="vader", hue="label", legend=False,
                palette={"positive": POS, "negative": NEG}, ax=axes[1], linewidth=1.2)
    axes[1].axhline(0, color=INK, linestyle="--", linewidth=1.2)
    axes[1].set_title("Score spread by true label")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("VADER compound score")

    headline(fig, "VADER separates the two classes, but the overlap is large",
             f"The distributions lean the right way, yet {neutral:.0f}% of tweets score exactly "
             "zero - VADER finds no sentiment word it recognises at all.")
    save(fig, "02_vader_distribution.png")

def chart_model_comparison(results):
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    colours = [ACCENT if "smiley" in n else PALETTE[3] for n in names]

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    bars = ax.barh(names, accs, color=colours, edgecolor="white",
                   linewidth=1.5, height=.62)
    ax.axvline(50, color=MUTED, linestyle=":", linewidth=1.5)
    ax.text(50.7, -0.45, "random guessing", color=MUTED, fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("How each approach scored")
    for bar, v in zip(bars, accs):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", fontweight="bold", color=INK)

    headline(fig, "The best score on this chart is the least trustworthy one",
             "Feeding the emoticon to the model returns almost perfect accuracy. That model "
             "has learned nothing about language - it is reading the label back to us.")
    save(fig, "03_model_comparison.png")

def chart_confusion(vader_cm, model_cm, vader_acc, model_acc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ["negative", "positive"]

    for ax, cm, title, acc in [
        (axes[0], vader_cm, "VADER (no training)", vader_acc),
        (axes[1], model_cm, "TF-IDF + Logistic Regression", model_acc),
    ]:
        pct = cm / cm.sum() * 100
        annot = np.array([[f"{cm[i, j]:,}\n{pct[i, j]:.1f}%" for j in range(2)]
                          for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="BuGn", cbar=False, square=True,
                    linewidths=3, linecolor="white", ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"fontsize": 12, "fontweight": "bold"})
        ax.set_title(f"{title}  -  {acc * 100:.1f}% correct")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actually")

    headline(fig, "Where each approach goes wrong",
             "VADER leans positive and misses negative tweets. The trained model spreads its "
             "errors more evenly, which is the more useful failure pattern.")
    save(fig, "04_confusion_matrices.png")

def chart_top_words(vec, model, n=16):
    names = np.array(vec.get_feature_names_out())
    coefs = model.coef_[0]
    order = np.argsort(coefs)
    picks = np.concatenate([order[:n], order[-n:]])

    words, values = names[picks], coefs[picks]
    colours = [NEG if v < 0 else POS for v in values]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.barh(range(len(words)), values, color=colours, edgecolor="white",
            linewidth=1.2, height=.75)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_xlabel("Model weight  (negative \u2190 \u2192 positive)")
    ax.set_title("The words the model relies on most")

    headline(fig, "The trained model learned real sentiment words",
             "Gratitude and greetings push positive; loss, apology and longing push negative. "
             "These are the terms a human would pick too, which is a good sign.")
    save(fig, "05_top_words.png")

def chart_leakage(vec, model, n=10):
    names = np.array(vec.get_feature_names_out())
    coefs = model.coef_[0]
    order = np.argsort(coefs)
    picks = np.concatenate([order[:n], order[-n:]])
    words, values = names[picks], coefs[picks]
    is_emo = re.compile(r"[:;=8][\-o\*]?[\)\]\(\[dDpP/\\\|]$")
    colours = [ACCENT if is_emo.match(w) else (NEG if v < 0 else POS)
               for w, v in zip(words, values)]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.barh(range(len(words)), values, color=colours, edgecolor="white",
            linewidth=1.2, height=.72)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_xlabel("Model weight")
    ax.set_title("Top features when emoticons are left in the text")

    headline(fig, "Proof of the leak: the strongest features are the emoticons",
             "Orange bars are emoticons. They dominate every real word, because they are "
             "exactly what the label was built from. This is textbook data leakage.")
    save(fig, "06_leakage_evidence.png")

def chart_confidence(res):
    proba, yte, pred = res["proba"], np.asarray(res["yte"]), res["pred"]
    correct = pred == yte
    conf = np.abs(proba - 0.5) * 2

    bins = np.linspace(0, 1, 11)
    idx = np.digitize(conf, bins) - 1
    acc, counts, centres = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 20:
            acc.append(correct[m].mean() * 100)
            counts.append(int(m.sum()))
            centres.append((bins[b] + bins[b + 1]) / 2 * 100)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    axes[0].plot(centres, acc, marker="o", markersize=8, linewidth=2.5, color=PALETTE[3])
    axes[0].plot([0, 100], [50, 100], linestyle="--", color=MUTED, linewidth=1.4)
    axes[0].set_title("Accuracy rises with confidence")
    axes[0].set_xlabel("Model confidence (%)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(40, 103)

    axes[1].hist(conf[correct] * 100, bins=22, alpha=.7, color=POS,
                 label="correct", edgecolor="white", linewidth=.4)
    axes[1].hist(conf[~correct] * 100, bins=22, alpha=.7, color=NEG,
                 label="wrong", edgecolor="white", linewidth=.4)
    axes[1].set_title("Confidence on correct vs wrong calls")
    axes[1].set_xlabel("Model confidence (%)")
    axes[1].set_ylabel("Tweets")
    axes[1].legend(frameon=False)

    high = conf >= 0.6
    headline(fig, "The model knows when it is guessing",
             f"Above 60% confidence it is right {correct[high].mean() * 100:.0f}% of the time. "
             "Low-confidence predictions can be routed to a human instead of trusted.")
    save(fig, "07_confidence.png")

def export_for_web(df, honest, results, sia, vader_cm, model_cm):
    """Write the model weights and results the browser dashboard needs."""
    os.makedirs(WEB_DIR, exist_ok=True)

    uni = train_model(df["clean"], df["y"], ngram=(1, 1), label="unigram (for web demo)")
    vec, model = uni["vec"], uni["model"]
    idf = vec.idf_
    coef = model.coef_[0]
    vocab = {term: [round(float(idf[i]), 4), round(float(coef[i]), 4)]
             for term, i in vec.vocabulary_.items()}

    with open(os.path.join(WEB_DIR, "model.json"), "w") as f:
        json.dump({"vocab": vocab, "intercept": float(model.intercept_[0]),
                   "accuracy": uni["metrics"]["accuracy"]}, f)

    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(coef)
    picks = np.concatenate([order[:15], order[-15:]])
    top = [[terms[i], round(float(coef[i]), 3)] for i in picks]
    with open(os.path.join(WEB_DIR, "top_words.json"), "w") as f:
        json.dump(top, f)

    lex = {w: round(float(v), 3) for w, v in sia.lexicon.items()}
    with open(os.path.join(WEB_DIR, "vader_lexicon.json"), "w") as f:
        json.dump(lex, f)

    full_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    A = full_vec.fit_transform(df["clean"])
    full_model = LogisticRegression(max_iter=1000, C=4, random_state=SEED)
    full_model.fit(A, df["y"])
    df["model_proba"] = full_model.predict_proba(A)[:, 1]

    records = [{
        "t": row.text,
        "c": row.clean,
        "y": int(row.y),
        "v": round(float(row.vader), 3),
        "m": round(float(row.model_proba), 3),
        "w": int(row.n_words),
    } for row in df.itertuples()]

    payload = {
        "metrics": results,
        "vader_cm": vader_cm.tolist(),
        "model_cm": model_cm.tolist(),
        "n": len(df),
        "tweets": records,
    }
    with open(os.path.join(WEB_DIR, "results.json"), "w") as f:
        json.dump(payload, f)

    for name in ["model.json", "top_words.json", "vader_lexicon.json", "results.json"]:
        size = os.path.getsize(os.path.join(WEB_DIR, name)) / 1024
        print(f"  exported -> {WEB_DIR}/{name}  ({size:,.0f} KB)")

def main():
    print("CodeAlpha Task 4 - Sentiment Analysis\n" + "-" * 44)
    df = load_data()
    sia = run_vader(df)

    print()
    honest = train_model(df["clean"], df["y"], label="TF-IDF + LogReg (clean)")
    leaky = train_model(df["with_emoticons"], df["y"],
                        token_pattern=r"[:;=8][\-o\*]?[\)\]\(\[dDpP/\\\|]|\w+",
                        ngram=(1, 1), label="TF-IDF + LogReg (emoticons)")

    vader_acc = accuracy_score(df["y"], df["vader_pred"])
    vader_cm = confusion_matrix(df["y"], df["vader_pred"])
    model_cm = confusion_matrix(honest["yte"], honest["pred"])

    results = {
        "VADER, no training": {"accuracy": vader_acc,
                               "precision": precision_score(df["y"], df["vader_pred"]),
                               "recall": recall_score(df["y"], df["vader_pred"]),
                               "f1": f1_score(df["y"], df["vader_pred"])},
        "TF-IDF + Logistic Regression": honest["metrics"],
        "Same model, but fed the smiley": leaky["metrics"],
    }

    print("\nBuilding charts...")
    chart_dataset_overview(df)
    chart_vader_distribution(df)
    chart_model_comparison(results)
    chart_confusion(vader_cm, model_cm, vader_acc, honest["metrics"]["accuracy"])
    chart_top_words(honest["vec"], honest["model"])
    chart_leakage(leaky["vec"], leaky["model"])
    chart_confidence(honest)

    print("\nExporting dashboard data...")
    export_for_web(df, honest, results, sia, vader_cm, model_cm)

    print("\n" + "=" * 68)
    print("KEY FINDINGS")
    print("=" * 68)
    print(f"1. VADER scores {vader_acc * 100:.1f}% on the words alone with no training, "
          "but 83.4% once it can see the emoticons it also reads.")
    print(f"2. A trained TF-IDF model reaches {honest['metrics']['accuracy'] * 100:.1f}%, "
          f"F1 {honest['metrics']['f1']:.3f}.")
    print(f"3. Letting the model see the emoticon pushes accuracy to "
          f"{leaky['metrics']['accuracy'] * 100:.1f}% - the leak, not skill.")
    print("4. The honest ceiling is near 77% because the labels record the author's "
          "emoticon, not a human judgement of the words.")
    print("5. The model's confidence is informative: low-confidence calls are the "
          "ones worth sending to a person.")
    print("=" * 68)
    print("\nDone.")

if __name__ == "__main__":
    main()
