"""
SYDE 572 - Assignment 4: Text Classification with Naive Bayes
Author: Tze Shuo (21291638)

This script implements:
1. Pure Python Multinomial Naive Bayes classifier from scratch (No ML toolkits).
2. Add-alpha (Laplace) smoothing and numerically stable log-space inference.
3. Sanity checks on sample phrases.
4. Evaluation and accuracy calculation on the 20% test split.
5. In-depth error analysis (false positives and false negatives).
6. Optional Task 1: Learning curve (training size vs. test accuracy).
7. Optional Task 2: Most informative features (log-odds ratio visualization).
8. Optional Task 3: Domain sensitivity / cross-domain adaptation.
9. Optional Task 4: Feature improvements (Bigrams and Negation tagging).
10. Optional Task 5: Scikit-learn LinearSVC benchmark comparison.
"""

from collections import Counter
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight"
})

np.random.seed(42)

BASE_DIR = Path(__file__).parent
candidate_paths = [
    BASE_DIR / "SYDE572 assn4 file" / "all_sentiment_shuffled.txt",
    BASE_DIR / "all_sentiment_shuffled.txt",
    Path("SYDE572 assn4 file/all_sentiment_shuffled.txt"),
    Path("all_sentiment_shuffled.txt"),
    BASE_DIR.parent / "4" / "SYDE572 assn4 file" / "all_sentiment_shuffled.txt",
    BASE_DIR.parent / "4" / "all_sentiment_shuffled.txt"
]
DATA_FILE = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def read_documents(doc_file, return_categories=False):
    """
    Reads the sentiment dataset.
    Line format: <category> <sentiment> <doc_id> <word1> <word2> ...
    Returns (docs, labels) matching the assignment PDF specification.
    If return_categories=True, returns (docs, labels, categories).
    """
    docs = []
    labels = []
    categories = []
    with open(doc_file, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 3:
                continue
            categories.append(words[0])
            labels.append(words[1])
            docs.append(words[3:])
    if return_categories:
        return docs, labels, categories
    return docs, labels


def train_nb(documents, labels, alpha=1.0):
    """
    Estimates Naive Bayes parameters from training documents and labels.
    Uses add-alpha (Laplace) smoothing and precomputes log probabilities.
    """
    num_docs = len(documents)
    class_counts = Counter(labels)
    classes = list(class_counts.keys())

    log_priors = {c: math.log(count / num_docs) for c, count in class_counts.items()}

    word_counts = {c: Counter() for c in classes}
    total_words = {c: 0 for c in classes}
    vocab = set()

    for doc, label in zip(documents, labels):
        word_counts[label].update(doc)
        total_words[label] += len(doc)
        vocab.update(doc)

    vocab_size = len(vocab)

    denominators = {c: total_words[c] + alpha * (vocab_size + 1) for c in classes}

    log_likelihoods = {c: {} for c in classes}
    for c in classes:
        denom = denominators[c]
        for w, count in word_counts[c].items():
            log_likelihoods[c][w] = math.log((count + alpha) / denom)

    log_unk = {c: math.log(alpha / denominators[c]) for c in classes}

    return {
        "classes": classes,
        "class_counts": class_counts,
        "log_priors": log_priors,
        "word_counts": word_counts,
        "total_words": total_words,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "alpha": alpha,
        "denominators": denominators,
        "log_likelihoods": log_likelihoods,
        "log_unk": log_unk
    }


def score_doc_label(document, label, model):
    """
    Applies the Naive Bayes formula to compute the log joint probability:
    log P(c, document) = log P(c) + sum_{w in document} log P(w | c)
    """
    log_prob = model["log_priors"][label]
    class_log_likes = model["log_likelihoods"][label]
    unk_log_prob = model["log_unk"][label]

    for w in document:
        log_prob += class_log_likes.get(w, unk_log_prob)

    return log_prob


def classify_nb(document, model):
    """
    Classifies a single document by returning the class with highest log joint probability.
    """
    best_class = None
    best_score = -float("inf")
    for c in model["classes"]:
        score = score_doc_label(document, c, model)
        if score > best_score:
            best_score = score
            best_class = c
    return best_class


def classify_documents(docs, model):
    """
    Classifies a collection of documents.
    """
    return [classify_nb(doc, model) for doc in docs]


def accuracy(true_labels, guessed_labels):
    """
    Computes classification accuracy (fraction of correctly classified documents).
    """
    correct = sum(1 for t, g in zip(true_labels, guessed_labels) if t == g)
    return correct / len(true_labels)


def run_sanity_checks(model):
    print("\n" + "=" * 70)
    print("Sanity Checks")
    print("=" * 70)

    doc_great = ["great"]
    log_pos_great = score_doc_label(doc_great, "pos", model)
    log_neg_great = score_doc_label(doc_great, "neg", model)
    prob_pos_great = math.exp(log_pos_great)
    prob_neg_great = math.exp(log_neg_great)

    print(f"Document ['great']:")
    print(f"  P(pos, ['great']) = {prob_pos_great:.6f} (log = {log_pos_great:.4f})")
    print(f"  P(neg, ['great']) = {prob_neg_great:.6f} (log = {log_neg_great:.4f})")
    assert prob_pos_great > prob_neg_great, "Sanity check failed: 'great' should be more positive than negative!"
    print("  => PASS: P(pos, ['great']) > P(neg, ['great'])")

    doc_bad = ["bad"]
    log_pos_bad = score_doc_label(doc_bad, "pos", model)
    log_neg_bad = score_doc_label(doc_bad, "neg", model)
    prob_pos_bad = math.exp(log_pos_bad)
    prob_neg_bad = math.exp(log_neg_bad)

    print(f"\nDocument ['bad']:")
    print(f"  P(pos, ['bad']) = {prob_pos_bad:.6f} (log = {log_pos_bad:.4f})")
    print(f"  P(neg, ['bad']) = {prob_neg_bad:.6f} (log = {log_neg_bad:.4f})")
    assert prob_neg_bad > prob_pos_bad, "Sanity check failed: 'bad' should be more negative than positive!"
    print("  => PASS: P(neg, ['bad']) > P(pos, ['bad'])")

    doc_rare = ["a", "top-quality", "performance"]
    try:
        log_pos_rare = score_doc_label(doc_rare, "pos", model)
        log_neg_rare = score_doc_label(doc_rare, "neg", model)
        guess_rare = classify_nb(doc_rare, model)
        print(f"\nDocument {doc_rare}:")
        print(f"  log P(pos, d) = {log_pos_rare:.4f}, log P(neg, d) = {log_neg_rare:.4f}")
        print(f"  Guessed label = {guess_rare}")
        print("  => PASS: score_doc_label executed successfully without crashing.")
    except Exception as e:
        print(f"  => FAIL: score_doc_label crashed with error: {e}")
        raise e


def run_error_analysis(eval_docs, eval_labels, predictions, model, num_examples=4):
    print("\n" + "=" * 70)
    print("Error Analysis: Examining Misclassified Reviews")
    print("=" * 70)

    false_positives = []
    false_negatives = []

    for idx, (doc, true_label, pred_label) in enumerate(zip(eval_docs, eval_labels, predictions)):
        if true_label != pred_label:
            score_pos = score_doc_label(doc, "pos", model)
            score_neg = score_doc_label(doc, "neg", model)
            diff = abs(score_pos - score_neg)
            item = {
                "idx": idx,
                "text": " ".join(doc),
                "length": len(doc),
                "true": true_label,
                "pred": pred_label,
                "score_pos": score_pos,
                "score_neg": score_neg,
                "diff": diff
            }
            if true_label == "neg" and pred_label == "pos":
                false_positives.append(item)
            elif true_label == "pos" and pred_label == "neg":
                false_negatives.append(item)

    false_positives.sort(key=lambda x: x["diff"], reverse=True)
    false_negatives.sort(key=lambda x: x["diff"], reverse=True)

    print(f"Total misclassifications: {len(false_positives) + len(false_negatives)}")
    print(f"  False Positives (True Neg, Predicted Pos): {len(false_positives)}")
    print(f"  False Negatives (True Pos, Predicted Neg): {len(false_negatives)}")

    print("\nTop False Positives (Expected NEG, Predicted POS - Largest Log-Score Margin):")
    for i, ex in enumerate(false_positives[:num_examples]):
        print(f"  [{i+1}] (Length: {ex['length']} words | diff: {ex['diff']:.2f}):")
        snippet = ex["text"][:140] + ("..." if len(ex["text"]) > 140 else "")
        print(f"      \"{snippet}\"")

    print("\nTop False Negatives (Expected POS, Predicted NEG - Largest Log-Score Margin):")
    for i, ex in enumerate(false_negatives[:num_examples]):
        print(f"  [{i+1}] (Length: {ex['length']} words | diff: {ex['diff']:.2f}):")
        snippet = ex["text"][:140] + ("..." if len(ex["text"]) > 140 else "")
        print(f"      \"{snippet}\"")

    return false_positives, false_negatives


def run_learning_curve(train_docs, train_labels, eval_docs, eval_labels):
    print("\n" + "=" * 70)
    print("Optional Task 1: Computing Learning Curve")
    print("=" * 70)

    fractions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    train_sizes = []
    accuracies = []

    for frac in fractions:
        n_samples = max(10, int(frac * len(train_docs)))
        sub_docs = train_docs[:n_samples]
        sub_labels = train_labels[:n_samples]

        sub_model = train_nb(sub_docs, sub_labels, alpha=1.0)
        preds = classify_documents(eval_docs, sub_model)
        acc = accuracy(eval_labels, preds)

        train_sizes.append(n_samples)
        accuracies.append(acc)
        print(f"  Training Size: {n_samples:5d} ({frac*100:5.1f}%) -> Test Accuracy: {acc*100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, [a * 100 for a in accuracies], marker="o", color="#176b68", linewidth=2.2, label="Test Accuracy")
    ax.set_xlabel("Number of Training Documents")
    ax.set_ylabel("Accuracy on Test Set (%)")
    ax.set_title("Optional Task 1: Naive Bayes Learning Curve")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(min(accuracies) * 100 - 3, max(accuracies) * 100 + 3)

    final_acc = accuracies[-1] * 100
    ax.annotate(f"Final: {final_acc:.2f}%",
                xy=(train_sizes[-1], final_acc),
                xytext=(-60, -20), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="#dceceb", ec="#176b68", lw=1),
                arrowprops=dict(arrowstyle="->", color="#176b68"))

    ax.legend(loc="lower right")
    plt.tight_layout()
    plot_path = MEDIA_DIR / "learning_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Learning Curve plot to: {plot_path}")
    return train_sizes, accuracies


def print_top_nb(arg1, arg2=None, k=15, alpha=10.0, min_freq=5):
    """
    Finds the words most indicative of positive and negative classes.
    Accepts print_top_nb(k, model) matching the PDF prompt, or print_top_nb(model, k).
    Uses log-odds ratio: log( P(w | pos) / P(w | neg) )
    Applies higher smoothing and frequency filter to prevent rare word noise.
    """
    if isinstance(arg1, int):
        k = arg1
        model = arg2
    else:
        model = arg1
        if isinstance(arg2, int):
            k = arg2
    print("\n" + "=" * 70)
    print(f"Optional Task 2: Top {k} Most Informative Features")
    print("=" * 70)

    vocab = model["vocab"]
    word_counts = model["word_counts"]
    total_words = model["total_words"]
    vocab_size = len(vocab)

    denom_pos = total_words["pos"] + alpha * (vocab_size + 1)
    denom_neg = total_words["neg"] + alpha * (vocab_size + 1)

    log_odds = {}
    for w in vocab:
        c_pos = word_counts["pos"][w]
        c_neg = word_counts["neg"][w]
        if (c_pos + c_neg) < min_freq:
            continue
        p_pos = (c_pos + alpha) / denom_pos
        p_neg = (c_neg + alpha) / denom_neg
        log_odds[w] = math.log(p_pos / p_neg)

    sorted_words = sorted(log_odds.items(), key=lambda x: x[1])
    top_negative = sorted_words[:k]
    top_positive = sorted_words[-k:][::-1]

    print(f"{'Rank':<5} | {'Top Positive Words':<22} {'Log-Odds':<10} | {'Top Negative Words':<22} {'Log-Odds':<10}")
    print("-" * 75)
    for i in range(k):
        pos_w, pos_score = top_positive[i]
        neg_w, neg_score = top_negative[i]
        print(f"{i+1:<5} | {pos_w:<22} {pos_score:+8.3f}   | {neg_w:<22} {neg_score:+8.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    pos_words = [x[0] for x in top_positive[::-1]]
    pos_vals = [x[1] for x in top_positive[::-1]]
    ax1.barh(pos_words, pos_vals, color="#2ca02c", alpha=0.85)
    ax1.set_xlabel("Log-Odds Ratio log[P(w|pos) / P(w|neg)]")
    ax1.set_title(f"Top {k} Positive Features")
    ax1.grid(True, linestyle=":", alpha=0.4, axis="x")

    neg_words = [x[0] for x in top_negative]
    neg_vals = [abs(x[1]) for x in top_negative]
    ax2.barh(neg_words, neg_vals, color="#d62728", alpha=0.85)
    ax2.set_xlabel("Absolute Negative Log-Odds log[P(w|neg) / P(w|pos)]")
    ax2.set_title(f"Top {k} Negative Features")
    ax2.grid(True, linestyle=":", alpha=0.4, axis="x")

    plt.suptitle("Optional Task 2: Most Informative Words for Sentiment Classification", fontsize=14)
    plt.tight_layout()
    plot_path = MEDIA_DIR / "informative_features.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Informative Features plot to: {plot_path}")
    return top_positive, top_negative


def run_domain_adaptation(docs, labels, categories):
    print("\n" + "=" * 70)
    print("Optional Task 3: Domain Sensitivity / Cross-Domain Adaptation")
    print("=" * 70)

    unique_domains = sorted(list(set(categories)))
    print(f"Available Domains ({len(unique_domains)}): {unique_domains}")

    domain_data = {d: {"docs": [], "labels": []} for d in unique_domains}
    for doc, label, cat in zip(docs, labels, categories):
        domain_data[cat]["docs"].append(doc)
        domain_data[cat]["labels"].append(label)

    domain_train = {}
    domain_test = {}
    for d in unique_domains:
        d_docs = domain_data[d]["docs"]
        d_labels = domain_data[d]["labels"]
        split = int(0.80 * len(d_docs))
        domain_train[d] = (d_docs[:split], d_labels[:split])
        domain_test[d] = (d_docs[split:], d_labels[split:])

    matrix = np.zeros((len(unique_domains), len(unique_domains)))
    for i, src in enumerate(unique_domains):
        src_docs, src_labels = domain_train[src]
        src_model = train_nb(src_docs, src_labels, alpha=1.0)
        for j, tgt in enumerate(unique_domains):
            tgt_docs, tgt_labels = domain_test[tgt]
            preds = classify_documents(tgt_docs, src_model)
            acc = accuracy(tgt_labels, preds)
            matrix[i, j] = acc

    print(f"\nCross-Domain Accuracy Matrix (%): (Rows = Train Source, Columns = Test Target)")
    header = f"{'Source':<10} | " + " | ".join([f"{d[:6]:>6}" for d in unique_domains])
    print(header)
    print("-" * len(header))
    for i, src in enumerate(unique_domains):
        row_str = f"{src:<10} | " + " | ".join([f"{matrix[i, j]*100:6.1f}" for j in range(len(unique_domains))])
        print(row_str)

    idx_cam = unique_domains.index("camera")
    idx_books = unique_domains.index("books")
    cam_in_domain = matrix[idx_cam, idx_cam] * 100
    cam_to_books = matrix[idx_cam, idx_books] * 100
    drop = cam_in_domain - cam_to_books
    print(f"\nPrompt Specific Question: Camera Review Model on Book Reviews:")
    print(f"  Camera In-Domain Accuracy:    {cam_in_domain:.2f}%")
    print(f"  Camera -> Books Test Accuracy: {cam_to_books:.2f}%")
    print(f"  Accuracy Drop:                {drop:.2f} percentage points")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix * 100, cmap="YlGnBu", vmin=65, vmax=90)
    ax.set_xticks(range(len(unique_domains)))
    ax.set_yticks(range(len(unique_domains)))
    ax.set_xticklabels([d.capitalize() for d in unique_domains])
    ax.set_yticklabels([d.capitalize() for d in unique_domains])
    ax.set_xlabel("Target Test Domain")
    ax.set_ylabel("Source Training Domain")
    ax.set_title("Optional Task 3: Cross-Domain Transfer Accuracy (%)")

    for i in range(len(unique_domains)):
        for j in range(len(unique_domains)):
            val = matrix[i, j] * 100
            color = "white" if val > 80 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight=weight)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Accuracy (%)", rotation=-90, va="bottom")
    plt.tight_layout()
    plot_path = MEDIA_DIR / "domain_adaptation.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Domain Adaptation plot to: {plot_path}")
    return matrix, unique_domains


def extract_improved_features(document, use_bigrams=False, use_negation=False):
    """
    Extracts features with optional bigrams and negation prefix tagging.
    Negation: tags words following 'not', 'n't', 'never', 'no', etc., with 'NOT_'.
    """
    negation_words = {"not", "n't", "no", "never", "hardly", "barely", "scarcely", "without"}
    tokens = []
    neg_active = False

    for w in document:
        if w in negation_words or w.endswith("n't"):
            neg_active = True
            tokens.append(w)
            continue
        if w in {".", ",", ";", "!", "?", "but", "however", "although"}:
            neg_active = False
            tokens.append(w)
            continue

        if use_negation and neg_active:
            tokens.append(f"NOT_{w}")
        else:
            tokens.append(w)

    features = list(tokens)
    if use_bigrams:
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i+1]}")

    return features


def run_feature_improvements(train_docs, train_labels, eval_docs, eval_labels):
    print("\n" + "=" * 70)
    print("Optional Task 4: Feature Improvement Experiments")
    print("=" * 70)

    experiments = [
        ("Baseline (Unigrams)", False, False),
        ("Negation Tagging (NOT_)", False, True),
        ("Unigrams + Bigrams", True, False),
        ("Unigrams + Bigrams + Negation", True, True)
    ]

    results = {}
    all_preds = {}
    for name, use_bi, use_neg in experiments:
        proc_train = [extract_improved_features(doc, use_bigrams=use_bi, use_negation=use_neg) for doc in train_docs]
        proc_eval = [extract_improved_features(doc, use_bigrams=use_bi, use_negation=use_neg) for doc in eval_docs]

        model = train_nb(proc_train, train_labels, alpha=1.0)
        preds = classify_documents(proc_eval, model)
        acc = accuracy(eval_labels, preds)
        results[name] = acc
        all_preds[name] = preds
        print(f"  {name:<32}: {acc * 100:.2f}% (Vocab: {model['vocab_size']:,} features)")

    base_correct = [p == t for p, t in zip(all_preds["Baseline (Unigrams)"], eval_labels)]
    bigram_correct = [p == t for p, t in zip(all_preds["Unigrams + Bigrams"], eval_labels)]
    b = sum(1 for bc, bg in zip(base_correct, bigram_correct) if bc and not bg)
    c = sum(1 for bc, bg in zip(base_correct, bigram_correct) if not bc and bg)
    chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    from scipy.stats import chi2
    p_value = float(chi2.sf(chi2_stat, df=1)) if (b + c) > 0 else 1.0
    print(f"\n  McNemar's Statistical Significance Test (Baseline vs Unigrams + Bigrams):")
    print(f"    Baseline correct, Bigrams wrong (b) = {b}")
    print(f"    Baseline wrong, Bigrams correct (c) = {c}")
    print(f"    chi2 = {chi2_stat:.4f}, p-value = {p_value:.3e}")
    if p_value < 0.05:
        print("    Result: Statistically significant improvement (p < 0.05).")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    bars = ax.barh(names, accs, color=["#4575b4", "#74add1", "#fdae61", "#d73027"], alpha=0.85)

    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("Optional Task 4: Impact of Feature Improvements on Classification Accuracy")
    ax.set_xlim(min(accs) - 2, max(accs) + 1.5)
    ax.grid(True, linestyle=":", alpha=0.4, axis="x")

    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.15, bar.get_y() + bar.get_height() / 2, f"{acc:.2f}%", va="center", fontweight="bold")

    plt.tight_layout()
    plot_path = MEDIA_DIR / "feature_improvements.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Feature Improvements plot to: {plot_path}")
    return results


def train_sklearn(docs, labels):
    """
    Trains a scikit-learn LinearSVC text classification pipeline matching the assignment PDF.
    Uses preprocessor=None, tokenizer=identity to process pre-tokenized documents.
    """
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest

    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None)
    sel = SelectKBest(k="all")
    clf = LinearSVC(dual=False, max_iter=2000, random_state=42)
    pipeline = make_pipeline(vec, sel, clf)
    pipeline.fit(docs, labels)
    return pipeline


def classify_documents_sklearn(docs, model):
    """
    Classifies documents using the fitted scikit-learn pipeline matching the assignment PDF.
    """
    return model.predict(docs)


def run_sklearn_benchmark(train_docs, train_labels, eval_docs, eval_labels, custom_acc):
    print("\n" + "=" * 70)
    print("Optional Task 5: Scikit-learn LinearSVC Benchmark")
    print("=" * 70)

    try:
        pipeline = train_sklearn(train_docs, train_labels)
        preds = classify_documents_sklearn(eval_docs, pipeline)
        svc_acc = accuracy(eval_labels, preds)

        print(f"  Custom Naive Bayes Accuracy:  {custom_acc * 100:.2f}%")
        print(f"  Scikit-Learn LinearSVC Accuracy: {svc_acc * 100:.2f}%")
        print(f"  Difference:                   {(svc_acc - custom_acc) * 100:+.2f} percentage points")
        return svc_acc
    except Exception as e:
        print(f"  Could not run sklearn benchmark: {e}")
        return None


def run_six_category_classifier(train_docs, train_cats, eval_docs, eval_cats):
    print("\n" + "=" * 70)
    print("Optional Task: Six-Category Topic Classifier")
    print("=" * 70)
    topic_model = train_nb(train_docs, train_cats, alpha=1.0)
    preds = classify_documents(eval_docs, topic_model)
    acc = accuracy(eval_cats, preds)
    print(f"  6-Category Topic Accuracy: {acc * 100:.2f}% (Chance baseline: 16.67%)")
    return acc


def main():
    print("=" * 70)
    print("SYDE 572 - Assignment 4: Text Classification with Naive Bayes")
    print("=" * 70)

    print(f"Loading review data from: {DATA_FILE}")
    all_docs, all_labels, all_categories = read_documents(DATA_FILE, return_categories=True)
    n_total = len(all_docs)
    print(f"Total documents loaded: {n_total:,}")
    print(f"Class distribution: {Counter(all_labels)}")
    print(f"Topic distribution: {Counter(all_categories)}")

    split_point = int(0.80 * n_total)
    train_docs, eval_docs = all_docs[:split_point], all_docs[split_point:]
    train_labels, eval_labels = all_labels[:split_point], all_labels[split_point:]
    train_cats, eval_cats = all_categories[:split_point], all_categories[split_point:]
    print(f"\nTrain set: {len(train_docs):,} documents (80%) | Distribution: {Counter(train_labels)}")
    print(f"Eval set:  {len(eval_docs):,} documents (20%) | Distribution: {Counter(eval_labels)}")

    print("\nTraining custom Naive Bayes classifier (alpha=1.0)...")
    model = train_nb(train_docs, train_labels, alpha=1.0)
    print(f"  Vocabulary size: {model['vocab_size']:,} unique words")
    print(f"  Total words in Positive class: {model['total_words']['pos']:,}")
    print(f"  Total words in Negative class: {model['total_words']['neg']:,}")
    print(f"  Prior log P(pos): {model['log_priors']['pos']:.4f}")
    print(f"  Prior log P(neg): {model['log_priors']['neg']:.4f}")

    run_sanity_checks(model)

    print("\n" + "=" * 70)
    print("Evaluating Classifier on 20% Evaluation Test Set")
    print("=" * 70)
    predictions = classify_documents(eval_docs, model)
    test_acc = accuracy(eval_labels, predictions)
    n_eval = len(eval_labels)
    n_correct = sum(1 for t, g in zip(eval_labels, predictions) if t == g)
    se = math.sqrt(test_acc * (1.0 - test_acc) / n_eval)
    ci_lower = (test_acc - 1.96 * se) * 100
    ci_upper = (test_acc + 1.96 * se) * 100
    print(f"Final Test Accuracy: {test_acc * 100:.2f}% ({n_correct} / {n_eval} correct)")
    print(f"95% Confidence Interval: [{ci_lower:.2f}%, {ci_upper:.2f}%]")

    false_positives, false_negatives = run_error_analysis(eval_docs, eval_labels, predictions, model)

    train_sizes, accuracies = run_learning_curve(train_docs, train_labels, eval_docs, eval_labels)
    top_pos, top_neg = print_top_nb(model, k=15, alpha=10.0, min_freq=5)
    domain_matrix, domains = run_domain_adaptation(all_docs, all_labels, all_categories)
    feature_results = run_feature_improvements(train_docs, train_labels, eval_docs, eval_labels)
    svc_acc = run_sklearn_benchmark(train_docs, train_labels, eval_docs, eval_labels, test_acc)
    six_cat_acc = run_six_category_classifier(train_docs, train_cats, eval_docs, eval_cats)

    print("\n" + "=" * 70)
    print("All Assignment 4 tasks and optional extensions completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
