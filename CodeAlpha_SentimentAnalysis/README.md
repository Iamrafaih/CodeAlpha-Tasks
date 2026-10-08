# 💬 Twitter Sentiment Analysis


A Python project that reads 10,000 tweets and decides whether each one is **positive** or **negative**. Two methods are compared, and the project also shows how an impressive-looking accuracy score can be misleading.

---

## 📌 What This Project Does

- Cleans the tweet text (removes links, @mentions and emoticons)
- Tests **two approaches** to sentiment analysis:
  1. **VADER**: a ready-made tool that needs no training
  2. **TF-IDF + Logistic Regression**: a model trained on the tweets
- Creates **7 charts** that explain the results
- Includes an **interactive dashboard** where you can type your own sentence and see it scored live

---

## 📊 Results

| Approach | Accuracy |
|----------|----------|
| VADER (no training) | 62.3% |
| **TF-IDF + Logistic Regression** | **76.6%** |
| Same model, but shown the emoticons | 99.8% ⚠️ |

### ⚠️ Why is 99.8% a warning sign?

The tweets were labelled using their emoticons: `:)` meant positive and `:(` meant negative. When the model can see the emoticon, it simply reads the answer instead of understanding the words. This is called **data leakage**.

That is why the real, honest result is **76.6%**, with the emoticons removed.

---

## 🔍 Key Findings

1. **Emoticons must be removed** before training, or the accuracy score cannot be trusted.
2. **The honest accuracy is about 77%.** The labels come from emoticons, not from a person reading the text, so this is close to the limit for this data.
3. **The model learned real sentiment words.** Gratitude and greetings push a tweet positive. Loss and apology push it negative.
4. **VADER is a good quick start.** It needs no training, though it is less accurate on this data.
5. **Confidence is useful.** When the model is very confident, it is right far more often. Low-confidence tweets could be sent to a human to check.

---

## 🖥️ Interactive Dashboard

Open **`dashboard.html`** in any web browser. No installation or internet connection is needed.

- ✍️ Type your own text and see both models score it instantly
- 🔎 See which words made the prediction positive or negative
- 📋 Browse, search and filter all 10,000 tweets
- 📈 View accuracy charts and confusion matrices

---

## 📁 Project Structure

```
CodeAlpha_SentimentAnalysis/
├── data/
│   └── tweets.csv              # 10,000 labelled tweets
├── charts/                     # 7 generated chart images
├── web/                        # Model files used by the dashboard
├── sentiment_analysis.py       # Main script
├── dashboard.html              # Interactive dashboard
├── template.html               # Dashboard template
├── requirements.txt            # Python libraries needed
└── README.md
```

---

## ⚙️ How to Run

**1. Install Python 3.9 or newer**, then install the libraries:

```bash
pip install -r requirements.txt
```

**2. Run the script:**

```bash
python sentiment_analysis.py
```

**3. See the results:**
- Charts are saved in the `charts/` folder
- Model files for the dashboard are saved in the `web/` folder
- Key findings are printed in the terminal

> Run the command from inside the project folder so the script can find `data/tweets.csv`. NLTK data downloads automatically on the first run.

---

## 📋 About the Dataset

| | |
|---|---|
| Total tweets | 10,000 |
| Positive | 5,000 |
| Negative | 5,000 |
| Source | NLTK `twitter_samples` corpus |
| Columns | `text`, `label` |

---

## 🛠️ Tools Used

- **Python**
- **NLTK (VADER)**: rule-based sentiment scoring
- **scikit-learn**: TF-IDF and Logistic Regression
- **Pandas** and **NumPy**: data handling
- **Matplotlib** and **Seaborn**: charts
- **Chart.js**: interactive dashboard

---

