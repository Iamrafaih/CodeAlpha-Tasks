# 💎 Diamond Price Analysis — Data Visualization

**Data Analytics Internship · Task 3**

A Python project that turns a dataset of **53,940 diamonds** into clear charts and an interactive dashboard, to answer one question: **what really drives a diamond's price?**

---

## 📌 What This Project Does

- Loads and cleans the diamonds dataset
- Creates **8 charts**, each with a headline that states the main finding
- Prints a short **Key Findings** summary in the terminal
- Includes an **interactive dashboard** (`dashboard.html`) that updates live when you pick filters

---

## 🔍 Key Findings

1. **Size is the biggest price driver.** Carat and price have a correlation of **0.92**.
2. **A trap in the data:** Ideal-cut diamonds look cheaper on average ($3,457) than Fair-cut ones ($4,358) — only because Ideal stones are smaller (0.70 vs 1.05 carat).
3. **Quality does pay once size is equal.** Per carat, Ideal cut earns $3,919 vs $3,766 for Fair.
4. **Clarity matters.** Flawless (IF) diamonds earn about **1.5x more per carat** than the lowest grade (I1).
5. **Depth and table barely matter.** Their correlation with price is close to zero.

---

## 📊 Charts Produced

| # | Chart | What it shows |
|---|-------|---------------|
| 1 | Price distribution | Most diamonds are cheap; a few are very expensive |
| 2 | Carat vs price | Bigger stones cost much more |
| 3 | Quality paradox | Why a better cut can look cheaper |
| 4 | Price per carat by quality | Cut, colour and clarity reward the buyer |
| 5 | Correlation heatmap | Which features move with price |
| 6 | Market composition | Which cut/clarity combinations are most common |
| 7 | Price bands | Price spread inside each size group |
| 8 | Executive dashboard | All the key insights on one page |

All charts are saved in the `charts/` folder.

---

## 🖥️ Interactive Dashboard

Open **`dashboard.html`** in any web browser — no installation needed.
Filter by **cut, colour, clarity, carat and price**, and the charts update instantly.

---

## 📁 Project Structure

```
CodeAlpha_DataVisualization/
├── data/
│   └── diamonds.csv              # Dataset (53,940 rows)
├── charts/                       # 8 generated chart images
├── diamond_visualization.py      # Main script
├── dashboard.html                # Interactive dashboard
├── template.html                 # Dashboard template
├── requirements.txt              # Python libraries needed
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
python diamond_visualization.py
```

**3. See the results:**
- Charts appear in the `charts/` folder
- Key findings are printed in the terminal

> Run the command from inside the project folder so the script can find `data/diamonds.csv`.

---

## 🧹 Data Cleaning

- Removed rows with impossible **zero** dimensions (x, y or z)
- Put `cut`, `color` and `clarity` in their proper quality order
- Added a new column, **`price_per_carat`**, to compare stones of different sizes fairly

---

## 📋 About the Dataset

| Column | Meaning |
|--------|---------|
| `carat` | Weight of the diamond |
| `cut` | Cut quality (Fair → Ideal) |
| `color` | Colour grade (J worst → D best) |
| `clarity` | Clarity grade (I1 worst → IF best) |
| `depth` | Total depth percentage |
| `table` | Width of the top relative to the widest point |
| `price` | Price in US dollars |
| `x`, `y`, `z` | Length, width and depth in mm |

---

## 🛠️ Tools Used

- **Python**
- **Pandas** and **NumPy** — data handling
- **Matplotlib** and **Seaborn** — charts
- **Chart.js** — interactive dashboard

---

