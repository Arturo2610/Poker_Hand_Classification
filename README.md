# Poker Hand Classification: Feature Engineering Over Model Complexity

A machine learning project demonstrating that thoughtful feature engineering with domain knowledge can outperform complex models on raw data.

---

## Key Result

**Perfect classification** (100% accuracy, 0 errors) on 1,000,000 poker hands using Logistic Regression with 6 domain-engineered features, outperforming a Random Forest baseline by 7.7× on balanced accuracy while maintaining interpretability.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [The Challenge](#the-challenge)
- [Approach](#approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [Project Highlights](#project-highlights)
- [Files Description](#files-description)
- [Future Enhancements](#future-enhancements)
- [Lessons Learned](#lessons-learned)
- [Author](#author)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)

---

## Problem Statement

Given 5 playing cards (each with suit and rank), classify the poker hand into one of 10 categories:

| Class | Hand Type | Example |
|-------|-----------|---------|
| 9 | Royal Flush | 10♠ J♠ Q♠ K♠ A♠ |
| 8 | Straight Flush | 5♣ 6♣ 7♣ 8♣ 9♣ |
| 7 | Four of a Kind | A♥ A♠ A♦ A♣ K♠ |
| 6 | Full House | K♥ K♠ K♦ 10♥ 10♠ |
| 5 | Flush | 2♦ 5♦ 8♦ J♦ K♦ |
| 4 | Straight | 5♥ 6♠ 7♦ 8♣ 9♥ |
| 3 | Three of a Kind | Q♥ Q♠ Q♦ 7♣ 3♥ |
| 2 | Two Pairs | J♥ J♠ 8♦ 8♣ 3♥ |
| 1 | One Pair | 9♥ 9♠ 5♦ 7♣ K♥ |
| 0 | Nothing | 2♥ 5♠ 8♦ J♣ K♥ |

**Dataset:**
- Training: 25,010 hands
- Testing: 1,000,000 hands
- Source: [Kaggle - Poker Game Dataset](https://www.kaggle.com/datasets/hosseinah1/poker-game-dataset)

---

## The Challenge

### Extreme Class Imbalance

The dataset reflects real poker probabilities, creating severe class imbalance:

| Hand Type | Frequency | % of Data |
|-----------|-----------|-----------|
| Nothing | 501,209 | 50.12% |
| One Pair | 422,498 | 42.25% |
| Two Pairs | 47,622 | 4.76% |
| Three of a Kind | 21,121 | 2.11% |
| Straight | 3,885 | 0.39% |
| Flush | 1,996 | 0.20% |
| Full House | 1,424 | 0.14% |
| Four of a Kind | 230 | 0.02% |
| Straight Flush | 12 | 0.001% |
| Royal Flush | 3 | 0.0003% |

The rarest class (Royal Flush) is **167,000× less frequent** than the most common class.

**Why this matters:**
- A naive model predicting only "Nothing" achieves 50% accuracy but is useless
- Standard accuracy metrics are misleading
- Rare classes are critical to identify (they're the most valuable poker hands!)

---

## Approach

### Baseline: Complex Model, Raw Features

**Model:** Random Forest (100 trees, max depth 20)  
**Features:** 10 raw features (suit1, rank1, suit2, rank2, ...)

**Hypothesis:** A complex ensemble model should learn poker patterns from raw data.

### Engineered: Simple Model, Smart Features

**Model:** Logistic Regression (linear classifier)  
**Features:** 6 domain-engineered features:

1. **is_flush** - Do all cards share the same suit?
2. **is_straight** - Do ranks form a sequence? (handles Ace-low and Ace-high)
3. **is_royal_straight** - Is this specifically 10-J-Q-K-A?
4. **unique_ranks** - How many distinct ranks?
5. **max_rank_count** - Most frequent rank occurrence (1=nothing, 2=pair, 3=trips, 4=quads)
6. **second_max_rank_count** - Second most frequent (distinguishes Full House from Three of a Kind)

**Hypothesis:** Encoding poker domain knowledge as features enables simple models to excel.

---

## Results

### Model Comparison

| Model | Features | Algorithm | Balanced Acc | F1 (Macro) | F1 (Weighted) | Training Time |
|-------|----------|-----------|--------------|------------|---------------|---------------|
| **Baseline** | 10 raw | Random Forest | 13.05% | 12.80% | 56.73% | 0.31s |
| **Engineered** | 6 domain | Logistic Regression | **100.00%** | **100.00%** | **100.00%** | 0.30s |

### Performance Breakdown

**Baseline Model:**
- Only learns the 2 dominant classes (Nothing, One Pair)
- **Zero recall** on 7 out of 10 classes
- Cannot identify rare but valuable hands (Royal Flush, Straight Flush, Four of a Kind)
- Essentially useless despite 60% raw accuracy

**Engineered Model:**
- Perfect precision and recall on **all 10 classes**
- Correctly classifies all 3 Royal Flushes in 1M test samples
- 99.79% mean prediction confidence
- Zero classification errors

### Test Set Performance (1,000,000 samples)

                     precision    recall  f1-score   support

    Nothing             1.0000    1.0000    1.0000    501209
    One pair            1.0000    1.0000    1.0000    422498
    Two pairs           1.0000    1.0000    1.0000     47622
    Three of a kind     1.0000    1.0000    1.0000     21121
    Straight            1.0000    1.0000    1.0000      3885
    Flush               1.0000    1.0000    1.0000      1996
    Full house          1.0000    1.0000    1.0000      1424
    Four of a kind      1.0000    1.0000    1.0000       230
    Straight flush      1.0000    1.0000    1.0000        12
    Royal flush         1.0000    1.0000    1.0000         3



**Overall Accuracy: 100.0000%**  
**Total Errors: 0 out of 1,000,000**

---

## Project Structure
poker-hand-classification/
├── README.md # Project documentation\
├── requirements.txt # Python dependencies\
├── app.py # Streamlit web application\
├── save_models.py # Model training script\
│
├── data/
│ ├── poker-hand-training.csv # Training data (25K samples)\
│ ├── poker-hand-testing.csv # Test data (1M samples)\
│ ├── train_engineered.csv # Engineered features (training)\
│ └── test_engineered.csv # Engineered features (test)\
│
├── models/
│ ├── best_model.pkl # Trained Logistic Regression\
│ └── scaler.pkl # Feature scaler\
│
├── src/
│ ├── feature_engineering.py # Core feature engineering logic\
│ ├── evaluation.py # Model evaluation metrics\
│ └── utils.py # Data loading and analysis\
│
└── notebooks/
├── 01_exploratory_analysis.ipynb # EDA and feature creation\
└── 02_modeling.ipynb # Model training and evaluation


---

## Installation

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/poker-hand-classification.git
cd poker-hand-classification

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- `numpy==1.24.3`
- `pandas==2.0.3`
- `scikit-learn==1.3.0`
- `matplotlib==3.7.2`
- `seaborn==0.12.2`
- `streamlit==1.25.0`
- `joblib==1.3.1`
- `jupyter`

---

## Usage

### 1. Interactive Web Application

Launch the Streamlit app to classify poker hands interactively:

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

Features:

- Select 5 cards from dropdown menus
- Instant hand classification
- Model confidence display
- Feature analysis breakdown
- Poker hand rankings reference


### 2. Exploratory Analysis

Run the Jupyter notebooks to see the full analysis:

```bash
jupyter notebook
```
Navigate to:
- `notebooks/01_exploratory_analysis.ipynb` - Data exploration and feature engineering
- `notebooks/02_modeling.ipynb` - Model training and comparison

### 3. Train Models from Scratch

```bash
python save_models.py
```

This will:

- Load raw data
- Generate engineered features
- Train Logistic Regression model
- Save model and scaler to models/

---

## Key Insights

### 1. Feature Engineering > Model Complexity

The engineered approach achieves **7.7× better balanced accuracy** with:
- **Fewer features** (6 vs 10)
- **Simpler algorithm** (linear vs ensemble)
- **Same training time** (0.30s vs 0.31s)
- **Full interpretability** (can explain every prediction)

### 2. Domain Knowledge is Critical

The 6 engineered features encode poker expertise:
- A poker player looks for flushes, straights, and matching ranks
- Raw positional data (suit of card 3, rank of card 5) is uninformative
- Translating domain knowledge into features makes the problem linearly separable

### 3. Metrics Matter for Imbalanced Data

Standard accuracy is misleading:
- Baseline achieves 60% accuracy but fails on 70% of classes
- Balanced accuracy (13% vs 100%) reveals the true performance gap
- Per-class metrics (precision, recall, F1) are essential

### 4. Edge Cases Validate Robustness

The model correctly handles tricky cases:
- **Wheel straight** (A-2-3-4-5): Ace as low card
- **Royal flush vs straight flush**: Requires `is_royal_straight` feature
- **Full house vs three of a kind**: Needs `second_max_rank_count`

### 5. Perfect Performance is Achievable

Poker hand classification is deterministic - every hand has exactly one correct label. With proper features, 100% accuracy is not just possible but expected. This validates the entire methodology.

---

## Tech Stack

**Languages & Libraries:**
- Python 3.10
- NumPy, Pandas - data manipulation
- scikit-learn - machine learning models and metrics
- Matplotlib, Seaborn - visualization

**Machine Learning:**
- Logistic Regression (multinomial classification)
- Random Forest (baseline comparison)
- StandardScaler (feature normalization)

**Deployment:**
- Streamlit - interactive web application
- Jupyter - exploratory analysis and documentation

**Development:**
- Git - version control
- Virtual environment - dependency isolation

---

## Project Highlights

This project demonstrates:

- **Full ML pipeline:** EDA → Feature Engineering → Modeling → Evaluation → Deployment
- **Problem diagnosis:** Identifying class imbalance as the core challenge
- **Metric selection:** Using balanced accuracy and F1 scores instead of misleading accuracy
- **Baseline comparison:** Systematic evaluation of competing approaches
- **Domain expertise:** Translating poker knowledge into ML features
- **Production thinking:** Deployed web app with validation and error handling
- **Code organization:** Modular structure with src/, notebooks/, and app files
- **Documentation:** Clear README, code comments, and analysis notebooks

---

## Files Description

**Core Code:**
- `src/feature_engineering.py` - PokerFeatureEngine class with domain logic
- `src/evaluation.py` - Custom evaluation functions for imbalanced classification
- `src/utils.py` - Data loading and visualization utilities

**Application:**
- `app.py` - Streamlit web interface for interactive predictions

**Notebooks:**
- `01_exploratory_analysis.ipynb` - Class distribution analysis, feature correlation
- `02_modeling.ipynb` - Baseline vs engineered comparison, final verification

**Scripts:**
- `save_models.py` - Standalone script to train and save models

---

## Future Enhancements

Potential improvements for learning or extension:

- Add ensemble of engineered-feature models (Logistic Regression + Gradient Boosting)
- Implement SHAP values to explain individual predictions
- Create API endpoint (FastAPI) for production deployment
- Add unit tests for feature engineering logic
- Extend to Texas Hold'em (7-card evaluation)
- Build confidence calibration analysis

---

## Lessons Learned

**What worked:**
- Investing time in understanding the problem domain (poker rules)
- Creating features that mirror human reasoning
- Using appropriate metrics for imbalanced data
- Systematic baseline comparison

**What didn't work:**
- Complex models on raw features (Random Forest baseline)
- Standard accuracy as primary metric
- Initial feature set without `is_royal_straight` (couldn't distinguish Royal from regular Straight Flush)

**Key takeaway:**
In well-defined problems with clear domain logic, thoughtful feature engineering enables simple, interpretable models to achieve perfect performance. This is often preferable to black-box models that are harder to debug, deploy, and explain.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Dataset: [Poker Game Dataset](https://www.kaggle.com/datasets/hosseinah1/poker-game-dataset) from Kaggle
- Inspiration: Demonstrating that domain knowledge beats algorithmic complexity
- Purpose: Portfolio project showcasing end-to-end ML skills

---

## Contributing

This is a portfolio project, but suggestions and feedback are welcome! Feel free to:
- Open an issue for bugs or questions
- Fork the repo and experiment
- Reach out with comments or ideas

---

*Built with passion for clean code, thoughtful ML, and practical problem-solving.*
# Poker_Hand_Classification
