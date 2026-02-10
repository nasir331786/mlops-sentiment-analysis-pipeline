# 🚀 MLOps Sentiment Analysis Pipeline

## 📊 End-to-End Production-Ready ML System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-red)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-orange)](https://optuna.org/)
[![Prefect](https://img.shields.io/badge/Prefect-Workflow%20Automation-purple)](https://www.prefect.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project demonstrates a **complete MLOps pipeline** for sentiment analysis on Flipkart product reviews. It transforms a basic ML notebook into a **production-ready system** with:

✅ **Experiment Tracking** with MLflow  
✅ **Hyperparameter Optimization** with Optuna  
✅ **Workflow Automation** with Prefect  
✅ **Model Registry & Versioning**  
✅ **Reproducible ML Pipelines**

---

## 🏗️ Architecture

```
📦 MLOps Pipeline
├── 📊 Data Processing
│   ├── NLP Text Cleaning
│   ├── Sentiment Labeling (Binary Classification)
│   └── TF-IDF Vectorization
├── 🔬 Experimentation
│   ├── Optuna: Automated Hyperparameter Search
│   └── MLflow: Experiment Tracking & Logging
├── 🤖 Model Training
│   ├── Logistic Regression
│   ├── Naive Bayes
│   └── Linear SVM
├── 📈 Evaluation
│   ├── F1-Score (Macro)
│   └── Cross-Validation
├── 🗃️ Model Registry
│   └── MLflow Model Versioning
└── ⚙️ Automation
    └── Prefect: Scheduled Workflows
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|--------|
| **MLflow** | Experiment tracking, model registry, versioning |
| **Optuna** | Bayesian hyperparameter optimization |
| **Prefect** | Workflow orchestration & scheduling |
| **scikit-learn** | ML models & preprocessing |
| **NLTK** | NLP text preprocessing |
| **Pandas** | Data manipulation |
| **Python 3.8+** | Core programming language |

---

## 📂 Project Structure

```
mlops-sentiment-analysis-pipeline/
├── data.csv                              # Flipkart reviews dataset
├── sentiment_mlflow_optuna.py            # Main training script with MLflow & Optuna
├── ml_orchestration_flipkart.py          # Prefect workflow automation
├── MLOps-Pipeline-Guide.pdf              # Complete project documentation
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
└── mlruns/                              # MLflow experiment tracking directory
```

---

## 🚦 Getting Started

### Prerequisites

```bash
python >= 3.8
pip >= 20.0
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nasir331786/mlops-sentiment-analysis-pipeline.git
cd mlops-sentiment-analysis-pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 🎬 Usage

### 1️⃣ Run Experiment Tracking with MLflow & Optuna

```bash
python sentiment_mlflow_optuna.py
```

**What it does:**
- Loads Flipkart reviews dataset
- Preprocesses text using NLP pipeline
- Optimizes 3 models (Logistic Regression, Naive Bayes, Linear SVM) using Optuna
- Logs experiments to MLflow
- Registers best models in MLflow Model Registry

### 2️⃣ View MLflow Dashboard

```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
```

Open: `http://localhost:5000`

### 3️⃣ Run Automated Workflow with Prefect

```bash
python ml_orchestration_flipkart.py
```

**Features:**
- Modular task-based pipeline
- Automated retraining (scheduled every 5 minutes)
- Real-time monitoring via Prefect dashboard

### 4️⃣ Access Prefect Dashboard

Open: Prefect Cloud or local Prefect server

---

## 📊 Dataset

**Flipkart Product Reviews**
- **Source:** Flipkart customer reviews
- **Features:** Review text, Ratings (1-5)
- **Sentiment Mapping:**
  - Ratings 1-2: **Negative (0)**
  - Ratings 3-5: **Positive (1)**

---

## 🔬 Experiment Results

| Model | CV F1-Score | Train F1 | Test F1 |
|-------|-------------|----------|----------|
| **Logistic Regression** | 0.7847 | 0.8191 | **0.7933** |
| **Naive Bayes** | 0.7803 | 0.8493 | 0.7763 |
| **Linear SVM** | 0.7845 | 0.8255 | **0.8004** |

**Key Insights:**
- All models show **< 5% train-test gap** (no overfitting)
- **Logistic Regression** offers best balance of performance & interpretability
- **Linear SVM** achieves highest test F1-score

---

## 🎯 Key Features

### 1. **Robust NLP Pipeline**
- Lowercasing
- Emoji removal
- URL & HTML tag removal
- Punctuation & number removal
- Stopword removal (preserving negations: "not", "no", "nor")
- Stemming (Porter Stemmer)

### 2. **Intelligent Hyperparameter Optimization**
- **Optuna** conducts Bayesian optimization
- Optimizes TF-IDF + Model hyperparameters jointly
- 20 trials per model
- Stratified K-Fold Cross-Validation (k=5)

### 3. **Comprehensive Experiment Tracking**
- MLflow logs:
  - Parameters (TF-IDF config, model hyperparameters)
  - Metrics (CV F1, Train F1, Test F1, training time)
  - Artifacts (trained pipelines, conda environments)
  - Model registry with versioning

### 4. **Production-Ready Automation**
- Prefect tasks for modular pipeline
- Scheduled execution (cron: every 5 minutes)
- Real-time monitoring & alerting

---

## 📈 MLflow Features

### Parallel Coordinates Analysis
Visualize hyperparameter impact on model performance:

- **Logistic Regression:** Best run `masked-chimp-785` (CV F1: 0.784)
- **Linear SVM:** Best run `blushing-sloth-276` (Test F1: 0.800)
- **Naive Bayes:** Best run `carefree-mouse-245` (CV F1: 0.780)

### Model Registry
All models registered under: **`FlipkartSentimentModel`**
- Version control
- Stage transitions (None → Staging → Production)
- Rollback capability

---

## 🔄 Workflow Automation with Prefect

### Pipeline Stages

```python
@flow(name="Flipkart Sentiment Analysis Flow")
def sentiment_workflow():
    df = load_data("data.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_vec, X_test_vec = vectorize_text(X_train, X_test)
    model = train_model(X_train_vec, y_train)
    train_f1, test_f1 = evaluate_model(model, X_train_vec, y_train, X_test_vec, y_test)
```

### Scheduling
```python
sentiment_workflow.serve(
    name="flipkart-sentiment-deployment",
    cron="*/5 * * * *"  # Every 5 minutes
)
```

---

## 📚 Documentation

For detailed documentation, see:
- **[MLOps-Pipeline-Guide.pdf](./MLOps-Pipeline-Guide.pdf)** - Complete technical guide with screenshots

---

## 🎓 Key Learnings

1. **F1-Score > Accuracy** for imbalanced sentiment tasks
2. **Optuna** enables data-driven model selection
3. **MLflow** provides reproducibility & collaboration
4. **Prefect** automates ML workflows reliably
5. **End-to-end pipelines** bridge experimentation & production

---

## 🚀 Future Enhancements

- [ ] Flask REST API for real-time predictions
- [ ] Data drift monitoring
- [ ] Docker containerization
- [ ] A/B testing framework
- [ ] Deep learning models (BERT, RoBERTa)

---

## 👨‍💻 Author

**Nasir Husain Tamanne**  
- 🔗 GitHub: [@nasir331786](https://github.com/nasir331786)  
- 💼 LinkedIn: [nasir-husain-tamanne](https://www.linkedin.com/in/nasir-husain-tamanne-9b9981377/)  
- 📧 Email: nasirhusain1137@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MLflow** team for excellent experiment tracking
- **Optuna** for powerful optimization framework
- **Prefect** for intuitive workflow orchestration
- **scikit-learn** for robust ML tools

---

## 📞 Contact

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Connect on LinkedIn
- Email: nasirhusain1137@gmail.com

---

**⭐ If you find this project helpful, please give it a star!**

---

*Last Updated: February 2026*
