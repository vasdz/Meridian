# RetailEdge (Meridian) - Enterprise ML Platform

## 🎯 Senior+ Level ML Platform for Retail Analytics

Production-grade machine learning platform designed for hyperscale retail operations.

### ✅ Core Capabilities

| Module | Description | Status |
|--------|-------------|--------|
| **Uplift Modeling** | CATE prediction for personalized marketing | ✅ Production |
| **Demand Forecasting** | Multi-horizon probabilistic forecasting | ✅ Production |
| **Dynamic Pricing** | Elasticity-based price optimization | ✅ Production |
| **A/B Testing** | Statistical experiment framework | ✅ Production |
| **Causal Discovery** | Automatic causal graph construction | ✅ Production |
| **Attribution** | Multi-touch Shapley/Markov attribution | ✅ Production |
| **ML Monitoring** | Drift detection, SLA tracking, alerting | ✅ Production |

---

## 🚀 Quick Start

```bash
# Install dependencies
poetry install

# Run migrations
poetry run alembic upgrade head

# Start server
poetry run uvicorn meridian.main:app --reload --port 8000

# Run tests
poetry run pytest tests/ -v
```

---

## 📊 API Endpoints

### Uplift Modeling (`/v1/uplift/`)
- `POST /predict` - Single prediction with CATE
- `POST /predict/batch` - Batch predictions
- `GET /models` - List available models

### Demand Forecasting (`/v1/forecast/`)
- `POST /demand` - Single series forecast
- `POST /batch` - Multi-series batch forecast
- `GET /models` - List forecasting models

### Dynamic Pricing (`/v1/pricing/`)
- `POST /optimize` - Batch price optimization
- `POST /elasticity` - Estimate price elasticity
- `POST /sensitivity` - Price sensitivity analysis
- `GET /models` - List pricing models

### A/B Experiments (`/v1/experiments/`)
- `GET /` - List experiments
- `POST /` - Create experiment
- `GET /{id}` - Get experiment details
- `POST /{id}/start` - Start experiment
- `POST /{id}/analyze` - Analyze results

### ML Monitoring (`/v1/monitoring/`)
- `GET /health/{model_id}` - Model health status
- `GET /health` - All models health
- `GET /drift/{model_id}` - Drift detection
- `GET /alerts` - Get alerts
- `GET /sla/{model_id}` - SLA compliance report
- `GET /metrics/{model_id}` - Current metrics

---

## 🏗️ Architecture

```
src/meridian/
├── api/                      # FastAPI layer
│   ├── routers/v1/           # API endpoints
│   ├── schemas/              # Pydantic models
│   ├── dependencies/         # DI (auth, db, rate limiting)
│   └── middleware/           # Security, correlation, timing
│
├── domain/                   # Business logic
│   ├── models/               # Domain entities
│   ├── repositories/         # Repository interfaces
│   └── services/             # Domain services
│       ├── uplift_calculator.py
│       ├── ab_testing.py     # Power analysis, MDE, corrections
│       ├── causal_discovery.py  # PC algorithm
│       ├── attribution.py    # Shapley, Markov attribution
│       └── experiment_design.py
│
├── application/              # Use cases
│   ├── use_cases/            # Application services
│   └── interfaces/           # Port interfaces
│
├── infrastructure/           # External integrations
│   ├── ml/
│   │   ├── uplift/           # T/S/X-Learner, Causal Forest, metrics
│   │   ├── forecasting/      # Ensemble, LightGBM, Holt-Winters
│   │   ├── pricing/          # Elasticity, optimization
│   │   ├── pipeline/         # ML pipelines
│   │   └── feature_store/    # RFM, behavioral features
│   ├── monitoring/           # Drift, SLA, alerting
│   ├── database/             # SQLAlchemy async
│   ├── cache/                # Redis
│   └── security/             # Encryption, vault
│
└── workers/                  # Celery async tasks
```

---

## 🧪 ML Models

### Uplift (CATE Estimation)
- **T-Learner**: Two-model approach
- **S-Learner**: Single model with treatment feature  
- **X-Learner**: Cross-model with propensity weighting
- **Causal Forest**: Tree-based heterogeneous effects

### Demand Forecasting
- **Ensemble**: LightGBM + Holt-Winters with auto-selection
- **LightGBM Quantile**: ML-based with uncertainty
- **Holt-Winters**: Classical exponential smoothing
- **Conformal Prediction**: Distribution-free intervals

### Pricing
- **Log-Log Elasticity**: Constant elasticity model
- **Bayesian Elasticity**: Posterior distribution via bootstrap
- **Profit Optimizer**: Constrained optimization

---

## 📈 Monitoring Features

### Drift Detection
- Population Stability Index (PSI)
- Kolmogorov-Smirnov test
- Wasserstein distance
- Categorical chi-squared

### SLA Tracking
- Latency SLA (p95, p99)
- Error rate SLA
- Accuracy SLA
- Availability monitoring

### Alerting
- Severity levels (info, warning, critical)
- Deduplication
- Acknowledgment tracking
- Notification handlers

---

## 🔬 A/B Testing Framework

### Power Analysis
- Sample size calculation (binary/continuous)
- Minimum Detectable Effect (MDE)
- Power calculation
- Duration estimation

### Statistical Tests
- Z-test for proportions
- Welch's t-test for continuous
- Chi-squared test

### Multiple Testing Corrections
- Bonferroni
- Holm-Bonferroni (step-down)
- Benjamini-Hochberg (FDR)

---

## 🔗 Causal Inference

### Causal Discovery
- PC Algorithm (constraint-based)
- Conditional independence testing
- V-structure detection
- Meek's rules for orientation

### Attribution
- **Shapley Values**: Game-theoretic fair attribution
- **Markov Chains**: Transition probability-based
- **Last/First Touch**: Baseline comparisons
- **Linear**: Equal credit distribution

---

## 🛡️ Security Features

- JWT authentication
- API key authentication
- Rate limiting (Redis)
- Field-level encryption (AES-256-GCM)
- SQL injection protection
- Security headers (HSTS, CSP, X-Frame-Options)
- Audit logging

---

## 📦 Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI, Pydantic v2 |
| Database | PostgreSQL, SQLAlchemy 2.0 async |
| Cache | Redis |
| ML | scikit-learn, LightGBM, scipy |
| Tasks | Celery |
| Monitoring | structlog, Prometheus-ready |
| Testing | pytest, 51+ unit tests |

---

## 🧪 Testing

```bash
# All tests
poetry run pytest tests/ -v

# Unit tests only
poetry run pytest tests/unit -v

# With coverage
poetry run pytest tests/ --cov=src/meridian --cov-report=html
```

---

## 📄 License

Proprietary - For portfolio demonstration purposes.

---

## 👨‍💻 Author

Built as a Senior+ level portfolio project demonstrating:
- Clean Architecture (DDD, Hexagonal)
- Production-grade ML systems
- Enterprise observability
- Statistical rigor in experimentation
- Causal inference expertise

