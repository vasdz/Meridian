# 🎯 Meridian

<div align="center">

[![CI](https://github.com/vasdz/Meridian/actions/workflows/ci.yml/badge.svg)](https://github.com/vasdz/Meridian/actions/workflows/ci.yml)
[![Security](https://github.com/vasdz/Meridian/actions/workflows/security.yml/badge.svg)](https://github.com/vasdz/Meridian/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/vasdz/Meridian/branch/main/graph/badge.svg)](https://codecov.io/gh/vasdz/Meridian)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**🇬🇧 English** | [🇷🇺 Русский](docs/README_RU.md)

*Enterprise-grade Marketing Analytics & Causal Inference Platform*

[Getting Started](#-quick-start) •
[Features](#-features) •
[API Docs](#-api-documentation) •
[Contributing](CONTRIBUTING.md) •
[Security](SECURITY.md)

</div>

---

## 📖 Overview

Meridian is a **production-ready ML platform** for retail analytics, providing:

- 🎯 **Uplift Modeling** — Predict individual treatment effects (CATE)
- 📈 **Demand Forecasting** — Multi-horizon probabilistic predictions
- 💰 **Price Optimization** — Maximize profit with elasticity models
- 🧪 **A/B Testing** — Design and analyze experiments

Built with **Clean Architecture** principles for enterprise scalability.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Gateway                                     │
│                           FastAPI + Uvicorn                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │    Uplift    │  │  Forecasting │  │    Pricing   │  │  Experiments │    │
│  │  /v1/uplift  │  │  /v1/forecast│  │  /v1/pricing │  │   /v1/exp    │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         └──────────────────┴─────────────────┴─────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Domain Services                                   │
│  ┌────────────────────┐  ┌─────────────────────┐  ┌───────────────────┐    │
│  │  UpliftCalculator  │  │  ExperimentDesign   │  │ PricingOptimizer  │    │
│  │  (Causal Forest)   │  │  (Power Analysis)   │  │  (Elasticity)     │    │
│  └────────────────────┘  └─────────────────────┘  └───────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Infrastructure                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ PostgreSQL  │  │    Redis    │  │   MLflow    │  │    Kafka    │        │
│  │    (DB)     │  │   (Cache)   │  │ (Tracking)  │  │  (Events)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Purpose | Key Components |
|-------|---------|----------------|
| **API** | HTTP interface, validation | Routers, Schemas, Middleware |
| **Application** | Use cases, orchestration | UseCases, DTOs |
| **Domain** | Business logic | Models, Services, Repositories (interfaces) |
| **Infrastructure** | External systems | DB, ML, Cache, Messaging |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| Poetry | 1.8+ |
| Docker | 24+ |
| Docker Compose | 2.20+ |

### Installation

```bash
# Clone repository
git clone https://github.com/vasdz/Meridian.git
cd Meridian

# Install dependencies
poetry install

# Copy environment file
cp .env.example .env

# Start infrastructure services
make docker

# Run database migrations
make migrate

# Seed sample data
make seed

# Start development server
make dev
```

Access the application:
- 🌐 **API**: http://localhost:8000
- 📚 **Swagger UI**: http://localhost:8000/docs
- 📖 **ReDoc**: http://localhost:8000/redoc
- 📊 **Dashboard**: http://localhost:8501 (Streamlit)

---

## 🧠 Features

### Uplift Modeling

Predict **Conditional Average Treatment Effect (CATE)** — the incremental impact of marketing actions.

| Model | Algorithm | Use Case |
|-------|-----------|----------|
| **Causal Forest** | EconML GRF | High-dimensional heterogeneity |
| **X-Learner** | Meta-learner | Unbalanced treatment groups |
| **T-Learner** | Two-model | Simple baseline |
| **S-Learner** | Single-model | Limited treatment data |

```python
# Example API call
POST /v1/uplift/predict
{
    "customer_id": "cust_123",
    "features": {"age": 35, "segment": "premium", "recency": 14},
    "treatment": "discount_10pct"
}

# Response
{
    "cate": 0.127,
    "confidence_interval": [0.089, 0.165],
    "recommendation": "TREAT"
}
```

### Demand Forecasting

Multi-horizon probabilistic predictions for inventory and supply chain.

| Model | Type | Features |
|-------|------|----------|
| **DeepAR** | RNN-based | Covariates support |
| **N-BEATS** | Pure ML | Interpretable stacks |

### Price Optimization

Elasticity-based pricing to maximize revenue or profit.

- Price elasticity estimation (log-log regression)
- Multi-constraint optimization (scipy.optimize)
- Competitor price response modeling

### A/B Experiments

Full experiment lifecycle management:

- **Design**: Power analysis, sample size calculation
- **Execution**: Assignment, event tracking
- **Analysis**: Statistical significance, confidence intervals

---

## 🔒 Security

Enterprise-grade security measures:

| Category | Implementation |
|----------|----------------|
| **Authentication** | JWT (RS256), API Keys |
| **Authorization** | RBAC with scopes |
| **Encryption** | AES-256-GCM (field-level) |
| **Transport** | TLS 1.3, HSTS |
| **Secrets** | HashiCorp Vault integration |
| **Audit** | Structured logging, compliance trail |
| **Input** | Pydantic v2 strict validation |
| **Dependencies** | Automated scanning (Snyk, pip-audit) |

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## 📚 API Documentation

### Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/v1/uplift/predict` | POST | CATE prediction |
| `/v1/forecast/demand` | POST | Demand forecast |
| `/v1/pricing/optimize` | POST | Price optimization |
| `/v1/experiments` | CRUD | Experiment management |

Full OpenAPI spec: [docs/api/openapi.yaml](docs/api/openapi.yaml)

---

## 📁 Project Structure

```
meridian/
├── src/meridian/           # Source code
│   ├── api/                # HTTP layer
│   │   ├── routers/        # Endpoint handlers
│   │   ├── schemas/        # Request/Response models
│   │   └── middleware/     # Security, logging
│   ├── core/               # Configuration, utilities
│   ├── domain/             # Business logic (pure Python)
│   │   ├── models/         # Domain entities
│   │   ├── services/       # Domain services
│   │   └── repositories/   # Repository interfaces
│   ├── application/        # Use cases
│   ├── infrastructure/     # External integrations
│   │   ├── database/       # SQLAlchemy, repositories
│   │   ├── ml/             # ML model wrappers
│   │   ├── cache/          # Redis
│   │   └── security/       # Encryption, vault
│   └── workers/            # Celery tasks
├── tests/                  # Comprehensive test suite
│   ├── unit/               # Fast isolated tests
│   ├── integration/        # External dependency tests
│   ├── e2e/                # Full scenario tests
│   └── security/           # Security-focused tests
├── configs/                # Hydra configurations
├── deployments/            # Docker, Kubernetes
├── docs/                   # Documentation
│   ├── architecture/       # ADRs, diagrams
│   └── api/                # OpenAPI spec
└── ui/                     # Streamlit dashboard
```

---

## 🛠️ Development

### Commands

```bash
# Run all checks
make check-all

# Format code
make format

# Run linters
make lint

# Run tests
make test

# Security scan
make security

# Generate coverage report
make test-cov
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 🐳 Deployment

### Docker Compose (Development)

```bash
make docker
```

Services:
- **app**: FastAPI application
- **postgres**: PostgreSQL 16
- **redis**: Redis 7
- **mlflow**: MLflow tracking server

### Kubernetes (Production)

```bash
kubectl apply -f deployments/k8s/
```

Includes:
- Deployment with HPA
- ConfigMaps & Secrets
- Ingress configuration
- Service mesh ready

---

## 📈 Monitoring

### Metrics

Prometheus endpoint at `/metrics`:
- Request latency (histogram)
- Request count by endpoint
- Error rates
- ML model inference time

### Logging

Structured JSON logging with:
- Request correlation IDs
- User context
- Audit trail

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Architecture Decision Records](docs/architecture/adr/) | Why we chose specific technologies |
| [C4 Diagrams](docs/architecture/diagrams/) | System context and containers |
| [OpenAPI Specification](docs/api/openapi.yaml) | Full API documentation |
| [Contributing Guide](CONTRIBUTING.md) | How to contribute |
| [Security Policy](SECURITY.md) | Vulnerability reporting |
| [Changelog](CHANGELOG.md) | Version history |

---

## 📄 License

Proprietary - All rights reserved.

---

<div align="center">

**Made with ❤️ for enterprise ML teams**

</div>

