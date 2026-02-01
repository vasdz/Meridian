# 🎯 Meridian

**Marketing Analytics & Causal Inference Platform**

Meridian is a production-ready platform for uplift modeling, demand forecasting, price optimization, and A/B experiment management.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
│                        (FastAPI)                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Uplift  │  │Forecasting│  │ Pricing  │  │Experiments│        │
│  │   API    │  │   API     │  │   API    │  │   API     │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │                │
├───────┴─────────────┴─────────────┴─────────────┴────────────────┤
│                      Domain Services                             │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐    │
│  │ UpliftCalculator│  │ExperimentDesign│  │PricingOptimizer │    │
│  └────────────────┘  └────────────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │PostgreSQL│  │  Redis  │  │ MLflow  │  │  Kafka  │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Poetry
- Docker & Docker Compose
- PostgreSQL 16+
- Redis 7+

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/meridian.git
cd meridian

# Install dependencies
poetry install

# Copy environment file
cp .env.example .env

# Start services with Docker
make docker

# Run migrations
make migrate

# Seed sample data
make seed

# Start development server
make dev
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Security tests
make test-security
```

## 📚 API Documentation

Once running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## 🧠 Features

### Uplift Modeling
- Causal Forest (EconML)
- X-Learner, T-Learner, S-Learner
- Confidence intervals
- CATE predictions

### Demand Forecasting
- DeepAR (GluonTS)
- N-BEATS
- Probabilistic forecasts

### Price Optimization
- Price elasticity estimation
- Profit/revenue maximization
- Constraint handling

### A/B Experiments
- Experiment lifecycle management
- Statistical analysis
- Power analysis & sample size calculation

## 🔒 Security

- JWT & API Key authentication
- Rate limiting
- Input validation
- SQL injection protection
- Field-level encryption
- Audit logging

## 🏛️ Project Structure

```
meridian/
├── src/meridian/           # Source code
│   ├── api/                # API layer (routers, schemas)
│   ├── core/               # Core utilities (config, logging)
│   ├── domain/             # Domain models & services
│   ├── application/        # Use cases
│   ├── infrastructure/     # External integrations
│   └── workers/            # Celery tasks
├── tests/                  # Test suite
├── configs/                # Hydra configurations
├── deployments/            # Docker & Kubernetes
├── docs/                   # Documentation
└── notebooks/              # Jupyter notebooks
```

## 📖 Documentation

- [Architecture Decision Records](docs/architecture/adr/)
- [API Specification](docs/api/openapi.yaml)
- [C4 Diagrams](docs/architecture/diagrams/)

## 🛠️ Development

```bash
# Format code
make format

# Run linters
make lint

# Security audit
make security
```

## 📄 License

Proprietary - All rights reserved.

