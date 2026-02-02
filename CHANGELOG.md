# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive security documentation (SECURITY.md)
- Contributor guidelines (CONTRIBUTING.md)
- Bilingual documentation (English/Russian)
- Gitleaks configuration for secret scanning
- Streamlit dashboard for ML model visualization
- Enhanced API documentation

### Changed
- Updated GitHub Actions workflows to use latest versions
- Improved test fixtures to avoid hardcoded secrets
- Enhanced security headers middleware

### Security
- Fixed potential secret leaks in test files
- Updated CodeQL action to v3
- Added permissions configuration to workflows

---

## [1.0.0] - 2026-02-01

### Added

#### Core Platform
- FastAPI-based REST API with OpenAPI documentation
- Clean Architecture implementation (Domain/Application/Infrastructure layers)
- Async SQLAlchemy 2.0 with PostgreSQL support
- Redis caching layer
- Celery workers for async tasks

#### ML Modules
- **Uplift Modeling**
  - Causal Forest (EconML integration)
  - X-Learner, T-Learner, S-Learner implementations
  - CATE prediction with confidence intervals
  - AUUC and Qini metrics

- **Demand Forecasting**
  - DeepAR integration (GluonTS)
  - N-BEATS model wrapper
  - Probabilistic forecasting with prediction intervals
  - Multi-horizon predictions

- **Price Optimization**
  - Price elasticity estimation
  - Profit/revenue maximization
  - Constraint-based optimization

- **A/B Testing**
  - Experiment lifecycle management
  - Statistical significance testing
  - Power analysis and sample size calculation

#### Security
- JWT and API Key authentication
- Rate limiting with Redis backend
- Input validation (Pydantic v2)
- SQL injection protection
- Security headers (HSTS, CSP, X-Frame-Options)
- Field-level encryption (AES-256-GCM)
- Audit logging

#### Infrastructure
- Docker Compose for local development
- Kubernetes deployment manifests
- GitHub Actions CI/CD pipelines
- MLflow integration for experiment tracking
- Prometheus metrics endpoint
- Structured logging (JSON format)

#### Documentation
- Architecture Decision Records (ADRs)
- C4 architecture diagrams
- OpenAPI specification

### Infrastructure
- Multi-stage Docker builds
- Kubernetes HPA for autoscaling
- Helm-ready configurations

---

## [0.1.0] - 2026-01-15

### Added
- Initial project structure
- Basic FastAPI setup
- Poetry dependency management
- pytest configuration

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-02-01 | Production-ready release with full ML stack |
| 0.1.0 | 2026-01-15 | Initial project setup |

---

[Unreleased]: https://github.com/your-org/meridian/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/meridian/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/your-org/meridian/releases/tag/v0.1.0
