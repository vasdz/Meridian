# Architecture Decision Record: Causal Inference Library

## Status
Accepted

## Context
We need to select libraries for causal inference and uplift modeling. Requirements:
- Support for heterogeneous treatment effect estimation
- Multiple meta-learner implementations
- Production-ready performance
- Active maintenance

## Decision
We use a combination of:
- **EconML** (Microsoft) for primary causal inference
- **CausalML** (Uber) as alternative/validation
- Custom implementations for specific use cases

## Rationale

### EconML
- Comprehensive causal ML library
- Causal Forest DML implementation
- Good integration with scikit-learn
- Active Microsoft Research backing
- Confidence intervals out of the box

### CausalML
- Strong meta-learner implementations (S, T, X, R learners)
- Uplift trees and forests
- Good visualization tools
- Production-tested at Uber

### Custom Implementations
- X-Learner for transparency and customization
- Better control over feature engineering
- Easier model explanation

## Alternatives Considered
- **DoWhy**: More focused on causal graph discovery
- **CausalImpact**: Time-series focused only
- **grf (R)**: R-only, harder to integrate

## Consequences
- Need to maintain multiple model types
- Testing required for model consistency
- Documentation for model selection criteria needed

