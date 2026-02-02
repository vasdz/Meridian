# 🎯 Meridian

<div align="center">

[![CI](https://github.com/vasdz/Meridian/actions/workflows/ci.yml/badge.svg)](https://github.com/vasdz/Meridian/actions/workflows/ci.yml)
[![Security](https://github.com/vasdz/Meridian/actions/workflows/security.yml/badge.svg)](https://github.com/vasdz/Meridian/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/vasdz/Meridian/branch/main/graph/badge.svg)](https://codecov.io/gh/vasdz/Meridian)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](../LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[🇬🇧 English](../README.md) | **🇷🇺 Русский**

*ML-платформа корпоративного уровня для маркетинговой аналитики и каузального вывода*

[Документация](#-документация) •
[Быстрый старт](#-быстрый-старт) •
[Возможности](#-возможности) •
[Архитектура](#-архитектура) •
[Операции](#-операции)

</div>

---

## 📖 Обзор

Meridian — это **production-ready ML платформа** для ритейл-аналитики, предоставляющая:

- 🎯 **Uplift-моделирование** — Предсказание индивидуальных эффектов воздействия (CATE)
- 📈 **Прогнозирование спроса** — Многошаговые вероятностные предсказания
- 💰 **Оптимизация цен** — Максимизация прибыли с моделями эластичности
- 🧪 **A/B тестирование** — Дизайн и анализ экспериментов
- 🧭 **MLOps основы** — Реестр моделей, мониторинг, аудитируемость

Построен на принципах **Clean Architecture** для корпоративной масштабируемости.

---

## 📚 Документация

| Документ | English | Русский |
|----------|---------|---------|
| Обзор проекта | `README.md` | `docs/README_RU.md` |
| OpenAPI спецификация | `docs/api/openapi.yaml` | `docs/api/openapi.yaml` |
| ADR по архитектуре | `docs/architecture/adr/` | `docs/architecture/adr/` |
| Политика безопасности | `SECURITY.md` | `SECURITY.md` |
| Гайд контрибьютора | `CONTRIBUTING.md` | `CONTRIBUTING.md` |

---

## 🏗️ Архитектура

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
│                            Доменные сервисы                                  │
│  ┌────────────────────┐  ┌─────────────────────┐  ┌───────────────────┐    │
│  │  UpliftCalculator  │  │  ExperimentDesign   │  │ PricingOptimizer  │    │
│  │  (Causal Forest)   │  │  (Power Analysis)   │  │  (Elasticity)     │    │
│  └────────────────────┘  └─────────────────────┘  └───────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Инфраструктура                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ PostgreSQL  │  │    Redis    │  │   MLflow    │  │    Kafka    │        │
│  │    (БД)     │  │   (Кэш)     │  │ (Tracking)  │  │  (События)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Ответственности слоёв

| Слой | Назначение | Ключевые компоненты |
|------|------------|---------------------|
| **API** | HTTP-интерфейс, валидация | Роутеры, Схемы, Middleware |
| **Application** | Use cases, оркестрация | UseCases, DTOs |
| **Domain** | Бизнес-логика | Модели, Сервисы, Репозитории (интерфейсы) |
| **Infrastructure** | Внешние системы | БД, ML, Кэш, Сообщения |

---

## 🚀 Быстрый старт

### Требования

| Требование | Версия |
|------------|--------|
| Python | 3.12+ |
| Poetry | 1.8+ |
| Docker | 24+ |
| Docker Compose | 2.20+ |

### Установка

```bash
# Клонируйте репозиторий
git clone https://github.com/vasdz/Meridian.git
cd Meridian

# Установите зависимости
poetry install

# Скопируйте файл окружения
cp .env.example .env

# Запустите инфраструктурные сервисы
make docker

# Выполните миграции БД
make migrate

# Загрузите тестовые данные
make seed

# Запустите сервер разработки
make dev
```

Доступ к приложению:
- 🌐 **API**: http://localhost:8000
- 📚 **Swagger UI**: http://localhost:8000/docs
- 📖 **ReDoc**: http://localhost:8000/redoc
- 📊 **Дашборд**: http://localhost:8501 (Streamlit)

---

## 🧠 Возможности

### Uplift-моделирование

Предсказание **Conditional Average Treatment Effect (CATE)** — инкрементального влияния маркетинговых действий.

| Модель | Алгоритм | Применение |
|--------|----------|------------|
| **Causal Forest** | EconML GRF | Высокоразмерная гетерогенность |
| **X-Learner** | Meta-learner | Несбалансированные группы |
| **T-Learner** | Две модели | Простой baseline |
| **S-Learner** | Одна модель | Ограниченные данные |

```python
# Пример API-запроса
POST /v1/uplift/predict
{
    "customer_id": "cust_123",
    "features": {"age": 35, "segment": "premium", "recency": 14},
    "treatment": "discount_10pct"
}

# Ответ
{
    "cate": 0.127,
    "confidence_interval": [0.089, 0.165],
    "recommendation": "TREAT"
}
```

### Прогнозирование спроса

Многошаговые вероятностные предсказания для управления запасами.

| Модель | Тип | Особенности |
|--------|-----|-------------|
| **DeepAR** | На основе RNN | Поддержка ковариат |
| **N-BEATS** | Чистый ML | Интерпретируемые стеки |

### Оптимизация цен

Ценообразование на основе эластичности для максимизации выручки или прибыли.

- Оценка ценовой эластичности (log-log регрессия)
- Оптимизация с ограничениями (scipy.optimize)
- Моделирование реакции конкурентов

### A/B эксперименты

Полное управление жизненным циклом экспериментов:

- **Дизайн**: Power analysis, расчёт размера выборки
- **Выполнение**: Назначение, отслеживание событий
- **Анализ**: Статистическая значимость, доверительные интервалы

---

## 🔒 Безопасность

Меры безопасности корпоративного уровня:

| Категория | Реализация |
|-----------|------------|
| **Аутентификация** | JWT (RS256), API Keys |
| **Авторизация** | RBAC с scopes |
| **Шифрование** | AES-256-GCM (на уровне полей) |
| **Транспорт** | TLS 1.3, HSTS |
| **Секреты** | Интеграция HashiCorp Vault |
| **Аудит** | Структурированное логирование |
| **Ввод** | Строгая валидация Pydantic v2 |
| **Зависимости** | Автосканирование (Snyk, pip-audit) |

См. `SECURITY.md` для отчётов об уязвимостях.

---

## 📈 Наблюдаемость

| Возможность | Детали |
|------------|--------|
| **Метрики** | Prometheus метрики на `/metrics` |
| **Логи** | JSON логирование с correlation IDs |
| **Трейсинг** | OTLP-ready (Tempo/Jaeger совместимость) |
| **Аудит** | Audit trail для чувствительных действий |

### Трейсинг (OTLP)

Включение через переменные окружения:

```bash
TRACING_ENABLED=true
TRACING_SERVICE_NAME=meridian-api
TRACING_OTLP_ENDPOINT=http://localhost:4318/v1/traces
```

Сервис отправляет атрибуты `service.version` и `deployment.environment`.

### Логирование запросов и ответов

- Тело запроса редактируется по PII ключам и ограничивается по размеру.
- Логирование ответа включается через `LOG_RESPONSE_BODY`.
- Лимиты: `MAX_REQUEST_BODY_BYTES`, `MAX_RESPONSE_BODY_BYTES`.

---

## 🔁 Версионирование API и депрекации

- Версионирование через `/v1/...` с обратной совместимостью.
- Депрекации через заголовки и staged feature flags.
- Изменения совместимости отражаются в `CHANGELOG.md`.

---

## 🧰 Операции

| Область | Подход |
|---------|--------|
| **Реестр моделей** | MLflow tracking с подписью артефактов (cosign-ready) |
| **Feature Store** | Redis через интерфейс `FeatureStore.get()` |
| **GitOps** | Манифесты под ArgoCD в `deployments/k8s/` |
| **Секреты** | External Secrets Operator / Vault |

---

## 📚 API-документация

### Обзор эндпоинтов

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Проверка здоровья |
| `/ready` | GET | Готовность к работе |
| `/metrics` | GET | Метрики Prometheus |
| `/v1/uplift/predict` | POST | Предсказание CATE |
| `/v1/forecast/demand` | POST | Прогноз спроса |
| `/v1/pricing/optimize` | POST | Оптимизация цен |
| `/v1/experiments` | CRUD | Управление экспериментами |

Полная спецификация OpenAPI: [docs/api/openapi.yaml](api/openapi.yaml)

---

## 📁 Структура проекта

```
meridian/
├── src/meridian/           # Исходный код
│   ├── api/                # HTTP-слой
│   │   ├── routers/        # Обработчики эндпоинтов
│   │   ├── schemas/        # Модели Request/Response
│   │   └── middleware/     # Безопасность, логирование
│   ├── core/               # Конфигурация, утилиты
│   ├── domain/             # Бизнес-логика (чистый Python)
│   │   ├── models/         # Доменные сущности
│   │   ├── services/       # Доменные сервисы
│   │   └── repositories/   # Интерфейсы репозиториев
│   ├── application/        # Use cases
│   ├── infrastructure/     # Внешние интеграции
│   │   ├── database/       # SQLAlchemy, репозитории
│   │   ├── ml/             # ML-модели
│   │   ├── cache/          # Redis
│   │   └── security/       # Шифрование, vault
│   └── workers/            # Celery задачи
├── tests/                  # Тестовый набор
│   ├── unit/               # Быстрые изолированные тесты
│   ├── integration/        # Тесты с внешними зависимостями
│   ├── e2e/                # Полные сценарии
│   └── security/           # Тесты безопасности
├── deployments/            # Docker, Kubernetes
├── docs/                   # Документация
│   ├── architecture/       # ADRs, диаграммы
│   └── api/                # OpenAPI спецификация
└── ui/                     # Streamlit дашборд
```

---

## 🛠️ Разработка

### Команды

```bash
# Все проверки
make check-all

# Форматирование кода
make format

# Запуск линтеров
make lint

# Запуск тестов
make test

# Сканирование безопасности
make security

# Отчёт о покрытии
make test-cov
```

### Pre-commit хуки

```bash
# Установка хуков
pre-commit install

# Ручной запуск
pre-commit run --all-files
```

---

## 🐳 Развёртывание

### Docker Compose (Разработка)

```bash
make docker
```

Сервисы:
- **app**: FastAPI приложение
- **postgres**: PostgreSQL 16
- **redis**: Redis 7
- **mlflow**: MLflow tracking сервер

### Kubernetes (Продакшен)

```bash
kubectl apply -f deployments/k8s/
```

Включает:
- Deployment с HPA
- ConfigMaps & Secrets
- Конфигурацию Ingress
- Готовность к service mesh

---

## 📄 Лицензия

Проприетарная - Все права защищены.

---

<div align="center">

**Сделано с ❤️ для enterprise ML-команд**

</div>
