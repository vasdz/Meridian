# Contributing to Meridian | Руководство для контрибьюторов

[English](#english) | [Русский](#русский)

---

## English

Thank you for your interest in contributing to Meridian! This document provides guidelines and instructions for contributing.

### Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment.

### How to Contribute

#### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version)

#### Suggesting Features

1. Check the roadmap and existing feature requests
2. Open an issue with the feature request template
3. Explain the use case and business value

#### Submitting Code

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Set up development environment**
   ```bash
   poetry install --with dev,test
   pre-commit install
   ```
4. **Make your changes**
5. **Run tests and checks**
   ```bash
   make check-all
   ```
6. **Commit with conventional commits**
   ```bash
   git commit -m "feat: add new uplift model variant"
   ```
7. **Push and create Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/meridian.git
cd meridian

# Install dependencies
poetry install --with dev,test,docs

# Install pre-commit hooks
pre-commit install

# Copy environment file
cp .env.example .env

# Start services
make docker

# Run migrations
make migrate

# Verify setup
make test
```

### Code Style

We use automated tools to maintain code quality:

| Tool | Purpose |
|------|---------|
| **Ruff** | Linting (replaces flake8, isort, pyupgrade) |
| **Black** | Code formatting |
| **MyPy** | Static type checking |
| **Bandit** | Security linting |

Run all checks:
```bash
make lint
```

Format code:
```bash
make format
```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `security`: Security improvements

**Examples:**
```
feat(uplift): add X-learner implementation
fix(api): correct rate limiting calculation
docs(readme): update installation instructions
security(auth): strengthen JWT validation
```

### Testing

All code must have tests:

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/unit/domain/test_uplift.py

# Run with coverage
make test-cov

# Run security tests
make test-security
```

**Test structure:**
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with external dependencies
├── e2e/            # Full scenario tests
└── security/       # Security-focused tests
```

### Architecture Guidelines

We follow **Clean Architecture** principles:

1. **Domain layer** - Pure business logic, no external dependencies
2. **Application layer** - Use cases, orchestration
3. **Infrastructure layer** - External integrations
4. **API layer** - HTTP handlers, serialization

**Key principles:**
- Dependencies point inward (toward domain)
- Domain models are ORM-independent
- Use dependency injection
- Write tests for all layers

### Pull Request Process

1. Update documentation if needed
2. Add/update tests for your changes
3. Ensure CI passes
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

### Release Process

We use semantic versioning:
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

---

## Русский

Спасибо за интерес к участию в Meridian! Этот документ содержит руководства и инструкции для контрибьюторов.

### Кодекс поведения

Участвуя в проекте, вы соглашаетесь поддерживать уважительную и инклюзивную среду.

### Как внести вклад

#### Сообщение об ошибках

1. Проверьте существующие issues на наличие дубликатов
2. Используйте шаблон bug report
3. Включите:
   - Чёткое описание
   - Шаги для воспроизведения
   - Ожидаемое vs фактическое поведение
   - Детали окружения (ОС, версия Python)

#### Предложение функций

1. Проверьте roadmap и существующие запросы
2. Откройте issue с шаблоном feature request
3. Объясните use case и бизнес-ценность

#### Отправка кода

1. **Форкните репозиторий**
2. **Создайте feature-ветку**
   ```bash
   git checkout -b feature/название-функции
   ```
3. **Настройте среду разработки**
   ```bash
   poetry install --with dev,test
   pre-commit install
   ```
4. **Внесите изменения**
5. **Запустите тесты и проверки**
   ```bash
   make check-all
   ```
6. **Закоммитьте с conventional commits**
   ```bash
   git commit -m "feat: добавить новый вариант uplift модели"
   ```
7. **Запушьте и создайте Pull Request**

### Настройка разработки

```bash
# Клонируйте форк
git clone https://github.com/your-username/meridian.git
cd meridian

# Установите зависимости
poetry install --with dev,test,docs

# Установите pre-commit хуки
pre-commit install

# Скопируйте файл окружения
cp .env.example .env

# Запустите сервисы
make docker

# Выполните миграции
make migrate

# Проверьте установку
make test
```

### Стиль кода

Мы используем автоматизированные инструменты:

| Инструмент | Назначение |
|------------|------------|
| **Ruff** | Линтинг (заменяет flake8, isort, pyupgrade) |
| **Black** | Форматирование кода |
| **MyPy** | Статическая проверка типов |
| **Bandit** | Security-линтинг |

Запуск всех проверок:
```bash
make lint
```

Форматирование кода:
```bash
make format
```

### Конвенция коммитов

Мы следуем [Conventional Commits](https://www.conventionalcommits.org/):

```
<тип>(<область>): <описание>

[опциональное тело]

[опциональный футер]
```

**Типы:**
- `feat`: Новая функция
- `fix`: Исправление ошибки
- `docs`: Документация
- `style`: Форматирование
- `refactor`: Рефакторинг
- `perf`: Улучшение производительности
- `test`: Добавление тестов
- `chore`: Обслуживание
- `ci`: Изменения CI/CD
- `security`: Улучшения безопасности

### Тестирование

Весь код должен иметь тесты:

```bash
# Запуск всех тестов
make test

# Запуск конкретного файла
poetry run pytest tests/unit/domain/test_uplift.py

# С покрытием
make test-cov

# Security тесты
make test-security
```

### Принципы архитектуры

Мы следуем **Clean Architecture**:

1. **Domain layer** - Чистая бизнес-логика
2. **Application layer** - Use cases, оркестрация
3. **Infrastructure layer** - Внешние интеграции
4. **API layer** - HTTP обработчики

**Ключевые принципы:**
- Зависимости направлены внутрь (к домену)
- Доменные модели независимы от ORM
- Используйте dependency injection
- Пишите тесты для всех слоёв

### Процесс Pull Request

1. Обновите документацию при необходимости
2. Добавьте/обновите тесты
3. Убедитесь, что CI проходит
4. Запросите review
5. Ответьте на feedback
6. Сквошьте коммиты при запросе

---

## Questions? | Вопросы?

- Open a [Discussion](https://github.com/your-org/meridian/discussions)
- Email: team@meridian-analytics.io

