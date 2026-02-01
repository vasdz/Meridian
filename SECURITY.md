# Security Policy | Политика безопасности

[English](#english) | [Русский](#русский)

---

## English

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

### Reporting a Vulnerability

We take security seriously at Meridian. If you discover a security vulnerability, please follow responsible disclosure practices.

#### How to Report

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Send a detailed report to: **security@meridian-analytics.io**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

#### Response Timeline

| Stage | Timeline |
|-------|----------|
| Initial Response | 24 hours |
| Preliminary Assessment | 72 hours |
| Fix Development | 7-14 days (severity dependent) |
| Public Disclosure | 90 days after fix |

#### Security Bug Bounty

We currently do not offer a paid bug bounty program, but we will:
- Credit you in our security advisories (with your permission)
- Provide a letter of recognition for responsible disclosure

### Security Measures

This project implements enterprise-grade security:

#### Authentication & Authorization
- JWT tokens with RSA256 signing
- API key authentication with rate limiting
- Role-based access control (RBAC)
- Constant-time comparison for sensitive data

#### Data Protection
- AES-256-GCM field-level encryption
- Secure key derivation (PBKDF2-HMAC-SHA256)
- No plaintext storage of secrets
- Environment-based configuration

#### Input Validation
- Pydantic v2 strict validation
- SQL injection prevention (parameterized queries)
- XSS protection headers
- Request size limits

#### Infrastructure Security
- Security headers (HSTS, CSP, X-Frame-Options)
- Audit logging for all sensitive operations
- Automated dependency scanning
- Container security (non-root user, read-only filesystem)

#### Secrets Management
- HashiCorp Vault integration
- Encrypted environment variables
- No secrets in version control

### Development Security

All contributors must:

1. Run security checks before committing:
   ```bash
   make security
   ```

2. Never commit:
   - API keys, tokens, or passwords
   - Private keys or certificates
   - Database credentials
   - Production configuration

3. Use pre-commit hooks:
   ```bash
   pre-commit install
   ```

---

## Русский

### Поддерживаемые версии

| Версия  | Поддержка          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

### Сообщение об уязвимости

Мы серьёзно относимся к безопасности в Meridian. Если вы обнаружили уязвимость, пожалуйста, следуйте практикам ответственного раскрытия.

#### Как сообщить

1. **НЕ** открывайте публичный issue на GitHub для уязвимостей
2. Отправьте подробный отчёт на: **security@meridian-analytics.io**
3. Включите:
   - Описание уязвимости
   - Шаги для воспроизведения
   - Потенциальное воздействие
   - Предлагаемое исправление (если есть)

#### Сроки ответа

| Этап | Срок |
|------|------|
| Первичный ответ | 24 часа |
| Предварительная оценка | 72 часа |
| Разработка исправления | 7-14 дней (зависит от серьёзности) |
| Публичное раскрытие | 90 дней после исправления |

#### Программа вознаграждений

На данный момент мы не предлагаем платную программу bug bounty, но мы:
- Укажем вас в security-уведомлениях (с вашего разрешения)
- Предоставим письмо о признании за ответственное раскрытие

### Меры безопасности

Проект реализует безопасность корпоративного уровня:

#### Аутентификация и авторизация
- JWT токены с подписью RSA256
- Аутентификация по API-ключам с rate limiting
- Ролевой доступ (RBAC)
- Сравнение чувствительных данных за константное время

#### Защита данных
- Шифрование полей AES-256-GCM
- Безопасное получение ключей (PBKDF2-HMAC-SHA256)
- Отсутствие хранения секретов в открытом виде
- Конфигурация через переменные окружения

#### Валидация ввода
- Строгая валидация Pydantic v2
- Защита от SQL-инъекций (параметризованные запросы)
- Заголовки защиты от XSS
- Лимиты на размер запросов

#### Безопасность инфраструктуры
- Заголовки безопасности (HSTS, CSP, X-Frame-Options)
- Аудит-логирование всех чувствительных операций
- Автоматическое сканирование зависимостей
- Безопасность контейнеров (non-root пользователь, read-only файловая система)

#### Управление секретами
- Интеграция с HashiCorp Vault
- Зашифрованные переменные окружения
- Отсутствие секретов в системе контроля версий

### Безопасность разработки

Все контрибьюторы должны:

1. Запускать проверки безопасности перед коммитом:
   ```bash
   make security
   ```

2. Никогда не коммитить:
   - API-ключи, токены или пароли
   - Приватные ключи или сертификаты
   - Учётные данные базы данных
   - Продакшен-конфигурацию

3. Использовать pre-commit хуки:
   ```bash
   pre-commit install
   ```

---

## Acknowledgments | Благодарности

We thank the following security researchers for their responsible disclosures:

*No disclosures yet - be the first!*

Мы благодарим следующих исследователей безопасности за ответственное раскрытие:

*Пока нет раскрытий - будьте первым!*

