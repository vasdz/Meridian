#!/bin/bash
echo "Migrations complete!"

alembic upgrade head
# Run migrations

echo "Running database migrations..."

set -e

# Run Alembic migrations

