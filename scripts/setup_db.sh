#!/bin/bash
# Database setup script

set -e

echo "Setting up PostgreSQL database..."

# Create database and user
psql -U postgres <<EOF
CREATE USER meridian WITH PASSWORD 'meridian';
CREATE DATABASE meridian OWNER meridian;
GRANT ALL PRIVILEGES ON DATABASE meridian TO meridian;
EOF

echo "Database setup complete!"

