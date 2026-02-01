"""Seed database with sample data."""

import asyncio
import uuid
from datetime import datetime, timedelta
import random

from meridian.infrastructure.database.connection import init_db, get_async_session
from meridian.infrastructure.database.models.customer import CustomerModel
from meridian.infrastructure.database.models.transaction import TransactionModel
from meridian.infrastructure.database.models.experiment import ExperimentModel


async def seed_customers(session, n: int = 1000):
    """Seed customer data."""
    segments = ["high_value", "medium_value", "low_value", "new", "at_risk"]
    regions = ["North", "South", "East", "West"]
    channels = ["online", "store", "mobile"]

    customers = []
    for i in range(n):
        customer = CustomerModel(
            id=str(uuid.uuid4()),
            external_id=f"EXT-{i:06d}",
            segment=random.choice(segments),
            region=random.choice(regions),
            channel=random.choice(channels),
            age=random.randint(18, 75),
            tenure_days=random.randint(0, 3650),
            total_spend=random.uniform(0, 10000),
            transaction_count=random.randint(0, 100),
            avg_basket_size=random.uniform(10, 200),
        )
        customers.append(customer)

    session.add_all(customers)
    await session.commit()
    print(f"Seeded {n} customers")
    return customers


async def seed_transactions(session, customers, n_per_customer: int = 10):
    """Seed transaction data."""
    products = [f"PROD-{i:04d}" for i in range(100)]
    categories = ["Electronics", "Clothing", "Food", "Home", "Sports"]

    transactions = []
    for customer in customers[:100]:  # Limit for demo
        for _ in range(random.randint(1, n_per_customer)):
            tx = TransactionModel(
                id=str(uuid.uuid4()),
                customer_id=customer.id,
                amount=random.uniform(10, 500),
                quantity=random.randint(1, 5),
                product_id=random.choice(products),
                category=random.choice(categories),
                channel=customer.channel,
                unit_price=random.uniform(5, 100),
                discount_amount=random.uniform(0, 20),
            )
            transactions.append(tx)

    session.add_all(transactions)
    await session.commit()
    print(f"Seeded {len(transactions)} transactions")


async def seed_experiments(session, n: int = 5):
    """Seed experiment data."""
    experiments = []
    for i in range(n):
        exp = ExperimentModel(
            id=str(uuid.uuid4()),
            name=f"Experiment {i+1}",
            hypothesis=f"Testing hypothesis {i+1} for uplift improvement",
            status=random.choice(["draft", "running", "completed"]),
            variants=[
                {"name": "control", "weight": 0.5},
                {"name": "treatment", "weight": 0.5},
            ],
            primary_metric="conversion_rate",
            target_sample_size=10000,
            confidence_level=0.95,
        )
        experiments.append(exp)

    session.add_all(experiments)
    await session.commit()
    print(f"Seeded {n} experiments")


async def main():
    """Run seeding."""
    print("Initializing database...")
    await init_db()

    async with get_async_session() as session:
        print("Seeding data...")
        customers = await seed_customers(session, n=1000)
        await seed_transactions(session, customers)
        await seed_experiments(session)

    print("Seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())

