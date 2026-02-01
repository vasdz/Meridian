"""SQL Customer repository implementation."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from meridian.domain.models.customer import Customer
from meridian.domain.repositories.customer import CustomerRepository
from meridian.infrastructure.database.models.customer import CustomerModel


class SQLCustomerRepository(CustomerRepository):
    """SQL implementation of customer repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def _to_domain(self, model: CustomerModel) -> Customer:
        """Convert ORM model to domain entity."""
        return Customer(
            id=model.id,
            external_id=model.external_id,
            segment=model.segment,
            region=model.region,
            channel=model.channel,
            age=model.age,
            tenure_days=model.tenure_days,
            total_spend=model.total_spend,
            transaction_count=model.transaction_count,
            avg_basket_size=model.avg_basket_size,
            features=model.features or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _to_model(self, entity: Customer) -> CustomerModel:
        """Convert domain entity to ORM model."""
        return CustomerModel(
            id=entity.id,
            external_id=entity.external_id,
            segment=entity.segment.value if hasattr(entity.segment, "value") else entity.segment,
            region=entity.region,
            channel=entity.channel,
            age=entity.age,
            tenure_days=entity.tenure_days,
            total_spend=entity.total_spend,
            transaction_count=entity.transaction_count,
            avg_basket_size=entity.avg_basket_size,
            features=entity.features,
        )

    async def get(self, id: str) -> Optional[Customer]:
        result = await self.session.execute(
            select(CustomerModel).where(CustomerModel.id == id)
        )
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[Customer]:
        result = await self.session.execute(
            select(CustomerModel).limit(limit).offset(offset)
        )
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]

    async def add(self, entity: Customer) -> Customer:
        model = self._to_model(entity)
        self.session.add(model)
        await self.session.flush()
        return entity

    async def update(self, entity: Customer) -> Customer:
        model = await self.session.get(CustomerModel, entity.id)
        if model:
            model.segment = entity.segment.value if hasattr(entity.segment, "value") else entity.segment
            model.region = entity.region
            model.total_spend = entity.total_spend
            model.transaction_count = entity.transaction_count
            await self.session.flush()
        return entity

    async def delete(self, id: str) -> bool:
        model = await self.session.get(CustomerModel, id)
        if model:
            await self.session.delete(model)
            return True
        return False

    async def exists(self, id: str) -> bool:
        result = await self.session.execute(
            select(CustomerModel.id).where(CustomerModel.id == id)
        )
        return result.scalar_one_or_none() is not None

    async def get_by_external_id(self, external_id: str) -> Optional[Customer]:
        result = await self.session.execute(
            select(CustomerModel).where(CustomerModel.external_id == external_id)
        )
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None

    async def get_by_segment(
        self,
        segment: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Customer]:
        result = await self.session.execute(
            select(CustomerModel)
            .where(CustomerModel.segment == segment)
            .limit(limit)
            .offset(offset)
        )
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]

    async def get_features(self, customer_ids: list[str]) -> dict[str, dict]:
        result = await self.session.execute(
            select(CustomerModel).where(CustomerModel.id.in_(customer_ids))
        )
        models = result.scalars().all()
        return {
            m.id: self._to_domain(m).to_feature_dict()
            for m in models
        }

    async def update_segment(self, customer_id: str, new_segment: str) -> bool:
        model = await self.session.get(CustomerModel, customer_id)
        if model:
            model.segment = new_segment
            await self.session.flush()
            return True
        return False

