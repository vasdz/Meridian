"""SQL Transaction repository implementation."""

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from meridian.domain.models.transaction import Transaction
from meridian.domain.repositories.transaction import TransactionRepository
from meridian.infrastructure.database.models.transaction import TransactionModel


class SQLTransactionRepository(TransactionRepository):
    """SQL implementation of transaction repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def _to_domain(self, model: TransactionModel) -> Transaction:
        return Transaction(
            id=model.id,
            customer_id=model.customer_id,
            amount=model.amount,
            quantity=model.quantity,
            product_id=model.product_id,
            category=model.category,
            channel=model.channel,
            store_id=model.store_id,
            unit_price=model.unit_price,
            discount_amount=model.discount_amount,
            is_return=model.is_return,
            promotion_id=model.promotion_id,
            created_at=model.created_at,
        )

    async def get(self, id: str) -> Transaction | None:
        model = await self.session.get(TransactionModel, id)
        return self._to_domain(model) if model else None

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[Transaction]:
        result = await self.session.execute(select(TransactionModel).limit(limit).offset(offset))
        return [self._to_domain(m) for m in result.scalars().all()]

    async def add(self, entity: Transaction) -> Transaction:
        model = TransactionModel(**entity.to_dict())
        self.session.add(model)
        await self.session.flush()
        return entity

    async def update(self, entity: Transaction) -> Transaction:
        model = await self.session.get(TransactionModel, entity.id)
        if model:
            model.amount = entity.amount
            await self.session.flush()
        return entity

    async def delete(self, id: str) -> bool:
        model = await self.session.get(TransactionModel, id)
        if model:
            await self.session.delete(model)
            return True
        return False

    async def exists(self, id: str) -> bool:
        result = await self.session.execute(
            select(TransactionModel.id).where(TransactionModel.id == id)
        )
        return result.scalar_one_or_none() is not None

    async def get_by_customer(
        self,
        customer_id: str,
        limit: int = 100,
    ) -> list[Transaction]:
        result = await self.session.execute(
            select(TransactionModel)
            .where(TransactionModel.customer_id == customer_id)
            .order_by(TransactionModel.created_at.desc())
            .limit(limit)
        )
        return [self._to_domain(m) for m in result.scalars().all()]

    async def get_by_date_range(
        self,
        start_date: date,
        end_date: date,
        limit: int = 1000,
    ) -> list[Transaction]:
        result = await self.session.execute(
            select(TransactionModel)
            .where(TransactionModel.created_at >= start_date)
            .where(TransactionModel.created_at <= end_date)
            .limit(limit)
        )
        return [self._to_domain(m) for m in result.scalars().all()]

    async def get_customer_totals(self, customer_id: str) -> dict:
        result = await self.session.execute(
            select(
                func.sum(TransactionModel.amount).label("total_amount"),
                func.count(TransactionModel.id).label("count"),
                func.avg(TransactionModel.amount).label("avg_amount"),
            ).where(TransactionModel.customer_id == customer_id)
        )
        row = result.one()
        return {
            "total_amount": float(row.total_amount or 0),
            "count": int(row.count or 0),
            "avg_amount": float(row.avg_amount or 0),
        }
