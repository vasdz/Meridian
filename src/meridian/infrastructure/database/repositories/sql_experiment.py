"""SQL Experiment repository implementation."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from meridian.core.constants import ExperimentStatus
from meridian.domain.models.experiment import Experiment, ExperimentResults, ExperimentVariant
from meridian.domain.repositories.experiment import ExperimentRepository
from meridian.infrastructure.database.models.experiment import ExperimentModel


class SQLExperimentRepository(ExperimentRepository):
    """SQL implementation of experiment repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def _to_domain(self, model: ExperimentModel) -> Experiment:
        variants = [
            ExperimentVariant(name=v["name"], weight=v.get("weight", 0.5))
            for v in (model.variants or [])
        ]

        results = None
        if model.results:
            results = ExperimentResults(**model.results)

        return Experiment(
            id=model.id,
            name=model.name,
            hypothesis=model.hypothesis,
            status=ExperimentStatus(model.status),
            variants=variants,
            primary_metric=model.primary_metric,
            target_sample_size=model.target_sample_size,
            target_mde=model.target_mde,
            confidence_level=model.confidence_level,
            start_date=model.start_date,
            end_date=model.end_date,
            results=results,
            winner_variant=model.winner_variant,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def get(self, id: str) -> Experiment | None:
        model = await self.session.get(ExperimentModel, id)
        return self._to_domain(model) if model else None

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[Experiment]:
        result = await self.session.execute(select(ExperimentModel).limit(limit).offset(offset))
        return [self._to_domain(m) for m in result.scalars().all()]

    async def add(self, entity: Experiment) -> Experiment:
        model = ExperimentModel(
            id=entity.id,
            name=entity.name,
            hypothesis=entity.hypothesis,
            status=entity.status.value,
            variants=[{"name": v.name, "weight": v.weight} for v in entity.variants],
            primary_metric=entity.primary_metric,
            target_sample_size=entity.target_sample_size,
            target_mde=entity.target_mde,
            confidence_level=entity.confidence_level,
        )
        self.session.add(model)
        await self.session.flush()
        return entity

    async def update(self, entity: Experiment) -> Experiment:
        model = await self.session.get(ExperimentModel, entity.id)
        if model:
            model.status = entity.status.value
            model.start_date = entity.start_date
            model.end_date = entity.end_date
            model.winner_variant = entity.winner_variant
            if entity.results:
                model.results = {
                    "lift": entity.results.lift,
                    "p_value": entity.results.p_value,
                    "is_significant": entity.results.is_significant,
                }
            await self.session.flush()
        return entity

    async def delete(self, id: str) -> bool:
        model = await self.session.get(ExperimentModel, id)
        if model:
            await self.session.delete(model)
            return True
        return False

    async def exists(self, id: str) -> bool:
        result = await self.session.execute(
            select(ExperimentModel.id).where(ExperimentModel.id == id)
        )
        return result.scalar_one_or_none() is not None

    async def get_by_status(
        self,
        status: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Experiment]:
        result = await self.session.execute(
            select(ExperimentModel)
            .where(ExperimentModel.status == status)
            .limit(limit)
            .offset(offset)
        )
        return [self._to_domain(m) for m in result.scalars().all()]

    async def get_running(self) -> list[Experiment]:
        return await self.get_by_status("running")

    async def update_status(self, experiment_id: str, new_status: str) -> bool:
        model = await self.session.get(ExperimentModel, experiment_id)
        if model:
            model.status = new_status
            await self.session.flush()
            return True
        return False

    async def set_results(self, experiment_id: str, results: dict) -> bool:
        model = await self.session.get(ExperimentModel, experiment_id)
        if model:
            model.results = results
            await self.session.flush()
            return True
        return False
