"""Kafka producer for event streaming."""

import json

from meridian.core.config import settings
from meridian.core.logging import get_logger

logger = get_logger(__name__)


class KafkaProducer:
    """Kafka producer for publishing events."""

    def __init__(self, bootstrap_servers: str | None = None):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._producer = None

    def _get_producer(self):
        """Get Kafka producer (lazy initialization)."""
        if self._producer is None and settings.kafka_enabled:
            try:
                from confluent_kafka import Producer

                self._producer = Producer(
                    {
                        "bootstrap.servers": self.bootstrap_servers,
                        "acks": "all",
                    }
                )
                logger.info("Kafka producer initialized")
            except ImportError:
                logger.warning("confluent-kafka not installed")
        return self._producer

    def send(
        self,
        topic: str,
        key: str,
        value: dict,
        headers: dict | None = None,
    ) -> bool:
        """Send message to Kafka topic."""
        producer = self._get_producer()

        if producer is None:
            logger.debug("Kafka disabled, message not sent", topic=topic)
            return False

        try:
            headers_list = None
            if headers:
                headers_list = [(k, v.encode()) for k, v in headers.items()]

            producer.produce(
                topic=topic,
                key=key.encode(),
                value=json.dumps(value).encode(),
                headers=headers_list,
            )
            producer.flush()

            logger.debug("Message sent to Kafka", topic=topic, key=key)
            return True

        except Exception as e:
            logger.error("Failed to send Kafka message", error=str(e))
            return False

    def close(self):
        """Close producer."""
        if self._producer:
            self._producer.flush()
            logger.info("Kafka producer closed")
