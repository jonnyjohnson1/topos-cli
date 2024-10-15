from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import json
import asyncio

class KafkaManager:
    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self.consumer = None

    async def start(self):
        self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
        )
        await self.producer.start()
        await self.consumer.start()

    async def stop(self):
        await self.producer.stop()
        await self.consumer.stop()

    async def send_message(self, key, value):
        await self.producer.send_and_wait(self.topic, key=key.encode('utf-8'), value=json.dumps(value).encode('utf-8'))

    async def consume_messages(self, callback):
        async for msg in self.consumer:
            message = json.loads(msg.value.decode('utf-8'))
            group_id = msg.key.decode('utf-8')
            await callback(message, group_id)
