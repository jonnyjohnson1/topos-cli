import asyncio
import json
from aiokafka import AIOKafkaConsumer, TopicPartition
from typing import List, Set, Dict, Any

KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'chat_topic'
class MissedMessageManager:

    async def get_filtered_missed_messages(self,
        timestamp_ms: int,
        key_filter: Set[str]
        # max_messages: int = 1000
    ) -> List[Dict[str, str]]:
        consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=None,  # Set to None to avoid committing offsets
        auto_offset_reset='earliest'
    )

        try:
            await consumer.start()

            # Get partitions for the topic
            partitions = consumer.partitions_for_topic(KAFKA_TOPIC)
            if not partitions:
                raise ValueError(f"Topic '{KAFKA_TOPIC}' not found")

            # Create TopicPartition objects
            tps = [TopicPartition(KAFKA_TOPIC, p) for p in partitions]

            # Find offsets for the given timestamp
            offsets = await consumer.offsets_for_times({tp: timestamp_ms for tp in tps})
            print(offsets)
            # Seek to the correct offset for each partition
            for tp, offset_and_timestamp in offsets.items():
                if offset_and_timestamp is None:
                    # If no offset found for the timestamp, seek to the end
                    consumer.seek_to_end(tp)
                else:
                    print(tp)
                    print(offset_and_timestamp.offset)
                    consumer.seek(tp, offset_and_timestamp.offset)

            # Collect filtered messages
            missed_messages = []
            while True:
                try:
                    message = await asyncio.wait_for(consumer.getone(), timeout=1.0)
                    if message.key and message.key.decode() in key_filter:
                        missed_messages.append({
                            "key": message.key.decode(),
                            "value": json.loads(message.value.decode()),
                            "msg_type": "MISSED"
                        })
                except asyncio.TimeoutError:
                    # No more messages within the timeout period
                    break

            return missed_messages

        finally:
            await consumer.stop()
