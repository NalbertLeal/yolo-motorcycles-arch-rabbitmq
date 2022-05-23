from yolo_motorcycles.rabbit_comunication import producer_consumer

producer_consumer = producer_consumer.RabbitMQProducerConsumer('localhost', ['pre_computed.0'], ['yolo_motorcycle.'])