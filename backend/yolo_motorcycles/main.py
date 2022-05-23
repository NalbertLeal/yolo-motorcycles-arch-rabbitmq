from yolo_motorcycles.rabbit_comunication import producer_consumer

producer_consumer = producer_consumer.RabbitMQProducerConsumer('localhost', ['pre_computed.1'], ['yolo_motorcycle.'])