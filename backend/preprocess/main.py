from preprocess.onnx_preprocessing import image
from preprocess.rabbit_comunication import producer_consumer

producer_consumer = producer_consumer.RabbitMQProducerConsumer('localhost', ['non_computed'], 'pre_computed.', 3)