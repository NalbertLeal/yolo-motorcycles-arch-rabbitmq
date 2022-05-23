from client.video_reader import reader
from client.rabbit_comunication import producer, consumer

media_name = 'casimiro_reage_motos'

casimiro = '/home/nalbertgml/Downloads/X2Download.com-CASIMIRO REAGE ROLE ZN - POLÍCIA ENQUADROU GERAL �� Nathan RJ Cortes do Casimito-(480p).mp4'
motos = '/home/nalbertgml/Documentos/python/motorcycles-only-kafka/backend/os-melhores-grau-e-corte-2020-14-🚀-grau-de-m.mp4'

video_reader = reader.Reader(motos)
rabbit_producer = producer.RabbitMQProducer('localhost', ['non_computed'], video_reader, media_name)
rabbit_consumer = consumer.RabbitMQConsumer('localhost', ['yolo_motorcycle.'+media_name])

rabbit_producer.start()
rabbit_consumer.start()
input()