import cv2
from client.kafka_comunication import producer, consumer
from client.video_reader import reader

producer = producer.kafkaProducer('localhost', 9092, 'cortes_do_casimiro_motos')
# consumer = consumer.KafkaConsumer('localhost', 9092, 108)
video_reader = reader.Reader('/home/nalbertgml/Downloads/X2Download.com-CASIMIRO REAGE ROLE ZN - POLÍCIA ENQUADROU GERAL �� Nathan RJ Cortes do Casimito-(480p).mp4')

while True:
    has_frame, frame = video_reader.next_frame()
    if not has_frame:
        break
    producer.send_frame(frame)

# for kafka_response in consumer.consumer:
#     (bboxes, frame) = consumer.deserialize_kafka_response(kafka_response)
#     cv2.imshow('video result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break