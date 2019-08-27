from kafka import KafkaProducer
from json import dumps
from time import sleep
import pandas as pd

def connect_kafka_producer():
    _producer = None
    try:
        _producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                                  value_serializer=lambda x: dumps(x).encode('utf-8'))
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(str(ex))
    finally:
        return _producer

if __name__ == '__main__':
    df = pd.read_csv('kc_house_data.csv')
    data = df[["bedrooms", "bathrooms", "floors", "condition", "grade", "price"]]

    kafka_producer = connect_kafka_producer()
    
    for index, row in data.iterrows():
        data_to_send = {'bedrooms': row['bedrooms'],
                        'bathrooms': row['bathrooms'],
                        'floors': row['floors'],
                        'condition': row['condition'],
                        'grade': row['grade'],
                        'price': row['price']}
        print(data_to_send)        	
        kafka_producer.send('house_data', value=data_to_send)
        sleep(0.2)
        
    if kafka_producer is not None:
    	kafka_producer.close()
