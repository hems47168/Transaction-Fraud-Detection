from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from kafka import KafkaProducer, KafkaConsumer
import json
import uuid
import threading


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


results = {}


class Transaction(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


def consume_results():

    consumer = KafkaConsumer(
        "fraud_results",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="fraud-result-api",
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        )
    )

    for message in consumer:

        result_data = message.value

        print(
            "Kafka result received:",
            result_data
        )

        transaction_id = result_data.get(
            "transaction_id"
        )

        result = result_data.get(
            "result"
        )

        fraud_probability = result_data.get(
            "fraudProbability"
        )

        if transaction_id:

            results[transaction_id] = {
                "result": result,
                "fraudProbability": fraud_probability
            }


result_thread = threading.Thread(
    target=consume_results,
    daemon=True
)

result_thread.start()


@app.get("/")
def home():

    return {
        "message": "Fraud Detection API is running"
    }


@app.post("/transaction")
def receive_transaction(
    transaction: Transaction
):

    transaction_id = str(
        uuid.uuid4()
    )

    transaction_data = {
        "transaction_id": transaction_id,
        "type": transaction.type,
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "oldbalanceDest": transaction.oldbalanceDest,
        "newbalanceDest": transaction.newbalanceDest
    }

    producer.send(
        "bank_transactions",
        transaction_data
    )

    producer.flush()

    return {
        "message": "Transaction sent to Kafka",
        "transaction_id": transaction_id
    }


@app.get("/transaction/{transaction_id}")
def get_transaction_result(
    transaction_id: str
):

    if transaction_id in results:

        result_data = results[
            transaction_id
        ]

        return {
            "transaction_id": transaction_id,
            "result": result_data["result"],
            "fraudProbability": result_data[
                "fraudProbability"
            ]
        }

    return {
        "transaction_id": transaction_id,
        "result": "PENDING"
    }
