import json

def lambda_handler(event, context):
    return {
        "taskInput": {
            "topic": "日常会話",
            "conversation" : event["dataObject"]["source"]
        }
    }
