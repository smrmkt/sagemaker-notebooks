import json

def lambda_handler(event, context):
    return {
        "taskInput": {
            "sourceRef" : event['dataObject']['source-ref']
        }
    }
