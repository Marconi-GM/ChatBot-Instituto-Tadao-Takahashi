import json
import os

from services.agent import ITTAgent

agent = None

def handler(event, context):
    global agent

    # print(event)
    
    try:
        if agent is None:
            print("Initializing ITTAgent...")
            
            # Check for API key before initializing
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("ERROR: GOOGLE_API_KEY not found in environment variables")
                return {
                    "statusCode": 500,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Content-Type",
                        "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
                    },
                    "body": json.dumps({"error": "API key not configured"}, ensure_ascii=False)
                }
            
            agent = ITTAgent()
            print("ITTAgent initialized successfully")

        if "body" in event and event["body"]:
            body = json.loads(event["body"])

        message = ""

        if "query" in body and body["query"]:
            message = body["query"]

        if not message:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
                },
                "body": json.dumps({"error": "No message provided"})
            }

        print(f"Processing message: {message[:100]}...")  # Log first 100 chars
        
        response = agent.process_query({"message": message})
        
        print("Query processed successfully")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
            },
            "body": json.dumps({ **response }, ensure_ascii=False)
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
            },
            "body": json.dumps({"error": "Internal server error occurred"}, ensure_ascii=False)
        }
