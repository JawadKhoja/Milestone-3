from google.cloud import pubsub_v1      # pip install google-cloud-pubsub  ##to install
import glob                             # for searching for json file 
import os 
import base64

# Locate the JSON file for Google Cloud credentials
gcp_credentials = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials[0]

# Define Google Cloud Pub/Sub parameters
project_id = "velvety-study-448822-h6"
topic_name = "Image"   # Change this for your topic name if needed
subscription_id = "Image-sub"   # Change this for your subscription name if needed

# Initialize Pub/Sub subscriber
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)
topic_path = f'projects/{project_id}/topics/{topic_name}'

print(f"Listening for messages on {subscription_path}..\n")

# Callback function to process received messages
def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    try:
        # Decode the base64 image to verify validity
        image_bytes = base64.b64decode(message.data)

        # Retrieve image name from attributes
        image_name = message.attributes.get('image_key', 'Unknown')

        # Log the image name
        print(f"Consumed image: {image_name}")
        
        # Acknowledge successful processing
        message.ack()
    except Exception as e:
        print(f"Error processing message: {e}")
        # NACK the message to allow reprocessing later
        message.nack()

with subscriber:
    # Subscribe to messages using the callback function
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
