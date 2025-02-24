from google.cloud import pubsub_v1  # Install with: pip install google-cloud-pubsub
import glob
import json
import os
import base64

# Locate the JSON file for Google Cloud credentials
gcp_credentials = glob.glob("*.json")  
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials[0]  

# Define Google Cloud Pub/Sub parameters
project_id = "velvety-study-448822-h6"
topic_name = "Image"   

# Initialize Pub/Sub publisher with ordering enabled
publisher_options = pubsub_v1.types.PublisherOptions(enable_message_ordering=True)
publisher = pubsub_v1.PublisherClient(publisher_options=publisher_options)
topic_path = publisher.topic_path(project_id, topic_name)
print(f"Publishing messages with ordering keys to {topic_path}.")

# Define the path to image dataset
images_directory = "Dataset_Occluded_Pedestrian/"

# Retrieve image files starting with 'A'
image_files = glob.glob(os.path.join(images_directory, "A*.*"))

print(f"Publishing images to {topic_name}...")
for image_file in image_files:
    with open(image_file, "rb") as img:
        # Encode the image as base64
        encoded_image = base64.b64encode(img.read()).decode('utf-8')

        # Fix base64 padding
        encoded_image += "=" * ((4 - len(encoded_image) % 4) % 4)

    image_key = os.path.basename(image_file)  # Extract filename as key

    # Construct JSON payload
    payload = {
        "ID": image_key,
        "Image": encoded_image
    }

    # Convert message to JSON and encode it
    json_payload = json.dumps(payload).encode('utf-8')

    try:
        # Publish message to Pub/Sub
        future = publisher.publish(topic_path, json_payload, ordering_key=image_key)
        future.result()  # Ensure successful publishing
        print(f"Successfully published: {image_key}")
    except Exception as e:
        print(f"Error publishing {image_key}: {e}")

print("All images have been published.")
