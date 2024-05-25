import os, boto3
import cv2
import numpy as np


ENDPOINT_NAME = "zomato-images-serverless-endpoint"
runtime= boto3.client("runtime.sagemaker")
# Get S3 access key and secret access key from env
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID_s3')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY_s3')
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)


def lambda_handler(event, context):
    print(event)
    body = event['body']
    input_key = body['inputKey']
    bucket_name = 'zomato-project-bucket'

    # Step 1: Get the image from S3
    fileObj = s3.get_object(Bucket=bucket_name, Key=input_key)
    file_content = fileObj['Body'].read()

    # Step 2: Convert the image bytes to a NumPy array
    nparr = np.frombuffer(file_content, np.uint8)
    
    # Step 3: Decode the NumPy array as an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Encode image as JPEG
    _, encoded_img = cv2.imencode('.jpg', img)
    payload = encoded_img.tobytes()
    
    response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            Body=payload,
            ContentType="image/jpeg"
        )

    # Process the response from the endpoint
    response_body = response['Body'].read()
    return response_body

