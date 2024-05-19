import json, os, io, boto3


ENDPOINT_NAME = ""
runtime= boto3.client("runtime.sagemaker")
# Get S3 access key and secret access key from env
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID_s3')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY_s3')
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)


def lambda_handler(event, context):
    print(event)
    body = event['body']
    s3Key = body['s3Key']
    name = body['name']
    bucket_name = 'zomato-project-bucket'
    image_file_key = f'{s3Key}'
    fileObj = s3.get_object(Bucket=bucket_name, Key=image_file_key)
    file_content = fileObj['Body'].read()
    response = runtime.invoke_endpoint(
            EndpointName = ENDPOINT_NAME,
            Body = file_content,
            ContentType = "application/json"
        )
    print(response)
    return {
        'statusCode': 200,
        'body': json.dumps(name)
    }
