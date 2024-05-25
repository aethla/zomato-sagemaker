import numpy as np
import torch, os, cv2
import tempfile
import boto3
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    env = os.environ
    model_path = os.path.join(model_dir, "code", "best.pt")
    model = YOLO(model_path)
    return model


def input_fn(request_body, request_content_type):
    print("Executing input_fn from inference.py ...")
    if request_content_type == 'image/jpeg':
        # Convert the bytes back to an image array
        image = np.frombuffer(request_body, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return img
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    
def predict_fn(input_data, model):
    print("Executing predict_fn from inference.py ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(input_data)
    return result
       
def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")
    for r in prediction_output:
        # Visualize the results
        for i, r in enumerate(prediction_output):
            # Plot results image
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)  # RGB-order numpy array
            im =Image.fromarray(im_rgb[..., ::-1])
            
    # Save the image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    im.save(temp_file.name, 'JPEG')
    temp_file.close()

    # Get S3 access key and secret access key from env
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = 'adas-project-bucket'
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    # Generate a timestamp to include in the S3 object path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    s3_path = f"results/result_{timestamp}.jpg"

    # Upload the temporary file to S3 bucket
    with open(temp_file.name, 'rb') as file:
        s3.upload_fileobj(file, bucket_name, s3_path)

    # Remove the temporary file
    os.remove(temp_file.name)

    # Return JSON object with the path to the uploaded image
    return {"message": "Image uploaded successfully to S3", "s3_path": s3_path}
