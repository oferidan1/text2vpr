import time
import glob
import csv
from google.genai import types    
from google import genai   
from PIL import Image
import json
import base64
from google.cloud import storage
#from google.cloud import aiplatform
import concurrent.futures
import argparse
import sys

def run_gemini(image_path):

    client = genai.Client()    
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        
    # prompt = 'describe all objects in this image from left to right in one line, including their attributes and colors, ignore dynamic objects like people and cars. in your response, use the format: object1, object2, object3, ...'
    # prompt = 'describe from left to right all distinctive features in one line for visual place recognition'
    prompt = 'Describe this location for visual place recognition. Focus on: 1) Scene type and setting, 2) Distinctive landmarks and architecture, 3) Unique visual patterns/colors/textures, 4) Spatial layout, 5) Key identifying features that distinguish this place from similar locations. Be specific about permanent visual elements, avoid temporary objects like people, car ,weather and lighting conditions, provide textual descriptions of items you are certain about only. the output is one line of text listing the items from left to right, separated by commas.'

    t1 = time.time()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ),
        prompt,
        ]
    )
    t2 = time.time()
    print(f'Inference time: {t2 - t1:.2f} seconds')

    print(response.text)
    
    return response.text

def upload_file(client, file_path):
    """A helper function to upload a single file."""
    try:
        uploaded_file = client.files.upload(file=file_path)
        print(f"Uploaded {file_path}. ID: {uploaded_file.name}")
        return uploaded_file, file_path
    except Exception as e:
        print(f"Failed to upload {file_path}: {e}")
        return None

def upload_all_files(client, file_paths):
    """Uploads a list of files concurrently using a thread pool."""
    file_ids = []
    # Use max_workers to control the number of concurrent uploads
    #upload files concurrently, return list of (file_id, file_path) tuples    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
    #     futures = [executor.submit(upload_file, client, path) for path in file_paths]
    #     for future in concurrent.futures.as_completed(futures):
    #         file_id, file_path = future.result()
    #         if file_id:
    #             file_ids.append((file_id, file_path))                
                
    for file_path in file_paths:
        file_id, file_path = upload_file(client, file_path)
        if file_id:
            file_ids.append((file_id, file_path))
    return file_ids


def describe_all_images(args):
    # find all jpg file recursively using glob
    image_paths = glob.glob(f"{args.image_folder}/**/*.jpg", recursive=True)
    # prompt 
    prompt = 'Describe this location for visual place recognition. Focus on: 1) Scene type and setting, 2) Distinctive landmarks and architecture, 3) Unique visual patterns/colors/textures, 4) Spatial layout, 5) Key identifying features that distinguish this place from similar locations. Be specific about permanent visual elements, avoid temporary objects like people, car ,weather and lighting conditions, provide textual descriptions of items you are certain about only. the output is one line of text listing the items from left to right, separated by commas.'
    # Prepare content for each image
    
    client = genai.Client()
    
    #aiplatform.init(project='gen-lang-client-0562403994', location='us-west1')
    
    start_time = time.time()
    
    #gemini-2.5-flash have quata limit of 10000 requests per day
    start_image = args.start_image
    end_image = start_image + args.max_image
    image_paths = image_paths[start_image:end_image]
    print("Limiting to first 10000 images due to Gemini API quota limits.")    
    
    file_ids_path = upload_all_files(client, image_paths)
    
    # Upload images to File API
    inline_requests = [] 
    #for image in image_paths:                
    for my_file, image_path in file_ids_path:
        #my_file = client.files.upload(file=image)
        #my_files.append(my_file)
        inline_requests.append({
            "contents": [
                    my_file,
                    {"text": prompt},
                ],
        })    
        
    print(f"Uploaded {len(inline_requests)} files for batch processing.")
    list_size = sys.getsizeof(inline_requests)
    print(f"Size of the list object: {list_size} bytes")
    
    # try/except to create batch job till success in a loop 
    max_attempts = 10
    attempts = 0
    while attempts < max_attempts:
        try:            
            batch_job = client.batches.create(
                model="models/gemini-2.5-flash",
                src=inline_requests,
                config={
                    'display_name': args.job_name,
                },
            )
            print(f"Created batch job: {batch_job.name}")    
            break  # Exit the loop if successful
        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts} failed: {e}")
            time.sleep(60)  # Wait before retrying    
    
     #Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step
    job_name = batch_job.name  # (e.g. 'batches/your-batch-id')    
    
    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])

    print(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name) # Initial get
    while batch_job.state.name not in completed_states:
        print(f"Current state: {batch_job.state.name}")
        time.sleep(30) # Wait for 30 seconds before polling again
        batch_job = client.batches.get(name=job_name)

    print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job.error}")

    results = []            
    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a file
            result_file_name = batch_job.dest.file_name
            print(f"Results are in file: {result_file_name}")

            print("Downloading result file content...")
            file_content = client.files.download(file=result_file_name)
            # Process file_content (bytes) as needed
            print(file_content.decode('utf-8'))

        # If batch job was created with inline request
        elif batch_job.dest and batch_job.dest.inlined_responses:
            # Results are inline            
            print("Results are inline:")
            for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                print(f"Response {i+1}:")
                if inline_response.response:
                    # Accessing response, structure may vary.
                    try:
                        print(inline_response.response.text)
                        my_file, image_path = file_ids_path[i]                        
                        results.append((image_path, inline_response.response.text))    
                    except AttributeError:
                        print(inline_response.response) # Fallback
                elif inline_response.error:
                    print(f"Error: {inline_response.error}")
        else:
            print("No results found (neither file nor inline).")
    else:
        print(f"Job did not succeed. Final state: {batch_job.state.name}")
        if batch_job.error:
            print(f"Error: {batch_job.error}")
            
    end_time = time.time()
    print(f"Total batch processing time: {end_time - start_time:.2f} seconds")
    
    ## save results to a csv file 
    csv_path = args.result
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'description'])
        writer.writerows(results)
    print(f"Results saved to {csv_path}")
  
# inline_requests = [
    #     {
    #         "contents": [
    #                 types.Part.from_bytes(data=image_bytes_1, mime_type='image/jpeg'),
    #                 {"text": prompt},
    #             ],
    #     },
    #     {
    #         "contents": [
    #                 types.Part.from_bytes(data=image_bytes_2, mime_type='image/jpeg'),
    #                 {"text": prompt},
    #             ],

    #     }
    # ]
    
# with open("my-batch-requests.jsonl", "w") as f:
    #     requests = [
    #         {
    #             "key": "request-1",
    #             "request": {"contents": [{"parts": [{"fileData": {"fileUri": "gs://bucket-vpr/dummy/file1.jpg", "mimeType": "image/jpeg"}}, {"text": prompt}]}]}
    #         },
    #         {
    #             "key": "request-2",
    #             "request": {"contents": [{"parts": [{"fileData": {"fileUri": "gs://bucket-vpr/dummy/file2.jpg", "mimeType": "image/jpeg"}}, {"text": prompt}]}]}                
    #         }
            
    #     ]
    
    # ## Create a sample JSONL file
    # # with open("my-batch-requests.jsonl", "w") as f:
    # #     requests = [
    # #         {"key": "request-1", "request": {"contents": [ types.Part.from_bytes(data=image_bytes_1, mime_type='image/jpeg'), {"text": prompt}]}},
    # #         {"key": "request-2", "request": {"contents": [ types.Part.from_bytes(data=image_bytes_2, mime_type='image/jpeg'), {"text": prompt}]}},
    # #     ]
    #     for req in requests:
    #         f.write(json.dumps(req) + "\n")

    # # Upload the file to the File API
    # uploaded_file = client.files.upload(
    #     file='my-batch-requests.jsonl',
    #     config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
    # )

    # print(f"Uploaded file: {uploaded_file.name}")
    
    # # Assumes `uploaded_file` is the file object from the previous step
    # batch_job = client.batches.create(
    #     model="gemini-2.5-flash",
    #     src=uploaded_file.name,
    #     config={
    #         'display_name': "file-upload-job-1",
    #     },
    # )

    # print(f"Created batch job: {batch_job.name}")      

def delete_single_file(client, file_name):
    """A helper function to delete a single file."""
    try:
        client.files.delete(name=file_name)
        print(f"Successfully deleted: {file_name}")
    except genai.errors.ClientError as e:
        print(f"Failed to delete {file_name}: {e}")

def list_and_delete_all_files():
    from concurrent.futures import ThreadPoolExecutor

    client = genai.Client()
    """Lists all files for the current project and deletes them concurrently."""
    print("Fetching list of all uploaded files...")
    try:
        # Get a list of all files using client.files.list()
        all_files = list(client.files.list())
        
        if not all_files:
            print("No files found to delete.")
            return

        print(f"Found {len(all_files)} files. Starting deletion...")

        file_names = [f.name for f in all_files]

        # Use a thread pool to delete files concurrently
        with ThreadPoolExecutor(max_workers=100) as executor:
            executor.map(delete_single_file, client, file_names)

        print("Deletion process for all files initiated.")

    except genai.errors.ClientError as e:
        print(f"An error occurred while trying to list files: {e}")



def tmp():
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel, Part
    import google.cloud.aiplatform as aiplatform

    PROJECT_ID = "gen-lang-client-0562403994"  # Replace with your project ID
    LOCATION = "europe-west1"           # Replace with your desired region
    MODEL_NAME = "gemini-2.5-flash"
    
    # Define GCS input and output URIs
    GCS_INPUT_URI = "gs://ofer-idan-bucket/input.jsonl"
    GCS_OUTPUT_URI = "gs://ofer-idan-bucket/output/"     

    # Define GCS image URIs
    gcs_image_1 = "gs://ofer-idan-bucket/dummy/@0543037.38@4180974.80@10@S@037.77510@-122.51130@1ooZhwTQyuNt0xiue5YhLg@@134@@@@201311@@.jpg"
    gcs_image_2 = "gs://ofer-idan-bucket/dummy/@0543050.42@4181850.28@10@S@037.78299@-122.51110@OA95urczPlYG44VTBM9i9A@@0@@@@200802@@.jpg"

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Define the model ID. For the base Gemini model, this is the publisher name.
    # Use aiplatform.get_publisher_model to get the full model resource name.
    MODEL_RESOURCE_NAME = "publishers/google/models/gemini-2.5-flash"

    # Create and start the BatchPredictionJob
    job = aiplatform.BatchPredictionJob.create(
        job_display_name="gemini_batch_prediction_job",
        model_name=MODEL_RESOURCE_NAME,
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=GCS_INPUT_URI,
        gcs_destination_prefix=GCS_OUTPUT_URI,
    )   
    
    print(f"Batch prediction job created: {job.display_name}")
    print(f"Waiting for job to complete. State: {job.state}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_folder", default='/mnt/d/data/sf_xl/small/train', help="Path to the folder containing images")
    parser.add_argument("--job_name", default='requests-job-1', help="Name of the batch job to check status")
    parser.add_argument("--result", default='descriptions.csv', help="Path to the result CSV file")        
    parser.add_argument("--start_image", type=int, default='0', help="start image index")        
    parser.add_argument("--max_image", type=int, default='10000', help="maximum number of images to process")        
    args = parser.parse_args()        

    #describe_all_images(args)
    tmp()
  
    #list_and_delete_all_files()