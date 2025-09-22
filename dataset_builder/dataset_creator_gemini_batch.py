import time
import glob
import csv
from google.genai import types    
from google import genai   
from PIL import Image
import json
import base64
from google.cloud import storage
from google.cloud import aiplatform

def run_gemini(image_path):

    client = genai.Client()    
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        
    prompt = 'describe all objects in this image from left to right in one line, including their attributes and colors, ignore dynamic objects like people and cars. in your response, use the format: object1, object2, object3, ...'
    prompt = 'describe from left to right all distinctive features in one line for visual place recognition'
    prompt = 'describe this location for visual place recognition. Focus on: 1) Scene type and setting, 2) Distinctive landmarks and architecture, 3) Unique visual patterns/colors/textures, 4) Spatial layout, 5) Key identifying features that distinguish this place from similar locations. Be specific about permanent visual elements, avoid temporary objects like people, car ,weather and lighting conditions, provide textual descriptions of items you are certain about only. the output is one line of text listing the items from left to right, separated by commas.'

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


def describe_all_images(image_folder):
    # find all jpg file recursively using glob
    image_paths = glob.glob(f"{image_folder}/**/*.jpg", recursive=True)
    # prompt 
    prompt = 'describe this location for visual place recognition. Focus on: 1) Scene type and setting, 2) Distinctive landmarks and architecture, 3) Unique visual patterns/colors/textures, 4) Spatial layout, 5) Key identifying features that distinguish this place from similar locations. Be specific about permanent visual elements, avoid temporary objects like people, car ,weather and lighting conditions. the output is one line of text listing the items from left to right, separated by commas.'    
    # Prepare content for each image
    
    client = genai.Client()
    
    aiplatform.init(project='gen-lang-client-0562403994', location='us-west1')
    
    start_time = time.time()
    
    # A list of dictionaries, where each is a GenerateContentRequest
    with open(image_paths[0], 'rb') as f:
        image_bytes_1 = f.read()
    with open(image_paths[1], 'rb') as f:
        image_bytes_2 = f.read()
        
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
    
    my_file_1 = client.files.upload(file=image_paths[0])
    my_file_2 = client.files.upload(file=image_paths[1])
    
    inline_requests = [
        {
            "contents": [
                    my_file_1,
                    {"text": prompt},
                ],
        },
        {
            "contents": [
                    my_file_2,
                    {"text": prompt},
                ],

        }
    ]
    
    batch_job = client.batches.create(
        model="models/gemini-2.5-flash",
        src=inline_requests,
        config={
            'display_name': "inlined-requests-job_1",
        },
    )

    print(f"Created batch job: {batch_job.name}")    
    
    
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
    
     #Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step
    job_name = batch_job.name  # (e.g. 'batches/your-batch-id')
    batch_job = client.batches.get(name=job_name)

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
        
    # Use the name of the job you want to check
# e.g., inline_batch_job.name from the previous step
    batch_job = client.batches.get(name=job_name)

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
    
    # results = []        
    # print(f"Response for image {i+1}:")
    # print(response.text) # Assuming text-based responses
    # results.append((image_path, response.text))    
    
    # save results to a csv file 
    # csv_path = "descriptions.csv"
    # with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['image_path', 'description'])
    #     writer.writerows(results)
    # print(f"Results saved to {csv_path}")
    

if __name__ == '__main__':
    #image_path = 'images/dizengoff.webp'
    image_path = 'images/@0543158.27@4180593.76@10@S@037.77166@-122.50995@M7mOh9X4Xw_OHp-DYe5hQg@@206@@@@201311@@.jpg'
    #run_intern_vl(image_path)
    #run_gemini(image_path)  # Uncomment to run Gemini example
    
    image_folder = '/mnt/d/data/sf_xl/small/dummy'
    describe_all_images(image_folder)
  