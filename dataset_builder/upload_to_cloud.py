from pathlib import Path
from google.cloud.storage import Client, transfer_manager
import os

def upload_folder_many_filenames(bucket_name, source_directory, gcs_destination):
    """
    Uploads a folder recursively using upload_many_from_filenames.

    Args:
        bucket_name (str): The ID of the GCS bucket.
        source_directory (str): The local directory path to upload.
        gcs_destination (str): The GCS prefix for the uploaded files.
    """
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    # Convert the source directory to a Path object
    source_path = Path(source_directory)

    # Recursively find all files in the source directory
    file_paths = [path for path in source_path.rglob("*") if path.is_file()]

    # Extract filenames relative to the source directory
    relative_filenames = [str(path.relative_to(source_path)) for path in file_paths]

    # Upload the files using the Transfer Manager
    print(f"Uploading files from '{source_directory}' to 'gs://{bucket_name}/{gcs_destination}'...")
    results = transfer_manager.upload_many_from_filenames(
        bucket=bucket,
        filenames=relative_filenames,
        source_directory=source_directory,
        blob_name_prefix=gcs_destination,
        max_workers=8
    )

    # Check the results and report any failures
    for file_path, result in zip(relative_filenames, results):
        if isinstance(result, Exception):
            print(f"Failed to upload '{file_path}': {result}")
        else:
            print(f"Uploaded '{file_path}' to 'gs://{bucket_name}/{gcs_destination}{file_path}'.")

def create_gcs_json(local_folder, gcs_bucket, gcs_prefix, output_file):
    """
    Creates a JSONL file with GCS URIs for images in a local folder.

    Args:
        local_folder (str): The local directory containing images.
        gcs_bucket (str): The GCS bucket name.
        gcs_prefix (str): The GCS prefix where images will be stored.
        output_file (str): The path to the output JSONL file.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    local_path = Path(local_folder)
    #prompt = "Describe this image in detail."
    prompt = 'Describe this location for visual place recognition. Focus on: 1) Scene type and setting, 2) Distinctive landmarks and architecture, 3) Unique visual patterns/colors/textures, 4) Spatial layout, 5) Key identifying features that distinguish this place from similar locations. Be specific about permanent visual elements, avoid temporary objects like people, car ,weather and lighting conditions, provide textual descriptions of items you are certain about only. the output is one line of text listing the items from left to right, separated by commas.'
    
    with open(output_file, 'w') as f:
        for image_path in local_path.rglob("*"):
            if image_path.suffix.lower() in image_extensions and image_path.is_file():
                folder = os.path.basename(str(image_path.parent))
                gcs_uri = f"gs://{gcs_bucket}/{gcs_prefix}{folder}/{image_path.name}"
                json_line = f'{{"request":{{"contents":[{{"role":"user","parts":[{{"file_data":{{"file_uri":"{gcs_uri}","mime_type":"image/jpeg"}}}},{{"text":"{prompt}"}}]}}]}}}}\n'
                f.write(json_line)
                #print(f"Added entry for '{image_path.name}' to '{output_file}'.")

def preduction_to_csv(pred_json):
    import json
    import pandas as pd

    # Load the JSONL file into a list of dictionaries
    data = []
    with open(pred_json, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Extract relevant fields and flatten the structure
    records = []
    for item in data:
        try:
            description = item['response']['candidates'][0]['content']['parts'][0]['text']
            image_uri = item['request']['contents'][0]['parts'][0]['file_data']['file_uri'].replace('gs://ofer-idan-bucket/sf-xl/','')
            records.append({'image_path': image_uri, 'description': description})
        except (KeyError, IndexError) as e:
            print(f"Skipping an entry due to missing fields: {e}")

    # Create a DataFrame and save it as CSV
    df = pd.DataFrame(records)
    output_csv = pred_json.replace('.jsonl', '.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to '{output_csv}'.")

# --- Example usage ---
if __name__ == "__main__":
    LOCAL_FOLDER = "/mnt/d/data/sf_xl/small/test/queries_v1/"
    BUCKET_NAME = "ofer-idan-bucket"
    GCS_DESTINATION = "sf-xl/test/queries_v1/"    
    
    #upload_folder_many_filenames(BUCKET_NAME, LOCAL_FOLDER, GCS_DESTINATION)
    
    gcs_file = "input_gcs.jsonl"
    #create_gcs_json(LOCAL_FOLDER, BUCKET_NAME, GCS_DESTINATION, gcs_file)
    
    pred_json = '/mnt/d/ofer/localization/text2vpr/dataset_builder/sf_xl_small_test_queries_predictions.jsonl'
    preduction_to_csv(pred_json)

    


