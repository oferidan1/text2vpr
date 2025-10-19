import time
import glob
import csv
import argparse
from pathlib import Path
from google.genai import types    
from google import genai   
from PIL import Image

def run_gemini(image_path, api_key=None):

    client = genai.Client(api_key=api_key)

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


def read_image_list(list_file):
    """Read image paths from a text file (one path per line)."""
    image_paths = []
    
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                image_paths.append(line)
    
    return image_paths

def describe_images_from_list(image_list_file, csv_path="descriptions.csv", prefix="", api_key=None):
    """Generate descriptions for images listed in a text file."""
    print(f"📋 Reading image list from: {image_list_file}")
    image_paths = read_image_list(image_list_file)
    print(f"📸 Found {len(image_paths)} images to process")
    
    if not image_paths:
        print("❌ No images found in the list file")
        return
    
    # prompt 
    prompt = 'describe this location for visual place recognition. Focus on: 1) Scene type and setting, 2) Distinctive landmarks and architecture, 3) Unique visual patterns/colors/textures, 4) Spatial layout, 5) Key identifying features that distinguish this place from similar locations. Be specific about permanent visual elements, avoid temporary objects like people, car ,weather and lighting conditions, provide textual descriptions of items you are certain about only. the output is one line of text listing the items from left to right, separated by commas.'
    
    client = genai.Client(api_key=api_key)
    results = []
    
    for i, image_path in enumerate(image_paths):
        # Add prefix if provided
        full_path = prefix + image_path if prefix else image_path
        
        print(f"[{i+1}/{len(image_paths)}] Processing: {Path(full_path).name}")
        
        # Check if image exists
        if not Path(full_path).exists():
            print(f"❌ Image not found: {full_path}")
            results.append((image_path, "ERROR: Image not found"))
            continue
        
        try:
            with open(full_path, 'rb') as f:
                image_bytes = f.read()
                
            # incase of failure with gemini API, wait for 30 seconds and retry
            max_attempts = 10
            attempts = 0
            while attempts < max_attempts:
                try:            
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-lite',
                        contents=[
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type='image/jpeg',
                            ),
                            prompt,
                        ]
                    )    
                    
                    print(f"✅ {Path(full_path).name}: {response.text[:100]}...")
                    results.append((image_path, response.text))    
                    
                    # save results to a csv file 
                    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['image_path', 'description'])
                        writer.writerows(results)
                        csvfile.flush() 
                        
                    break  # exit the retry loop on success
                        
                except Exception as e:
                    attempts += 1
                    print(f"❌ Error processing image {full_path}: {e}")
                    if attempts < max_attempts:
                        print('⏳ Sleeping for 30 seconds before retrying...')
                        time.sleep(30)
                    else:
                        print(f"❌ Failed after {max_attempts} attempts, skipping...")
                        results.append((image_path, f"ERROR: Failed after {max_attempts} attempts"))
        
        except Exception as e:
            print(f"❌ Error reading image {full_path}: {e}")
            results.append((image_path, f"ERROR: {e}"))
    
    print(f"💾 Results saved to {csv_path}")
    return results

def describe_all_images(image_folder, api_key=None):
    # find all jpg file recursively using glob
    image_paths = glob.glob(f"{image_folder}/**/*.jpg", recursive=True)
    # prompt 
    prompt = 'describe this location for visual place recognition. Focus on: 1) Scene type and setting, 2) Distinctive landmarks and architecture, 3) Unique visual patterns/colors/textures, 4) Spatial layout, 5) Key identifying features that distinguish this place from similar locations. Be specific about permanent visual elements, avoid temporary objects like people, car ,weather and lighting conditions, provide textual descriptions of items you are certain about only. the output is one line of text listing the items from left to right, separated by commas.'
    # Prepare content for each image
    
    csv_path = "descriptions.csv"
    
    client = genai.Client(api_key=api_key)
    
    results = []
    for i, image_path in enumerate(image_paths):
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            
        # incase of failure with gemini API, wait for 30 seconds and retry
        # TBD : do we get an exception or other error? 
        
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            try:            
                response = client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg',
                        ),
                        prompt,
                    ]
                )    
                
                print(f"Response for image {i+1}:")
                print(response.text) # Assuming text-based responses
                results.append((image_path, response.text))    
                
                # save results to a csv file 
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['image_path', 'description'])
                    writer.writerows(results)
                    csvfile.flush() 
                    
                break  # exit the retry loop on success
                    
            except Exception as e:
                attempts += 1
                print(f"Error processing image {image_path}: {e}")
                print('sleeping for 30 seconds before retrying...')
                time.sleep(30)  # wait for a second before retrying or moving to the next image            
        
        
    print(f"Results saved to {csv_path}")
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate descriptions for images using Gemini')
    parser.add_argument('--folder', '-f', help='Process all JPG images in a folder (recursively)')
    parser.add_argument('--list', '-l', help='Process images from a text file list (one path per line)')
    parser.add_argument('--output', '-o', default='descriptions.csv', help='Output CSV file (default: descriptions.csv)')
    parser.add_argument('--prefix', '-p', default='', help='Prefix to add to image paths from list (optional)')
    parser.add_argument('--single', '-s', help='Process a single image file')
    parser.add_argument('--api-key', '-k', help='Google AI API key (or set GOOGLE_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    # Get API key from argument or environment variable
    api_key = args.api_key or None
    
    if args.single:
        print(f"🖼️  Processing single image: {args.single}")
        result = run_gemini(args.single, api_key)
        print(f"Description: {result}")
        
    elif args.list:
        print(f"📋 Processing images from list: {args.list}")
        describe_images_from_list(args.list, args.output, args.prefix, api_key)
        
    elif args.folder:
        print(f"📁 Processing all JPG images in folder: {args.folder}")
        describe_all_images(args.folder, api_key)
        
    else:
        # Default behavior - process the hardcoded folder
        print("📁 Using default folder (no arguments provided)")
        image_folder = '/mnt/d/data/sf_xl/small/dummy'
        #image_folder = '/mnt/d/data/sf_xl/small/test/queries_v1'
        describe_all_images(image_folder, api_key)