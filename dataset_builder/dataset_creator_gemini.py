import time
import glob
import csv
from google.genai import types    
from google import genai   
from PIL import Image

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
    
    csv_path = "descriptions.csv"
    
    client = genai.Client()
    
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
                    model='gemini-2.5-flash',
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
    #image_path = 'images/dizengoff.webp'
    image_path = 'images/@0543158.27@4180593.76@10@S@037.77166@-122.50995@M7mOh9X4Xw_OHp-DYe5hQg@@206@@@@201311@@.jpg'
    #run_intern_vl(image_path)
    run_gemini(image_path)  # Uncomment to run Gemini example
    
    image_folder = '/mnt/d/data/sf_xl/small/dummy'
    image_folder = '/mnt/d/data/sf_xl/small/test/queries_v1'
    #describe_all_images(image_folder)