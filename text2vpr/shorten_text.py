import pandas as pd
import os
import numpy as np
from numpy import nan

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

def clean_texts_from_csv_phi4(csv_file):    
    
    model_id = "microsoft/Phi-4-mini-instruct"

    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True,
        attn_implementation="flash_attention_2" # Optional: speeds up generation on supported GPUs
    )
    
    results = []
    # parse csv file
    df = pd.read_csv(csv_file)
    # go line by line and read columns
    for index, row in df.iterrows():
        image_path = row['image_path']
        description = row['description']    
        to_filter = row['manualy filter']
        if to_filter is nan:
            results.append([image_path, description])
        else:
            # call llm to remove to_filter text from description
            # 3. Format the Prompt
            messages = [
                {"role": "system", "content": "You are a precise editor. Your task is to rewrite paragraphs to remove specific topics while keeping the rest of the text natural and grammatically correct."},
                {"role": "user", "content": f"Rewrite the following paragraph to remove all mention of '{to_filter}':\n\n{description}"}
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 4. Generate the Response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=2000, 
                do_sample=False                
            )
            
            # Slice the output: [0, len(inputs.input_ids[0]):]
            # This ignores the first 'n' tokens (the prompt)
            generated_tokens = outputs[0][len(inputs.input_ids[0]):]

            # Decode only the new tokens
            new_description = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            results.append([image_path, new_description, description])
    
    # save results to updated 
    #csv_file_name = os.path.basename(csv_file)
    csv_file_name = 'amstertime_objects_cleaned.csv'
    df2 = pd.DataFrame(results, columns=['image_path', 'description', 'original_description'])
    df2.to_csv(csv_file_name, index=False)
    
    
def remove_topic_v2(model, tokenizer, messages):
   
    # 3. Execution
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=False # Greedy decoding for maximum consistency
    )

    #my output is batch     
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def remove_topic_v2_batch(model, tokenizer, messages_batch):
    """
    messages_batch: List of lists (e.g., [[{"role": "user", "content": "..."}], [...]])
    """
    # 1. Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Apply chat template to the batch
    # padding=True and return_dict=True are essential for batching
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        return_dict=True
    ).to(model.device)

    # 3. Execution
    outputs = model.generate(
        **inputs, # Unpack input_ids and attention_mask
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=False
    )

    # 4. Extract only the NEW tokens for the entire batch
    # We slice from the length of the input sequence
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_length:]

    # 5. Decode all responses in the batch
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def build_context(new_paragraph):
     # 1. Define the 1-shot example data
    example_input = (
        "A multi-story brick building with a tiled gabled roof, dormer window, and "
        "rectangular grid windows, featuring the text 'DE COST GAET VOOR DE BAET UYT.' "
        "and 'HANDELSINRICHTINGEN' on its facade; a boat docked in a canal; a dark brick "
        "bridge structure with a railing; a tall, ornate streetlight; a light-colored "
        "building with classical architectural elements; a second tall, ornate streetlight; "
        "a large, dark brick building with a prominent, narrow, pointed-roof tower and tall windows."
    )
    example_output = (
        #"brick building with 'DE COST GAET VOOR DE BAET UYT.' and 'HANDELSINRICHTINGEN' on facade, dormer, and grid windows. Canal boat docked alongside. Adjacent dark brick bridge with railing. Ornate streetlight stands between gabled structure and light classical building. Second streetlight positioned before a large, dark brick building."
        "Gabled 'DE COST GAET' facade left, canal boat below, brick bridge center, two ornate streetlights foreground, classical light building right."
    )

    # 2. Build the message list with the 1-shot example
    messages = [        
        {"role": "system", 
        "content": "You are an expert in Geospatial localization and Computer Vision. Your task is to compress long scene descriptions into highly discriminative Spatial Signatures for a text-to-image retrieval system. Condense this scene into a 20-word spatial signature, preserving unique landmarks, distinctive signs texts, and precise object-to-object positioning while removing all non-visual narrative fluff."
        },
        
        # THE 1-SHOT EXAMPLE
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": example_output},
        
        # THE ACTUAL REQUEST
        {"role": "user", "content": new_paragraph}
    ]
    return messages


def clean_texts_from_csv(csv_file_in, csv_file_out,  model_id):    
    results = []  
     # parse csv file
    df = pd.read_csv(csv_file_in)
    i = 0    
    # open csv_file_name and make short list of all files not in csv_file
    if os.path.exists(csv_file_out):
        df2 = pd.read_csv(csv_file_out)
        files_in_csv = df2['image_path'].tolist()
        description_in_csv = df2['description'].tolist()
        original_description_in_csv = df2['original_description'].tolist()
        results = list(zip(files_in_csv, description_in_csv, original_description_in_csv))        
        df = df[~df['image_path'].isin(files_in_csv)]
        
    # 1. Define FP8 Quantization Config
    # Note: Ensure you have bitsandbytes installed
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, # Standard 8-bit
        # For actual FP8 (hardware accelerated), usually requires pre-quantized weights 
        # or libraries like vLLM/AutoFP8.
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 2. Load model with Flash Attention and Quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, # Compute dtype should stay bfloat16
        device_map="auto",
        attn_implementation="flash_attention_2" # Enforce Flash Attention 2
    )
    
    batch_size = 100    
    batch_items = []
    batch_images = []
    batch_descriptions = []    
    
    # go line by line and read columns
    for index, row in df.iterrows():
        image_path = row['image_path']
        description = row['description']    

        # messages = [
        #     {"role": "system", "content": "You are an expert in Geospatial localization and Computer Vision. Your task is to compress long scene descriptions into highly discriminative Spatial Signatures for a text-to-image retrieval system."},
        #     {"role": "user", "content": f"Condense this scene into a 50-word spatial signature, preserving unique landmarks, distinctive signs texts, and precise object-to-object positioning while removing all non-visual narrative fluff.':\n\n{description}"}
        # ]
        
        messages = build_context(description)
        
        batch_items.append(messages)
        batch_images.append(image_path)
        batch_descriptions.append(description)
        if len(batch_items) == batch_size:
            new_descriptions = remove_topic_v2_batch(model, tokenizer, batch_items)
            for img, new_desc, old_desc in zip(batch_images, new_descriptions, batch_descriptions):
                results.append([img, new_desc.strip(), old_desc])
                batch_items = []     
                batch_images = []
                batch_descriptions = []          

            df2 = pd.DataFrame(results, columns=['image_path', 'description', 'original_description'])
            df2.to_csv(csv_file_out, index=False)
    
    if len(batch_items)>0:
        new_descriptions = remove_topic_v2_batch(model, tokenizer, batch_items)
        for img, new_desc, old_desc in zip(batch_images, new_descriptions, batch_descriptions):
            results.append([img, new_desc.strip(), old_desc])
            
    # save results to updated 
    #csv_file_name = os.path.basename(csv_file)    
    df2 = pd.DataFrame(results, columns=['image_path', 'description', 'original_description'])
    df2.to_csv(csv_file_out, index=False)

    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv_file", type=str, default="amstertime_descriptions.csv")
    parser.add_argument("--out_file", type=str, default="amstertime_short.csv")
    args = parser.parse_args()       
    
 
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    #model_id = "microsoft/Phi-4-mini-instruct"

    clean_texts_from_csv(args.csv_file, args.out_file, model_id)
        
        
       
