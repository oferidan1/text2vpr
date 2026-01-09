import pandas as pd
import os
import numpy as np
from numpy import nan

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
    

def remove_topic_v2(model, tokenizer, new_paragraph, new_topic):
    # 1. Define the 1-shot example data
    example_topic = "tower"
    example_input = (
        "A multi-story brick building with a tiled gabled roof, dormer window, and "
        "rectangular grid windows, featuring the text 'DE COST GAET VOOR DE BAET UYT.' "
        "and 'HANDELSINRICHTINGEN' on its facade; a boat docked in a canal; a dark brick "
        "bridge structure with a railing; a tall, ornate streetlight; a light-colored "
        "building with classical architectural elements; a second tall, ornate streetlight; "
        "a large, dark brick building with a prominent, narrow, pointed-roof tower and tall windows."
    )
    example_output = (
        "A multi-story brick building with a tiled gabled roof, dormer window, and "
        "rectangular grid windows, featuring the text 'DE COST GAET VOOR DE BAET UYT.' "
        "and 'HANDELSINRICHTINGEN' on its facade; a boat docked in a canal; a dark brick "
        "bridge structure with a railing; a tall, ornate streetlight; a light-colored "
        "building with classical architectural elements; a second tall, ornate streetlight; "
        "a large, dark brick building."
    )

    # 2. Build the message list with the 1-shot example
    messages = [
        {
            "role": "system", 
            "content": "You are a precise text editor. Remove descriptions of the specified topic while preserving the rest of the text and punctuation exactly. Provide ONLY the edited text."
        },
        # THE 1-SHOT EXAMPLE
        {"role": "user", "content": f"Topic: {example_topic}\nParagraph: {example_input}"},
        {"role": "assistant", "content": example_output},
        
        # THE ACTUAL REQUEST
        {"role": "user", "content": f"Topic: {new_topic}\nParagraph: {new_paragraph}"}
    ]

    # 3. Execution
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False # Greedy decoding for maximum consistency
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def clean_texts_from_csv(csv_file, model_id):    

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
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
            new_description = remove_topic_v2(model, tokenizer, description, to_filter)
    
            results.append([image_path, new_description, description])
    
    # save results to updated 
    #csv_file_name = os.path.basename(csv_file)
    csv_file_name = 'amstertime_objects_cleaned.csv'
    df2 = pd.DataFrame(results, columns=['image_path', 'description', 'original_description'])
    df2.to_csv(csv_file_name, index=False)

    

if __name__ == "__main__":
    csv_file = 'amstertime_objects.csv'
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    model_id = "microsoft/Phi-4-mini-instruct"

    clean_texts_from_csv(csv_file, model_id)

        