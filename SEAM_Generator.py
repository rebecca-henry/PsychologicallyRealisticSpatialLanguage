from datetime import datetime
import json
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
import os

# settings
# adjust for new batch 
batch_name = "batch_0_1"
reasoning_effort = "low" # can be "minimal", "low", "medium", or "high" - adjust in prompt if changing
batch_iter = f"{reasoning_effort}_iter01" #"lowreasoning_5" or "minreasoning" and "minreasoning_2"

# captions file name (to be read), prompt file name (to be saved to), seams file name (to be saved to)
input_captions = open(f"captions/SEAM_DB/{batch_name}_captions.txt").readlines()
input_captions_fname = f"captions/SEAM_DB/{batch_name}_captions.txt"
prompt_fname = f"prompts/SEAM_DB/V3/{batch_name}_{batch_iter}_prompts.jsonl"
SEAMs_fname = f"SEAMs/SEAM_DB/V3/{batch_name}_{batch_iter}_SEAMS.jsonl"

# shouldn't need to change ever
# historic log stores every single log, master only stores most recent update of each batch
master_log_fname = "records/SEAM_DB/master_log.jsonl"
historic_log_fname = "records/SEAM_DB/historic_log.jsonl"

# API settings
openai_api_key = open("settings/apikey.txt").read().strip()
client = OpenAI(api_key=openai_api_key)
language_description = open("language_description.txt").read()
batch_size = len(input_captions)
model = "gpt-5-nano"

# Define pydantic classes to enforce specific output structure
class Spatial_Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_element: str 
    target_element: str
    relation_desc: str
    reference_centre_point: str
    target_centre_point: str
    distance_value: float 
    distance_unit: Literal["mm", "cm", "m", "km"] 
    horizontal_direction_value: float = Field(ge=0, le=359) 
    vertical_direction_value: float = Field(ge=-90, le=90) 
    direction_unit: Literal["degrees"]

class Size(BaseModel):
    model_config = ConfigDict(extra="forbid")
    height: float
    height_unit: Literal["mm", "cm", "m", "km"]
    width: float
    width_unit: Literal["mm", "cm", "m", "km"]
    depth: float
    depth_unit: Literal["mm", "cm", "m", "km"]

class Reference_Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometry_alignment_desc: str
    reference_elements: list[str]
    location : list[Spatial_Reference]
    location_desc: str
        
class Geometry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    shape: str
    size: Size

class Boundary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    size: Size
    landmark: bool 
    surrounding_type: Literal["above", "below", "none"] 
    orientation_desc: str
    orientation_value: float = Field(ge=0, le=359)
    orientation_unit: Literal["degrees"]
    reference_elements: list[str]
    location : list[Spatial_Reference]
    location_desc: str
    
class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    size: Size
    shape: str
    landmark: bool 
    orientation_desc: str
    orientation_value: float = Field(ge=0, le=359)
    orientation_unit: Literal["degrees"]
    reference_elements: list[str]
    location : list[Spatial_Reference]
    location_desc: str
    
class ObjectItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name : str
    size : Size
    landmark: bool 
    orientation_desc: str
    orientation_value: float = Field(ge=0, le=359)
    orientation_unit: Literal["degrees"]
    reference_elements: list[str]
    location : list[Spatial_Reference]
    location_desc: str
    
class Surrounding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    geometric_desc: str
    surrounding_type: Literal["above", "below"]

class SEAM(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name : str
    geometry: Geometry
    reference_frame: Reference_Frame 
    boundaries: list[Boundary]
    segments: list[Segment]
    objects: list[ObjectItem]
    surroundings: list[Surrounding]

# Create the prompt file that holds the prompts in their properly formatted structure
def create_prompts():
    # convert pydantic class to JSON schema so it can be used in batch request
    seam_schema = SEAM.model_json_schema()
    requests = []
    # create batch file (jsonl) to send to API
    n = 0
    for cap in input_captions: 
        n = n + 1
        custom_id = str(batch_name) + "_" + str("V1") + "_" + str(n)
        req = {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", 
                        "body": {
                            "model": model, 
                            "reasoning": {"effort": reasoning_effort},
                            "input": [{"role": "system", "content": language_description},
                                {"role": "user", "content": "Scene: " + cap}],
                            "text": { 
                                "format": { 
                                "name": "seam_schema",
                                "type": "json_schema",
                                "schema" : seam_schema,                                             
                                } }}}
        requests.append(req)
    with open(prompt_fname, "w") as batch_prompts:
        for req in requests:
            batch_prompts.write(json.dumps(req) + "\n")
                            
    print(f"Prompts created in {prompt_fname}")
    
    # Check file size before uploading
    file_size_bytes = os.path.getsize(prompt_fname)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    print(f"Batch prompt file size: {file_size_mb:.2f} MB")
    return file_size_mb

# Make the batch request by uploading the prompt file
def create_batch(file_size_mb):
    # upload the prompt file to the client
    prompt_file = client.files.create(
        file=open(prompt_fname, "rb"),
        purpose="batch")

    # create the batch with the prompt file
    batch = client.batches.create(
        input_file_id = prompt_file.id,
        endpoint="/v1/responses",
        completion_window="24h",)

    print("batch created, batch id: ", batch.id)

    # log the batch creation in master_toc and historic_toc
    with open(master_log_fname, "a") as f:
        f.write(json.dumps({"batch_id": batch.id, "metadata": {"status": "Generating", "batch_name": batch_name, "caption_file": input_captions_fname, "prompt_file": prompt_fname, "filename": f"{batch_name}_{batch_iter}", "file_id" : None, "batch_size": batch_size, "size_mb": str(file_size_mb) + "MB","date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}) + "\n")
    with open(historic_log_fname, "a") as f:
        f.write(json.dumps({"batch_id": batch.id, "metadata": {"status": "Generating", "batch_name": batch_name, "caption_file": input_captions_fname, "prompt_file": prompt_fname, "filename": f"{batch_name}_{batch_iter}","file_id" : None,  "batch_size": batch_size, "size_mb": str(file_size_mb) + "MB", "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}) + "\n")

# retrieve and print error content
def retrieve_error(file_id):
    error_content = client.files.retrieve_content(file_id)
    print(error_content)

# to read and write jsonl files
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]
    
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# save the batch results when completed
def save_batch(batch_id):
    # use the batch id to check status of batch
    batch = client.batches.retrieve(batch_id)

    # Read the original log from master_toc to maintain the filename and get paths
    data = read_jsonl(master_log_fname)
    original_record = next(
        (l for l in data if l["batch_id"] == batch_id and l["metadata"]["status"] == "Generating"),
        None
    )
    original_filename = original_record["metadata"]["filename"] if original_record else f"{batch_name}_{batch_iter}"
    original_caption_file = original_record["metadata"]["caption_file"] if original_record else input_captions_fname
    original_prompt_file = original_record["metadata"]["prompt_file"] if original_record else prompt_fname
    original_batch_name = original_record["metadata"]["batch_name"] if original_record else batch_name
    original_batch_size = original_record["metadata"]["batch_size"] if original_record else batch_size
    original_size_mb = original_record["metadata"]["size_mb"] if original_record else None
    seams_path = f"SEAMs/SEAM_DB/V3/{original_filename}_SEAMS.jsonl"

    # if completed, retrieve the result
    if batch.status == "completed":
        
        # if no output file id, must be an error
        if batch.output_file_id is None:
            print(f"No output file ID found. Must be an error, see error_file_id: {batch.error_file_id}.")
            with open(historic_log_fname, "a") as f:
                f.write(json.dumps({"batch_id": batch.id, "metadata": {"status": "Failure", "batch_name": original_batch_name, "caption_file": original_caption_file, "prompt_file": original_prompt_file, "filename": original_filename, "file_id": batch.error_file_id, "batch_size": original_batch_size, "size_mb": original_size_mb, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}) + "\n")

            for line in data:
                if line["batch_id"] == batch_id and line["metadata"]["status"] == "Generating":
                    line["metadata"]["status"] = "Failure"
                    line["metadata"]["error_file_id"] = str(batch.error_file_id)
                    line["metadata"]["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
            write_jsonl(master_log_fname, data)
            retrieve_error(batch.error_file_id)
            return
        
        # otherwise, retrieve the output file
        file_id = batch.output_file_id
        response = client.files.content(file_id).text
        
    # store the responses into jsonl file:
        # each line is a separate json object
        response_lines = [
            json.loads(line)
            for line in response.splitlines()
            if line.strip()]

        # process and save cleanly
        for item in response_lines:
            body = item.get("response", {}).get("body", {})
            output_items = body.get("output", [])

            # 1) extract the model's output_text.text
            text = None
            for out in output_items:
                if isinstance(out, dict) and out.get("type") == "message":
                    for c in out.get("content", []) or []:
                        if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                            text = c.get("text")
                            break
                if text is not None:
                    break

            # 2) put the model's JSON text into a dict
            seam = json.loads(text)

            # 3) write clean JSONL
            with open(seams_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "custom_id": item.get("custom_id"),
                    "batch_id": batch_id,
                    "batch_name": original_batch_name,
                    "SEAM_title": seam["name"],
                    "scene_description": seam
                }, ensure_ascii=False) + "\n")

        # 4) log success
        with open(historic_log_fname, "a") as f:
            f.write(json.dumps({"batch_id": batch.id, "metadata": {"status": "Success", "batch_name": original_batch_name, "caption_file": original_caption_file, "prompt_file": original_prompt_file, "filename": original_filename, "file_id": file_id, "batch_size": original_batch_size, "size_mb": original_size_mb, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SEAMs": seams_path}}) + "\n")

        for line in data:
            if line["batch_id"] == batch_id and line["metadata"]["status"] == "Generating":
                line["metadata"]["status"] = "Success"
                line["metadata"]["file_id"] = str(file_id)
                line["metadata"]["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line["metadata"]["SEAMs"] = seams_path
                break
        write_jsonl(master_log_fname, data)
        
        print("Batch completed and results saved successfully.")

    # if failed to make batch
    elif batch.status == "failed":
        # log failure
        with open(historic_log_fname, "a") as f:
                f.write(json.dumps({"batch_id": batch.id, "metadata": {"status": "Failure", "batch_name": original_batch_name, "caption_file": original_caption_file, "prompt_file": original_prompt_file, "filename": original_filename, "file_id": batch.error_file_id, "batch_size": original_batch_size, "size_mb": original_size_mb, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}) + "\n")
        for line in data:
            if line["batch_id"] == batch_id and line["metadata"]["status"] == "Generating":
                line["metadata"]["status"] = "Failure"
                line["metadata"]["error_file_id"] = str(batch.error_file_id)
                line["metadata"]["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        write_jsonl(master_log_fname, data)
        
        # print error info
        if batch.error_file_id is None:
            print("Batch failed but no error_file_id found.")
            print("Here's the batch response: ",batch, "\n")
        else:
            print(f"Batch failed. error_file_id: {batch.error_file_id}.")
            retrieve_error(batch.error_file_id)
    
    elif batch.status == "expired":

        with open(historic_log_fname, "a") as f:
                f.write(json.dumps({"batch_id": batch.id, "metadata": {"status": "Expired", "batch_name": original_batch_name, "caption_file": original_caption_file, "prompt_file": original_prompt_file, "filename": original_filename, "file_id": batch.error_file_id, "batch_size": original_batch_size, "size_mb": original_size_mb, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}) + "\n")
        for line in data:
            if line["batch_id"] == batch_id and line["metadata"]["status"] == "Generating":
                line["metadata"]["status"] = "Expired"
                line["metadata"]["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        write_jsonl(master_log_fname, data)

    # print status unless completed or failed
    else:
        print(f"\nBatch {batch.id} Status: {batch.status}")


# check if batch for prompt file exists, if not create it
def check_create_batch():
    make_new = True
    with open(master_log_fname) as f:
        data = [json.loads(line) for line in f]
        for line in data:
            if line["metadata"]["prompt_file"] == prompt_fname and line["metadata"]["status"] == "Success":
                print("\nThis batch has already been succesfully created! Check master toc for more information.")
                make_new = False
            elif line["metadata"]["prompt_file"] == prompt_fname and line["metadata"]["status"] == "Generating":
                print("\nThis batch is already in progress. Check master toc for more information.\n")
                batch_id = line["batch_id"]
                make_new = False
                return make_new
            
            elif line["metadata"]["prompt_file"] == prompt_fname and line["metadata"]["status"] == "Failure":
                with open(prompt_fname, "w") as batch_prompts:
                    batch_prompts.truncate(0)
                make_new = True 
            elif line["metadata"]["prompt_file"] == prompt_fname and line["metadata"]["status"] == "Expired":
                with open(prompt_fname, "w") as batch_prompts:
                    batch_prompts.truncate(0)
                make_new = True 
        if make_new == True:
            print("This batch has not been run before or has previously failed or expired. Creating prompts and batch. Run program again to check status.")
        return make_new

# get list of pending batches
def get_pending_batches():
    batch_ids = []
    with open(master_log_fname) as f:
        data = [json.loads(line) for line in f]
        for line in data:
            if line["metadata"]["status"] == "Generating":
                batch_id = line["batch_id"]
                batch_ids.append(batch_id)
                print(f"\nBatch {batch_id} request counts: {client.batches.retrieve(batch_id).request_counts}")
    return batch_ids

# Main execution
if check_create_batch() == True:
    file_size_mb = create_prompts()
    if file_size_mb > 199:
        print("Batch file size exceeds 199MB, cannot create batch.")
    else:
        print("batch file size: ", file_size_mb, "MB")
        create_batch(file_size_mb)
        print("New batch created")

batch_ids = get_pending_batches()

for batch_id in batch_ids:
    save_batch(batch_id)
