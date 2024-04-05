import os
import json
import random

def get_records(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        record_count = len(data)
        print("Number of records:", record_count)
    return data

def average_words(data):
    total_words = 0; record_count = 0
    for record in data:
        if record['body']:
            words = record['body'].split()
            total_words += len(words)
            record_count += 1
    average = total_words / record_count if record_count > 0 else 0
    return average

# Function to clean the text
def clean_text(text):
    if text is None:
        return ''
    
    text = text.replace('\n', ' ')
    text = text.replace('--------- SPLIT ELEMENT ---------', '')

    #return text.split("--------- SPLIT ELEMENT ---------", 1)[0]
    return text

def random_sample(data, sample_size=10000, seed=0):
    random.seed(seed)
    # Select a random sample from the full dataset
    full_sample = random.sample(data, sample_size)
    # For each item in the sample, create a new dictionary with only "id" and "body"
    reduced_sample = [
        {
            "id": item["id"], 
            "title": clean_text(item.get('title', '')), 
            "subtitle": clean_text(item.get('subtitle', '')), 
            "body": clean_text(item["body"])
        }
        for item in full_sample 
        if item["id"] and item["body"] and clean_text(item["body"]).strip() 
        and not clean_text(item["body"]).strip()[-1].isalnum()
    ]
    return reduced_sample

def random_word_count_sample(data, sample_size=10000, seed=0, target_words=1500):
    reduced_sample = []
    sample_data = random_sample(data, sample_size=sample_size, seed=seed)
    word_count = 0
    for item in sample_data:
        item_word_count = len(item['body'].split())
        if item_word_count > target_words:
            continue  # Skip this item and move on to the next one
        if word_count + item_word_count <= target_words:
            reduced_sample.append(item)
            word_count += item_word_count
        else:
            break
    print(f"The length of the reduced sample is: {len(reduced_sample)}")
    return reduced_sample

def write_sample_to_file(sample_data, output_path):
    with open(output_path, 'w', encoding='utf8') as json_file:
        json.dump(sample_data, json_file, ensure_ascii=False)

# Count words in txt-files
def count_words_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        words = text.split()
        print(f'Number of words in {file_path}: {len(words)}')
        return len(words)

# Write to brat (series of txt-files)
def convert_to_brat_format(json_file, output_dir):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    total_word_count = 0
    for item in data:
        id = item['id']
        text = f"{item.get('title', '')}\n{item.get('subtitle', '')}\n{item['body']}"
        # Save the text to a .txt file
        txt_file_name = os.path.join(output_dir, f'{id}.txt')
        with open(txt_file_name, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

        words_in_file = count_words_in_file(txt_file_name)
        total_word_count += words_in_file

        # Create an empty .ann file with the same base name
        ann_file_name = os.path.join(output_dir, f'{id}.ann')
        open(ann_file_name, 'a').close()  # Using 'a' (append) mode will not overwrite any existing data
    
    print(f'Total word count in all files: {total_word_count}')

data = get_records('articles_with_front_headlines.json')
average = average_words(data)
#test_sample = random_sample(data, sample_size=50)
#test_sample = random_word_count_sample(data, sample_size=10000, target_words=1500)
#convert_to_brat_format('test_sample.json', 'brat_data/test_collection1')
#test_sample2 = random_word_count_sample(data, sample_size=10000, target_words=900, seed=42)
# Write the training and test samples to files
#write_sample_to_file(test_sample2, 'test_sample2.json')
#convert_to_brat_format('test_sample2.json', 'brat_data/unannotated/test_collection2')
#agr_sample = random_word_count_sample(data, sample_size=10000, target_words=5000, seed=1)
# Write the agreement samples to files
#write_sample_to_file(agr_sample, 'agr_sample.json')
#convert_to_brat_format('agr_sample.json', 'brat_data/unannotated/agr_collection')

samuel_main = random_word_count_sample(data, sample_size=10000, target_words=20000, seed=2)
stephanie_main = random_word_count_sample(data, sample_size=10000, target_words=20000, seed=3)

def check_duplicates(dir1, dir2):
    """This function checks if any duplicates exist between files in two directories.
    """
    if os.path.exists(dir1) and os.path.exists(dir2):
        # Create a list of file IDs (names minus extensions for all txt-files) for both directories
        dir1_files = [os.path.splitext(f)[0] for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
        dir2_files = [os.path.splitext(f)[0] for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]
    
    # For each item in dir1, check if there is a corresponding duplicate in dir2
    duplicates = [f for f in dir1_files if f in dir2_files]
    # If duplicates exist, print them, otherwise print nothing
    if len(duplicates) > 0:
        print(f'The following files are duplicates in {dir1} and {dir2}:')
        for f in duplicates:
            print(f)
    else:
        print(f'No duplicates found in {dir1} and {dir2}')
    return

# data = get_records('articles_with_front_headlines.json')
average = average_words(data)
data_dir = "brat_data"

def create_sample_files(data, sample_name, target_words, sample_size, seed, output_dir):
    # General function to get a sample based on word counts, write to json file, and convert to brat format
    
    # Get sample data
    sample_data = random_word_count_sample(data, target_words=target_words, sample_size=sample_size, seed=seed)
    # Write sample data to json-file
    write_sample_to_file(sample_data, os.path.join(output_dir, f'{sample_name}.json'))
    # Convert sample data to brat format
    convert_to_brat_format(os.path.join(output_dir, f'{sample_name}.json'), os.path.join(output_dir, f'unannotated/{sample_name}'))

#test_sample = random_word_count_sample(data, sample_size=10000, target_words=1500)
#convert_to_brat_format('test_sample.json', 'brat_data/test_collection1')
#test_sample2 = random_word_count_sample(data, sample_size=10000, target_words=900, seed=42)
# Write the training and test samples to files
#write_sample_to_file(test_sample2, 'test_sample2.json')
#convert_to_brat_format('test_sample2.json', 'brat_data/unannotated/test_collection2')
#agr_sample = random_word_count_sample(data, sample_size=10000, target_words=5000, seed=1)
# Write the agreement samples to files
#write_sample_to_file(agr_sample, 'agr_sample.json')
#convert_to_brat_format('agr_sample.json', 'brat_data/unannotated/agr_collection')

# Function call to create sample Samuel_main, target_words=20000, seed=2, sample_size=10000, output_dear="brat_data"
create_sample_files(data, 'Samuel_main', 20000, 10000, 2, data_dir)
# Create Stephanie_main
create_sample_files(data, "Stephanie_main", 20000, 10000, 3, data_dir)

# Check duplicates between Samuel_main and Stephanie_main
check_duplicates(os.path.join(data_dir, 'unannotated/Samuel_main'), os.path.join(data_dir, 'unannotated/Stephanie_main'))
# Check duplicates between the two new samples and test_collection2, test_collection2, and agr_collection, respectively
check_duplicates(os.path.join(data_dir, 'unannotated/Samuel_main'), os.path.join(data_dir, 'unannotated/test_collection1'))
check_duplicates(os.path.join(data_dir, 'unannotated/Samuel_main'), os.path.join(data_dir, 'unannotated/test_collection2'))
check_duplicates(os.path.join(data_dir, 'unannotated/Samuel_main'), os.path.join(data_dir, 'unannotated/agr_collection'))
check_duplicates(os.path.join(data_dir, 'unannotated/Stephanie_main'), os.path.join(data_dir, 'unannotated/test_collection1'))
check_duplicates(os.path.join(data_dir, 'unannotated/Stephanie_main'), os.path.join(data_dir, 'unannotated/test_collection2'))
check_duplicates(os.path.join(data_dir, 'unannotated/Stephanie_main'), os.path.join(data_dir, 'unannotated/agr_collection'))

