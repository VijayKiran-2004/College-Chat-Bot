import csv
import random
import sys
import os
import asyncio
from pathlib import Path
from collections import OrderedDict

# Add app directory to path
sys.path.insert(0, os.getcwd())

import backend
from backend import QueryRequest

LOG_FILE = "logs/response_log.csv"
TARGET_COUNT = 210

# Expanded Data
ROLES = [
    "Principal", "HOD of CSE", "Dean of Academics", "HOD of ECE", "Sports Director", 
    "Exam Branch Incharge", "Librarian", "Placement Officer", "HOD of IT", 
    "HOD of Civil", "HOD of Mechanical", "Admission Incharge", "Transport Incharge",
    "HOD of EEE", "HOD of MBA", "Administrative Officer", "Rector", "Director", 
    "Vice Principal", "Lab Assistant", "Project Coordinator", "NCC Officer", "NSS Officer"
]
LOCATIONS = [
    "Library", "Canteen", "Administrative Block", "CSE Department", "ECE Department", 
    "Seminar Hall", "Playground", "Hostel", "Placement Cell", "Exam Branch", 
    "Basketball Court", "Parking Area", "Main Gate", "Auditorium", 
    "Indoor Games Room", "Volleyball Court", "Chemistry Lab", "Physics Lab", "Workshop",
    "Girls Hostel", "Boys Hostel", "Gymnasium", "Board Room", "Reception"
]
TOPICS = [
    "NCC", "NSS", "Sports", "Placements", "Bus Transport", "Scholarships", 
    "Examinations", "Results", "Admissions", "Library Rules", "Hostel Fees", 
    "Canteen Menu", "College Timings", "Identity Cards", "Dress Code",
    "Anti-Ragging", "Student Clubs", "Medical Facilities", "Internet Access", "Events",
    "Industrial Visits", "Internships", "Research", "Alumni Association", "Convocation"
]
FACILITIES = [
    "Wi-Fi", "Library", "Bus", "Hostel", "Canteen", "Gym", "Medical Room", 
    "ATM", "Stationery Shop", "Digital Library", "Seminar Hall", "Conference Room",
    "Smart Classroom", "Auditorium", "Solar Power", "Water Purifier"
]
YEARS = ["1st", "2nd", "3rd", "4th"]

TEMPLATES = [
    "Who is the {role}?",
    "How can I contact the {role}?",
    "Can you tell me who the {role} is?",
    "What is the contact for {role}?",
    "Details about {role}?",
    "Where is the {location}?",
    "How do I get to the {location}?",
    "Where can I find the {location}?",
    "Location of {location}?",
    "Is the {location} open?",
    "Tell me about {topic}.",
    "What are the details for {topic}?",
    "Do you have info on {topic}?",
    "Can you give me information about {topic}?",
    "Explain {topic} to me.",
    "I want to know about {topic}.",
    "Is {facility} available?",
    "Does the college have a {facility}?",
    "Is there a {facility} on campus?",
    "Do we have {facility} facilities?",
    "What is the fee structure for {year} year?",
    "Tell me the fees for {year} year.",
    "How much is the fee for {year} year?",
    "Fee details for {year} year student?",
    "Who is in charge of {topic}?",
    "Who manages {topic}?",
]

def clean_log_file():
    print("Cleaning log file...", flush=True)
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} does not exist.", flush=True)
        return set(), 0

    unique_queries = OrderedDict() # Use dict to keep order but tracking unique keys
    rows_kept = []
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            for row in reader:
                if len(row) < 2: continue
                query = row[1].strip()
                if query not in unique_queries and query:
                    unique_queries[query] = row
                    rows_kept.append(row)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return set(), 0

    # Write back cleaned file
    with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows_kept)
    
    print(f"Cleaned logs. Kept {len(rows_kept)} unique rows.")
    return set(unique_queries.keys()), len(rows_kept)

def generate_new_prompts(count, existing_queries):
    prompts = []
    attempts = 0
    max_attempts = 50000
    
    print(f"Need to generate {count} NEW prompts...")
    
    while len(prompts) < count and attempts < max_attempts:
        attempts += 1
        template = random.choice(TEMPLATES)
        
        if "{role}" in template:
            filler = random.choice(ROLES)
            prompt = template.format(role=filler)
        elif "{location}" in template:
            filler = random.choice(LOCATIONS)
            prompt = template.format(location=filler)
        elif "{topic}" in template:
            filler = random.choice(TOPICS)
            prompt = template.format(topic=filler)
        elif "{facility}" in template:
            filler = random.choice(FACILITIES)
            prompt = template.format(facility=filler)
        elif "{year}" in template:
            filler = random.choice(YEARS)
            prompt = template.format(year=filler)
        else:
            prompt = template
            
        if prompt not in existing_queries and prompt not in prompts:
            prompts.append(prompt)
            
    return prompts

async def main():
    # 1. Clean Duplicates
    existing_queries, count = clean_log_file()
    
    missing = TARGET_COUNT - count
    if missing <= 0:
        print(f"Goal met! We have {count} unique rows.")
        return

    print(f"We need {missing} more unique rows to reach {TARGET_COUNT}.")
    
    # 2. Generate New Prompts
    new_prompts = generate_new_prompts(missing + 10, existing_queries) # +10 buffer
    if len(new_prompts) < missing:
        print(f"Warning: Could only generate {len(new_prompts)} unique prompts.")
    
    # 3. Use Backend to Process
    print("Initializing Backend...")
    backend.initialize_systems()
    
    print("Processing new prompts...", flush=True)
    processed = 0
    batch_size = 20
    
    for i, prompt in enumerate(new_prompts):
        if processed >= missing:
            break
            
        print(f"[{i+1}/{missing}] Processing: {prompt}", flush=True)
        try:
            # Double check uniqueness just in case
            if prompt in existing_queries:
                print("Skipping duplicate (race condition check)", flush=True)
                continue
                
            request = QueryRequest(message=prompt, session_id="clean_runner")
            await backend.chat(request)
            existing_queries.add(prompt)
            processed += 1
            
            if processed % batch_size == 0:
                print(f"--- Batch of {batch_size} completed. Total: {processed}/{missing} ---", flush=True)

        except Exception as e:
            print(f"Error: {e}", flush=True)

    print("\nFill complete!")
    clean_log_file() # Clean one last time just to be sure

if __name__ == "__main__":
    asyncio.run(main())
