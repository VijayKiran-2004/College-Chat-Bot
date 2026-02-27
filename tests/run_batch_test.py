import asyncio
import random
import sys
import os
from pathlib import Path

print("Script started... importing modules...", flush=True)

# Add project root to path (one level up from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

import backend
from backend import QueryRequest

print("Imports complete.", flush=True)

import csv

# Expanded Data to ensure >200 unique combinations
ROLES = [
    "Principal", "HOD of CSE", "Dean of Academics", "HOD of ECE", "Sports Director", 
    "Exam Branch Incharge", "Librarian", "Placement Officer", "HOD of IT", 
    "HOD of Civil", "HOD of Mechanical", "Admission Incharge", "Transport Incharge",
    "HOD of EEE", "HOD of MBA", "Administrative Officer", "Rector", "Director"
]
LOCATIONS = [
    "Library", "Canteen", "Administrative Block", "CSE Department", "ECE Department", 
    "Seminar Hall", "Playground", "Hostel", "Placement Cell", "Exam Branch", 
    "Basketball Court", "Parking Area", "Main Gate", "Auditorium", 
    "Indoor Games Room", "Volleyball Court", "Chemistry Lab", "Physics Lab", "Workshop"
]
TOPICS = [
    "NCC", "NSS", "Sports", "Placements", "Bus Transport", "Scholarships", 
    "Examinations", "Results", "Admissions", "Library Rules", "Hostel Fees", 
    "Canteen Menu", "College Timings", "Identity Cards", "Dress Code",
    "Anti-Ragging", "Student Clubs", "Medical Facilities", "Internet Access", "Events"
]
FACILITIES = [
    "Wi-Fi", "Library", "Bus", "Hostel", "Canteen", "Gym", "Medical Room", 
    "ATM", "Stationery Shop", "Digital Library", "Seminar Hall", "Conference Room"
]
YEARS = ["1st", "2nd", "3rd", "4th"]

TEMPLATES = [
    "Who is the {role}?",
    "How can I contact the {role}?",
    "Can you tell me who the {role} is?",
    "What is the contact for {role}?",
    "Where is the {location}?",
    "How do I get to the {location}?",
    "Where can I find the {location}?",
    "Location of {location}?",
    "Tell me about {topic}.",
    "What are the details for {topic}?",
    "Do you have info on {topic}?",
    "Can you give me information about {topic}?",
    "Explain {topic} to me.",
    "Is {facility} available?",
    "Does the college have a {facility}?",
    "Is there a {facility} on campus?",
    "What is the fee structure for {year} year?",
    "Tell me the fees for {year} year.",
    "How much is the fee for {year} year?",
    "Who is in charge of {topic}?",
    "Who manages {topic}?",
]

def get_existing_queries(log_file):
    """Read existing queries from the log file to avoid duplicates."""
    existing = set()
    path = Path(log_file)
    if not path.exists():
        return existing
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if len(row) > 1:
                    existing.add(row[1].strip())
    except Exception as e:
        print(f"Warning: Could not read existing logs: {e}")
    return existing

def generate_unique_prompts(count, existing_queries):
    prompts = []
    attempts = 0
    max_attempts = 10000 # Avoid infinite loop
    
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

async def run_batch(batch_size=10):
    print("Initialize Backend Systems...")
    backend.initialize_systems()
    
    log_file = backend.response_logger.log_file
    existing_queries = get_existing_queries(log_file)
    print(f"Found {len(existing_queries)} existing unique queries in logs.")
    
    print(f"Generating {batch_size} NEW unique prompts...")
    prompts = generate_unique_prompts(batch_size, existing_queries)
    
    if not prompts:
        print("Could not generate any new unique prompts! You might have exhausted the variations.")
        return

    print(f"Starting Batch Processing of {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i:03d}/{batch_size}] Processing: {prompt}")
        try:
            request = QueryRequest(message=prompt, session_id="batch_runner_incremental")
            await backend.chat(request)
        except Exception as e:
            print(f"Error on prompt '{prompt}': {e}")
            
    print("\nBatch Run Complete!")
    print(f"Check logs at: {log_file}")

if __name__ == "__main__":
    # You can change the batch size here
    BATCH_SIZE = 10
    if len(sys.argv) > 1:
        try:
            BATCH_SIZE = int(sys.argv[1])
        except ValueError:
            pass
            
    asyncio.run(run_batch(BATCH_SIZE))
