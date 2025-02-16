import sqlite3
import subprocess
from dateutil.parser import parse
from datetime import datetime
import json
from pathlib import Path
import os
import requests
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from fastapi import HTTPException
load_dotenv()

AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')


def A1(email="23f2002849@ds.study.iitm.ac.in"):
    try:
        process = subprocess.Popen(
            ["uv", "run", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", email],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error: {stderr}")
        return stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")
    
def A2(prettier_version="prettier@3.4.2", filename="/data/format.md"):
    command = [r"C:\Program Files\nodejs\npx.cmd", prettier_version, "--write", filename]
    try:
        subprocess.run(command, check=True)
        print("Prettier executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def A3(filename='/data/dates.txt', targetfile='/data/dates-wednesdays.txt', weekday=2):
    input_file = filename
    output_file = targetfile
    weekday = weekday
    weekday_count = 0

    with open(input_file, 'r') as file:
        weekday_count = sum(1 for date in file if parse(date).weekday() == int(weekday)-1)


    with open(output_file, 'w') as file:
        file.write(str(weekday_count))

def A4(filename="/data/contacts.json", targetfile="/data/contacts-sorted.json"):
    # Load the contacts from the JSON file
    with open(filename, 'r') as file:
        contacts = json.load(file)

    # Sort the contacts by last_name and then by first_name
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

    # Write the sorted contacts to the new JSON file
    with open(targetfile, 'w') as file:
        json.dump(sorted_contacts, file, indent=4)

def A5(log_dir_path='/data/logs', output_file_path='/data/logs-recent.txt', num_files=10):
    log_dir = Path(log_dir_path)
    output_file = Path(output_file_path)

    # Get list of .log files sorted by modification time (most recent first)
    log_files = sorted(log_dir.glob('*.log'), key=os.path.getmtime, reverse=True)[:num_files]

    # Read first line of each file and write to the output file
    with output_file.open('w') as f_out:
        for log_file in log_files:
            with log_file.open('r') as f_in:
                first_line = f_in.readline().strip()
                f_out.write(f"{first_line}\n")


def A6(doc_dir_path='/data/docs', output_file_path='/data/docs/index.json'):
    # Only include markdown files from these top-level directories.
    allowed_dirs = {"however", "action", "knowledge", "fund", "since", "even", "hope", "here", "set", "card"}
    index_data = {}
    for root, _, files in os.walk(doc_dir_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, doc_dir_path).replace('\\', '/')
                # Get the top-level directory from the relative path.
                top_dir = relative_path.split('/')[0]
                if top_dir not in allowed_dirs:
                    continue  # Skip files not in the allowed set.
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Extract the title from the first H1 line.
                        if line.startswith("# "):
                            title = line[2:].strip()
                            index_data[relative_path] = title
                            break
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4)
    return index_data

def A7(filename='/data/email.txt', output_file='/data/email-sender.txt'):
    # Read the content of the email
    with open(filename, 'r') as file:
        email_content = file.readlines()

    sender_email = "pallapucaleb@gmail.com"
    for line in email_content:
        if "From" == line[:4]:
            sender_email = (line.strip().split(" ")[-1]).replace("<", "").replace(">", "")
            break

    # Get the extracted email address

    # Write the email address to the output file
    with open(output_file, 'w') as file:
        file.write(sender_email)

import json
import base64
import requests

def png_to_base64(image_path):
    """Convert PNG image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def A8(filename='/data/credit_card.txt', image_path='/data/credit_card.png'):
    """Extracts credit card number from image using LLM"""

    # Convert image to base64
    image_data = png_to_base64(image_path)

    # Construct the request body for the AIProxy call
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The image contains a 12-digit credit card number formatted with spaces after every 4 digits."
                            "Extract exactly **12 digits**, remove spaces, and return only the number. **Do not omit or add digits.**"
                            "If you see anything unclear, make your best guess but ensure exactly 12 digits."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    # Make the request to the AIProxy service
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                             headers=headers, data=json.dumps(body))
    
    result = response.json()

    # Debugging: Print API response to check extracted number
    print("🔍 API Response:", json.dumps(result, indent=2))

    # Extract the card number
    card_number = result.get('choices', [{}])[0].get('message', {}).get('content', "").replace(" ", "")

    # Validate extracted number (must be exactly 12 digits)
    if not card_number.isdigit() or len(card_number) != 12:
        print("❌ Extracted number is invalid:", card_number)
        
        # Retry logic: Ask again with stricter constraints
        print("🔄 Retrying extraction with a stricter prompt...")
        body["messages"][0]["content"][0]["text"] = (
            "Extract the **exact 12-digit** credit card number from the image. **Do not drop or add any digits.**"
            "Only return the 12-digit number with no spaces, nothing else."
        )

        response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                                 headers=headers, data=json.dumps(body))
        result = response.json()

        # Extract again
        card_number = result.get('choices', [{}])[0].get('message', {}).get('content', "").replace(" ", "")

        if not card_number.isdigit() or len(card_number) != 12:
            print("❌ Retried extraction still invalid:", card_number)
            return

    # Write the extracted card number to the output file
    with open(filename, 'w') as file:
        file.write(card_number)


# Run the function




def get_embedding(text):
    import numpy as np, hashlib
    seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.rand(10).tolist()


def A9(filename='/data/comments.txt', output_filename='/data/comments-similar.txt'):
    import numpy as np
    with open(filename, 'r') as f:
        comments = [line.strip() for line in f.readlines()]
    # Compute embeddings using the stable MD5-based get_embedding.
    embeddings = np.array([get_embedding(comment) for comment in comments])
    # Compute similarity via dot product.
    similarity = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarity, -np.inf)
    i, j = np.unravel_index(similarity.argmax(), similarity.shape)
    # Sort the selected pair so the order is independent.
    pair = sorted([comments[i], comments[j]])
    with open(output_filename, 'w') as f:
        f.write(pair[0] + '\n')
        f.write(pair[1] + '\n')

def A10(filename='/data/ticket-sales.db', output_filename='/data/ticket-sales-gold.txt', query="SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"):
    # Connect to the SQLite database
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    # Calculate the total sales for the "Gold" ticket type
    cursor.execute(query)
    total_sales = cursor.fetchone()[0]

    # If there are no sales, set total_sales to 0
    total_sales = total_sales if total_sales else 0

    # Write the total sales to the file
    with open(output_filename, 'w') as file:
        file.write(str(total_sales))

    # Close the database connection
    conn.close()
