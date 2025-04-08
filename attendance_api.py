import requests
import json
import time
import uuid
from datetime import datetime

def post_attendance_swipe(name, employee_id, timestamp, latitude, longitude, swipe_type="In-Time", max_retries=3, retry_delay=2):
    # Validate and format the swipe type
    valid_types = ["In-Time", "Out-Time"]
    if swipe_type not in valid_types:
        print(f"Invalid swipe type: {swipe_type}. Defaulting to In-Time")
        swipe_type = "In-Time"
    
    # Format timestamp to ensure it's in the correct format
    try:
        # Parse the timestamp to ensure it's valid
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.000Z")
        formatted_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    except ValueError:
        print(f"Invalid timestamp format: {timestamp}")
        return {"status": "error", "message": "Invalid timestamp format"}

    # Generate a unique identifier for this swipe
    unique_id = str(uuid.uuid4())
    
    # Structure the data correctly
    data = {
        "employee_id": int(employee_id),
        "swipe_time": formatted_timestamp,
        "status": "Open",
        "primary_flag": "Y",
        "company_id": None,
        "app_id": 1,
        "completed_by": None,
        "completed_date": None,
        "cancelled_by": None,
        "cancelled_date": None,
        "row_version": 1,
        "created": formatted_timestamp,
        "created_by": name.upper(),
        "updated": formatted_timestamp,
        "updated_by": name.upper(),
        "swipe_type": swipe_type,
        "latitude": latitude,
        "longitude": longitude,
        "employee_name": name
    }

    # Print the data being sent for debugging
    print("\n=== Sending Data to API ===")
    print(f"Employee: {name}")
    print(f"Employee ID: {employee_id}")
    print(f"Swipe Type: {swipe_type}")
    print(f"Timestamp: {formatted_timestamp}")
    print(f"Location: {latitude}, {longitude}")
    print("Full Data:", json.dumps(data, indent=2))

    # API endpoint URL
    url = 'https://ddottt6z7ccpe0a-apexdb.adb.me-jeddah-1.oraclecloudapps.com/ords/otrix/oc_hcm_employee_attendance_swips/'
    
    # Add authentication headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Basic dXNlcjpwYXNzd29yZA=="  # Basic auth with user:password
    }

    for attempt in range(max_retries):
        try:
            # Add a small delay between retries
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1} of {max_retries}...")
                time.sleep(retry_delay)
            
            # Send POST request with JSON-encoded data
            response = requests.post(url, headers=headers, data=json.dumps(data))

            # Print the full response for debugging
            print("\n=== API Response ===")
            print(f"Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")

            if response.status_code in [200, 201]:
                # The API returns the created record with an ID, which indicates success
                response_data = response.json()
                if isinstance(response_data, dict) and "id" in response_data:
                    print(f"\n=== Success ===")
                    print(f"Attendance swipe successful for {name}")
                    print(f"Record ID: {response_data['id']}")
                    return {"status": "success", "message": "Attendance swipe successful", "record_id": response_data['id']}
                else:
                    print(f"\n=== Warning ===")
                    print(f"Unexpected response format: {response_data}")
                    return {"status": "error", "message": "Unexpected response format"}
            else:
                error_data = response.json()
                if "ORA-04091" in str(error_data):
                    print(f"\n=== Warning ===")
                    print(f"Mutating table error detected, will retry...")
                    # Increase delay for subsequent retries
                    retry_delay = retry_delay * 2
                    continue
                else:
                    print(f"\n=== Error ===")
                    print(f"Error in swipe attendance: {response.status_code}")
                    print(f"Error details: {response.text}")
                    return {"status": "error", "message": response.text}
        
        except Exception as e:
            print(f"\n=== Error ===")
            print(f"Error in sending request: {str(e)}")
            if attempt == max_retries - 1:  # Last attempt
                return {"status": "error", "message": str(e)}
    
    return {"status": "error", "message": "Max retries exceeded"}
