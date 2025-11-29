import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8080"
SCREEN_ENDPOINT = f"{API_URL}/screen"
HEALTH_ENDPOINT = f"{API_URL}/health"

def test_health_check():
    """Tests the /health endpoint for basic connectivity."""
    print("--- 1. Testing Health Check ---")
    try:
        response = requests.get(HEALTH_ENDPOINT)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print(f"PASS: Health check succeeded. Status: {data.get('status')}")
                return True
            else:
                print(f"FAIL: Health check returned 200 but status is not 'healthy'. Response: {data}")
                return False
        else:
            print(f"FAIL: Health check returned status code {response.status_code}.")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"FAIL: Could not connect to the API at {API_URL}. Is the server running?")
        return False

def test_prediction(test_name, input_data, expected_code):
    """Generic function to test the /screen endpoint."""
    print(f"\n--- 2. Testing Prediction: {test_name} ---")
    
    try:
        response = requests.post(SCREEN_ENDPOINT, json=input_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            predicted_code = data.get("prediction_code")
            predicted_label = data.get("prediction")
            
            if predicted_code == expected_code:
                print(f"PASS: Prediction is correct. Result: {predicted_label} ({predicted_code})")
                return True
            else:
                print(f"FAIL: Wrong prediction. Expected code {expected_code}, got {predicted_code} ({predicted_label}).")
                print(f"   Input Text: {data.get('synthesized_text')}")
                return False
        
        elif response.status_code == 400:
             print(f"FAIL: Received 400 Bad Request. Test expects 200. Response: {response.json()}")
             return False

        else:
            print(f"FAIL: Request failed with status code {response.status_code}. Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"FAIL: Request error: {e}")
        return False

def test_invalid_input():
    """Tests the /screen endpoint for graceful handling of missing data (400 check)."""
    print("\n--- 3. Testing Invalid Input (Expected 400) ---")
    input_data = {"age": 70, "mmse_score": 25} # Missing 4 required binary fields
    
    try:
        response = requests.post(SCREEN_ENDPOINT, json=input_data, timeout=10)

        if response.status_code == 400:
            print(f"PASS: Received expected 400 error. Message: {response.json().get('message')}")
            return True
        else:
            print(f"FAIL: Expected status 400, but got {response.status_code}. Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"FAIL: Request error: {e}")
        return False

if __name__ == "__main__":
    
    LOW_RISK_DATA = {
        "age": 65, 
        "mmse_score": 30, 
        "memory_complaints": 0, 
        "confusion": 0, 
        "disorientation": 0, 
        "daily_tasks_difficulty": 0
    }
    HIGH_RISK_DATA = {
        "age": 85, 
        "mmse_score": 15, 
        "memory_complaints": 0, 
        "confusion": 1, 
        "disorientation": 0, 
        "daily_tasks_difficulty": 0
    }
    TEST_A_AGE = {
        "age": 55, "mmse_score": 30, "memory_complaints": 0, 
        "confusion": 0, "disorientation": 0, "daily_tasks_difficulty": 0
    }
    TEST_B_MMSE = {
        "age": 65, "mmse_score": 28, "memory_complaints": 0, 
        "confusion": 0, "disorientation": 0, "daily_tasks_difficulty": 0
    }
    TEST_C_SYMPTOM = {
        "age": 65, "mmse_score": 30, "memory_complaints": 1, # Only 1 symptom is '1'
        "confusion": 0, "disorientation": 0, "daily_tasks_difficulty": 0
    }

    print("\n\n--- RUNNING DEBUG TESTS TO ISOLATE FEATURE BIAS ---")
    
    # Test A: Checking sensitivity to age (expecting 0)
    # test_prediction("DEBUG A: Healthy, Age 55", TEST_A_AGE, expected_code=0) 
    
    # Test B: Checking sensitivity to MMSE (expecting 0)
    # test_prediction("DEBUG B: Healthy, MMSE 28", TEST_B_MMSE, expected_code=0) 
    
    # Test C: Checking if one symptom instantly triggers a high risk (expecting 0, but testing weight)
    # test_prediction("DEBUG C: Healthy, ONE Symptom (Memory Complaints)", TEST_C_SYMPTOM, expected_code=0) 
    
    # print("\n--- RERUN THE ORIGINAL LOW-RISK FAIL CASE FOR REFERENCE ---")
    # Original Fail Case for comparison
    # test_prediction("Original Low Risk Fail Case (Age 65, MMSE 30)", 
                     #{"age": 65, "mmse_score": 30, "memory_complaints": 0, 
                     #"confusion": 0, "disorientation": 0, "daily_tasks_difficulty": 0}, expected_code=0)

    results = {}
    
    print("\n\n--- RUNNING INITIAL SMOKE TESTS ---")
    results['health'] = test_health_check()
    # Testing the known failing case for confirmation
    results['low_risk_fail_case'] = test_prediction("Original Low Risk Fail Case", LOW_RISK_DATA, expected_code=0)
    results['high_risk'] = test_prediction("High Risk Profile", HIGH_RISK_DATA, expected_code=1)
    results['invalid_input'] = test_invalid_input()
    
    # --- New Debug Tests (3 total) ---
    print("\n\n--- RUNNING DEBUG TESTS TO ISOLATE FEATURE BIAS ---")
    results['debug_age_55'] = test_prediction("DEBUG A: Healthy, Age 55", TEST_A_AGE, expected_code=0) 
    results['debug_mmse_28'] = test_prediction("DEBUG B: Healthy, MMSE 28", TEST_B_MMSE, expected_code=0) 
    results['debug_one_symptom'] = test_prediction("DEBUG C: Healthy, ONE Symptom (Memory Complaints 1)", TEST_C_SYMPTOM, expected_code=0)
   
    total_passed = sum(results.values())
    total_tests = len(results)
    print("\n==============================")
    print(f"SMOKE TEST SUMMARY: {total_passed}/{total_tests} Tests Passed")
    print("==============================")