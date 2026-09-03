from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    print("\n--- Testing GET / ---")
    response = client.get("/")
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    print("GET / passed!")

def test_ask_in_scope_question():
    print("\n--- Testing POST /ask (In-Scope Question) ---")
    question = "How long do I have to return a product?"
    response = client.post("/ask", json={"question": question})
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Question: {data['question']}")
    print(f"Answer: {data['answer']}")
    print(f"Retrieved Context Count: {len(data['retrieved_context'])}")
    assert response.status_code == 200
    assert data["question"] == question
    assert len(data["retrieved_context"]) == 3
    print("POST /ask (In-Scope) passed!")

def test_ask_out_of_scope_question():
    print("\n--- Testing POST /ask (Out-of-Scope Question) ---")
    question = "What is the distance from Earth to Mars?"
    response = client.post("/ask", json={"question": question})
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Question: {data['question']}")
    print(f"Answer: {data['answer']}")
    assert response.status_code == 200
    assert "I don't have enough information to answer that question." in data["answer"]
    print("POST /ask (Out-of-Scope Fallback) passed!")

if __name__ == "__main__":
    test_root_endpoint()
    test_ask_in_scope_question()
    test_ask_out_of_scope_question()
    print("\n All API tests passed successfully!")
