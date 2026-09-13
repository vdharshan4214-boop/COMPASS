from fastapi.testclient import TestClient

from main import app, get_db
from auth import hash_password

client = TestClient(app)
app.state.limiter.enabled = False



def reset_db():
    db = get_db()
    for collection in ["users", "complaints", "notifications", "counters"]:
        db[collection].delete_many({}) if hasattr(db[collection], "delete_many") else None


def bootstrap_user(email: str, role: str = "student"):
    db = get_db()
    user = {
        "_id": 1 if role == "student" else 2,
        "id": 1 if role == "student" else 2,
        "name": "Test User" if role == "student" else "Admin User",
        "email": email,
        "password_hash": hash_password("secret123"),
        "role": role,
        "streak_count": 0,
        "last_active_date": None,
    }
    db["users"].insert_one(user)
    return user


def test_register_and_login():
    reset_db()
    response = client.post("/api/auth/register", json={"name": "Alice", "email": "alice@example.com", "password": "secret123", "role": "student"})
    assert response.status_code == 200
    payload = response.json()
    assert "token" in payload

    login = client.post("/api/auth/login", json={"email": "alice@example.com", "password": "secret123"})
    assert login.status_code == 200


def test_submit_and_dedup_and_feedback_flow():
    reset_db()
    boot_user = bootstrap_user("student@example.com")
    admin = bootstrap_user("admin@example.com", role="admin")
    token = client.post("/api/auth/login", json={"email": "student@example.com", "password": "secret123"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    r1 = client.post("/api/complaints", data={"description": "WiFi is down again in Block A", "location": "Block A"}, headers=headers)
    assert r1.status_code == 200

    r2 = client.post("/api/complaints", data={"description": "Wifi is down in Block A this morning", "location": "Block A"}, headers=headers)
    assert r2.status_code == 200

    complaint = client.get("/api/complaints", headers={"Authorization": f"Bearer {client.post('/api/auth/login', json={'email': 'admin@example.com', 'password': 'secret123'}).json()['token']}"})
    assert complaint.status_code == 200
    assert len(complaint.json()) >= 1

    my = client.get("/api/complaints/mine", headers=headers)
    assert my.status_code == 200
    assert len(my.json()) >= 1

    resolved = client.post("/api/complaints/1/resolve", json={"approved": True, "message": "Fixed."}, headers={"Authorization": f"Bearer {client.post('/api/auth/login', json={'email': 'admin@example.com', 'password': 'secret123'}).json()['token']}"})
    assert resolved.status_code == 200

    feedback = client.post("/api/complaints/1/feedback", json={"rating": 5, "comment": "Good fix"}, headers=headers)
    assert feedback.status_code == 200


def test_predictions_and_admin_summary():
    reset_db()
    bootstrap_user("student@example.com")
    bootstrap_user("admin@example.com", role="admin")
    student_token = client.post("/api/auth/login", json={"email": "student@example.com", "password": "secret123"}).json()["token"]
    admin_token = client.post("/api/auth/login", json={"email": "admin@example.com", "password": "secret123"}).json()["token"]

    client.post("/api/complaints", data={"description": "Internet is not working in Block A", "location": "Block A"}, headers={"Authorization": f"Bearer {student_token}"})
    pred = client.get("/api/predictions/nearby?location=Block%20A", headers={"Authorization": f"Bearer {student_token}"})
    assert pred.status_code == 200
    assert "predictions" in pred.json()

    summary = client.get("/api/admin/summary", headers={"Authorization": f"Bearer {admin_token}"})
    assert summary.status_code == 200
    assert "total_complaints" in summary.json()


def test_duplicate_co_submitter_in_my_complaints():
    reset_db()
    reg1 = client.post("/api/auth/register", json={"name": "Student One", "email": "student1@example.com", "password": "password123", "role": "student"}).json()
    reg2 = client.post("/api/auth/register", json={"name": "Student Two", "email": "student2@example.com", "password": "password123", "role": "student"}).json()

    headers1 = {"Authorization": f"Bearer {reg1['token']}"}
    headers2 = {"Authorization": f"Bearer {reg2['token']}"}

    # Student 1 submits first complaint
    c1 = client.post("/api/complaints", data={"description": "Water supply stopped completely in Hostel 2 block B", "location": "Hostel 2"}, headers=headers1)
    assert c1.status_code == 200

    # Student 2 submits duplicate complaint
    c2 = client.post("/api/complaints", data={"description": "Water supply stopped completely in Hostel 2 block B", "location": "Hostel 2"}, headers=headers2)
    assert c2.status_code == 200
    assert "merged_into" in c2.json()

    # Check Student 2 my_complaints
    mine2 = client.get("/api/complaints/mine", headers=headers2)
    assert mine2.status_code == 200
    complaints2 = mine2.json()
    assert len(complaints2) == 1
    assert complaints2[0]["id"] == c1.json()["complaint"]["id"]
    assert complaints2[0]["frequency"] == 2

