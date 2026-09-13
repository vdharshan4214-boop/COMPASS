import os
import shutil
import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from apscheduler.schedulers.background import BackgroundScheduler

from database import get_db, next_id, seed_demo_data
from auth import hash_password, verify_password, make_token, parse_token
from ml_utils import (
    answer_admin_query,
    classify_category,
    compute_urgency,
    compute_priority,
    draft_resolution_message,
    find_duplicate,
    infer_department,
)
from email_service import send_email
from campus_rag import retrieve_campus_context
from chatbot_rag import chat_answer
from predictive_utils import predict_upcoming_issues
from agents.graph import run_reactive_pipeline
from agents.predictive_graph import run_predictive_pipeline
from vision_utils import classify_photo
from rag.vector_store import upsert_documents

BASE_DIR = os.path.dirname(__file__)
if os.getenv("VERCEL"):
    UPLOAD_DIR = "/tmp/uploads"
else:
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = FastAPI(title="Compass API")
limiter = Limiter(key_func=lambda request: request.headers.get("Authorization") or get_remote_address(request))
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    item.strip() for item in os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",") if item.strip()
]

scheduler = BackgroundScheduler()


@app.on_event("startup")
def startup_seed_demo_data():
    seed_demo_data()
    if not scheduler.running:
        scheduler.add_job(send_weekly_summary, "cron", day_of_week="mon", hour=8, minute=0, id="weekly_summary")
        scheduler.start()


@app.on_event("shutdown")
def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown()


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_current_user(authorization: Optional[str] = Header(None), db=Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    token = authorization.split(" ", 1)[1]
    payload = parse_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    user = db["users"].find_one({"_id": payload["uid"]})
    if not user:
        raise HTTPException(401, "User not found")
    return user


def require_role(user: dict, role: str):
    if user.get("role") != role:
        raise HTTPException(403, f"{role} access required")


def update_streak(user: dict, db):
    today = datetime.date.today()
    last_value = user.get("last_active_date")
    if isinstance(last_value, datetime.datetime):
        last = last_value.date()
    elif isinstance(last_value, datetime.date):
        last = last_value
    else:
        last = None

    streak_count = user.get("streak_count") or 0
    if last == today:
        return

    if last == (today - datetime.timedelta(days=1)):
        streak_count += 1
    else:
        streak_count = 1

    last_saved = datetime.datetime.combine(today, datetime.time.min)
    db["users"].update_one(
        {"_id": user["_id"]},
        {"$set": {"streak_count": streak_count, "last_active_date": last_saved}},
    )


def format_iso_utc(dt):

    if dt is None:
        return None
    if isinstance(dt, datetime.datetime):
        s = dt.isoformat()
        if not s.endswith("Z") and "+" not in s:
            s += "Z"
        return s
    s = str(dt)
    if not s.endswith("Z") and "+" not in s:
        s += "Z"
    return s


def serialize_complaint(c: dict, include_student: bool = False, student: Optional[dict] = None) -> dict:
    db = get_db()
    if student is None and c.get("student_id"):
        student = db["users"].find_one({"_id": c.get("student_id")})

    created_at = c.get("created_at")
    resolved_at = c.get("resolved_at")
    student_conf = 0.0
    if student:
        student_conf = float(student.get("confidence_score", 0.0))
    desc = c.get("description") or ""
    title = c.get("title") or (desc[:50] + ("..." if len(desc) > 50 else ""))
    out = {
        "id": c.get("id") or c.get("_id"),
        "title": title,
        "description": desc,
        "location": c.get("location"),
        "category": c.get("category"),
        "category_source": c.get("category_source") or "text",
        "department": c.get("department") or infer_department(c.get("category"), desc),
        "status": c.get("status"),
        "urgency": round(float(c.get("urgency") or 0), 2),
        "frequency": c.get("frequency") or 1,
        "days_open": c.get("days_open") or 0,
        "priority_score": round(float(c.get("priority_score") or 0), 1),
        "photo_path": c.get("photo_path"),
        "anonymous": bool(c.get("anonymous")),
        "draft_message": c.get("draft_message"),
        "admin_approved": bool(c.get("admin_approved")),
        "created_at": format_iso_utc(created_at),
        "resolved_at": format_iso_utc(resolved_at),
        "student_confidence_score": round(student_conf, 1),
    }
    if include_student:
        if c.get("anonymous"):
            out["student_name"] = "Anonymous"
            out["student_email"] = "N/A"
        elif student:
            out["student_name"] = student.get("name", "Student")
            out["student_email"] = student.get("email", "student@compass.edu")
    return out


def notify(db, student_id, complaint_id: int, message: str):
    """Create a lightweight in-app notification for a student."""
    notif_id = next_id("notifications")
    db["notifications"].insert_one({
        "_id": notif_id,
        "id": notif_id,
        "student_id": student_id,
        "complaint_id": complaint_id,
        "message": message,
        "read": False,
        "created_at": datetime.datetime.utcnow(),
    })


def serialize_notification(n: dict):
    created_at = n.get("created_at")
    return {
        "id": n.get("id") or n.get("_id"),
        "complaint_id": n.get("complaint_id"),
        "message": n.get("message"),
        "read": bool(n.get("read")),
        "created_at": format_iso_utc(created_at),
    }



class RegisterIn(BaseModel):
    name: str
    email: str
    password: str
    role: str


class LoginIn(BaseModel):
    email: str
    password: str


class ResolveIn(BaseModel):
    approved: bool
    message: Optional[str] = None


class AdminQueryIn(BaseModel):
    query: str


class ChatIn(BaseModel):
    message: str
    history: list[dict] | None = None


class FeedbackIn(BaseModel):
    rating: int
    comment: Optional[str] = None


class ProfileUpdateIn(BaseModel):
    name: Optional[str] = None
    block: Optional[str] = None
    email_notifications: Optional[bool] = None


def validate_report_text(value: str, field_name: str, *, min_length: int, max_length: int) -> str:
    cleaned = (value or "").strip()
    if len(cleaned) < min_length or len(cleaned) > max_length:
        raise HTTPException(400, f"{field_name} must be between {min_length} and {max_length} characters")
    return cleaned


@app.post("/api/auth/register")
def register(data: RegisterIn, db=Depends(get_db)):
    users = db["users"]
    if data.role not in ("student", "admin"):
        raise HTTPException(400, "role must be 'student' or 'admin'")
    if users.find_one({"email": data.email.lower()}):
        raise HTTPException(400, "Email already registered")

    user_id = next_id("users")
    user = {
        "_id": user_id,
        "id": user_id,
        "name": data.name,
        "email": data.email.lower(),
        "password_hash": hash_password(data.password),
        "role": data.role,
        "confidence_score": 0.0,
        "streak_count": 0,
        "last_active_date": None,
    }
    users.insert_one(user)
    token = make_token(user_id, data.role)
    return {"token": token, "role": data.role, "name": data.name, "user_id": user_id}


@app.post("/api/auth/login")
@limiter.limit("5/minute")
def login(request: Request, data: LoginIn, db=Depends(get_db)):
    user = db["users"].find_one({"email": data.email.lower()})
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    update_streak(user, db)
    user = db["users"].find_one({"_id": user["_id"]})
    token = make_token(user["_id"], user["role"])
    return {"token": token, "role": user["role"], "name": user["name"], "user_id": user["_id"], "streak_count": user.get("streak_count") or 0}


@app.get("/api/me")
def me(user: dict = Depends(get_current_user)):
    return {"id": user["_id"], "name": user["name"], "email": user["email"], "role": user["role"], "streak_count": user.get("streak_count") or 0}


@app.post("/api/complaints")
@limiter.limit("10/minute")
def create_complaint(
    request: Request,
    description: str = Form(...),
    location: str = Form(...),
    title: Optional[str] = Form(None),
    anonymous: bool = Form(False),
    photo: Optional[UploadFile] = File(None),
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    require_role(user, "student")
    update_streak(user, db)

    description = validate_report_text(description, "Complaint description", min_length=10, max_length=500)
    location = validate_report_text(location, "Location", min_length=2, max_length=120)
    if title and title.strip():
        clean_title = validate_report_text(title, "Complaint title", min_length=3, max_length=120)
    else:
        clean_title = description[:50] + ("..." if len(description) > 50 else "")

    photo_path = None
    if photo is not None:
        try:
            from PIL import Image
            with Image.open(photo.file) as img:
                img.verify()
        except Exception:
            raise HTTPException(400, "Uploaded file is not a valid image")
        photo.file.seek(0)
        ext = os.path.splitext(photo.filename)[1] or ".jpg"
        photo_path = f"uploads/{user['_id']}_{int(datetime.datetime.utcnow().timestamp())}{ext}"
        with open(os.path.join(BASE_DIR, photo_path), "wb") as f:
            shutil.copyfileobj(photo.file, f)

    category_source = "text"
    text_category = classify_category(description)
    vision_category = classify_photo(os.path.join(BASE_DIR, photo_path)) if photo_path else None
    final_category = text_category
    if vision_category and vision_category in {"wifi", "plumbing", "electrical", "hostel", "safety", "other"}:
        if vision_category != text_category:
            final_category = text_category if text_category in {"wifi", "plumbing", "electrical", "hostel", "safety", "other"} else vision_category
            category_source = "agreed" if vision_category == text_category else "photo"
        else:
            category_source = "agreed"
    category = final_category
    urgency = compute_urgency(description)
    complaint_state = run_reactive_pipeline(description, location, photo_path=photo_path, open_complaints=list(db["complaints"].find({"status": {"$ne": "resolved"}})))
    category = complaint_state.get("category") or category
    urgency = complaint_state.get("urgency") or urgency
    department = complaint_state.get("department") or infer_department(category, description)
    dedup_match = complaint_state.get("dedup_match")

    complaints = db["complaints"]
    if dedup_match:
        dup = complaints.find_one({"id": dedup_match})
        if dup:
            new_freq = (dup.get("frequency") or 1) + 1
            new_score = compute_priority(dup["urgency"], new_freq, dup.get("days_open") or 0)
            co_subs = list(dup.get("co_submitters") or [])
            if user["_id"] not in co_subs and user["_id"] != dup.get("student_id"):
                co_subs.append(user["_id"])
            complaints.update_one(
                {"_id": dup["_id"]},
                {
                    "$set": {
                        "frequency": new_freq,
                        "priority_score": new_score,
                        "department": dup.get("department") or department,
                        "co_submitters": co_subs,
                    }
                },
            )
            # Award +3 points to co-submitter for merging duplicate
            db["users"].update_one({"_id": user["_id"]}, {"$inc": {"confidence_score": 3.0}})
            dup = complaints.find_one({"_id": dup["_id"]})
            return {"merged_into": dup["id"], "message": "Matched an existing open complaint; merged as a duplicate.", "complaint": serialize_complaint(dup)}

    complaint_id = next_id("complaints")
    complaint = {
        "_id": complaint_id,
        "id": complaint_id,
        "student_id": user["_id"],
        "co_submitters": [],
        "anonymous": bool(anonymous),
        "title": clean_title,
        "description": description,
        "location": location,
        "photo_path": photo_path,
        "category": category,
        "category_source": category_source,
        "department": department,
        "urgency": urgency,
        "frequency": 1,
        "days_open": 0,
        "status": "submitted",

        "draft_message": None,
        "admin_approved": False,
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow(),
        "resolved_at": None,
        "priority_score": compute_priority(urgency, 1, 0),
    }
    complaints.insert_one(complaint)
    # Award +3 points to student for submitting report
    db["users"].update_one({"_id": user["_id"]}, {"$inc": {"confidence_score": 3.0}})
    try:
        upsert_documents("complaints", [{"id": str(complaint_id), "text": description, "category": category, "location": location}])
    except Exception:
        pass
    return {"message": "Complaint submitted", "complaint": serialize_complaint(complaint)}


@app.get("/api/complaints/mine")
def my_complaints(user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "student")
    refresh_days_open(db)
    uid = user["_id"]
    uids = [uid]
    if isinstance(uid, int):
        uids.append(str(uid))
    elif isinstance(uid, str) and uid.isdigit():
        uids.append(int(uid))
    if user.get("id") and user.get("id") not in uids:
        uids.append(user.get("id"))
    all_complaints = list(db["complaints"].find().sort("created_at", -1))
    mine = []
    for c in all_complaints:
        st_id = c.get("student_id")
        co_subs = c.get("co_submitters") or []
        if st_id in uids or any(u in uids for u in co_subs):
            mine.append(serialize_complaint(c))
    return mine



@app.get("/api/streak")
def streak(user: dict = Depends(get_current_user)):
    require_role(user, "student")
    conf_score = round(float(user.get("confidence_score", 0.0)), 1)
    return {
        "streak_count": user.get("streak_count") or 0,
        "confidence_score": conf_score,
        "last_active_date": str(user.get("last_active_date")),
    }


class BonusConfidenceIn(BaseModel):
    bonus: float = 10.0


@app.post("/api/profile/bonus-confidence")
def add_bonus_confidence(data: BonusConfidenceIn, user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "student")
    current_conf = float(user.get("confidence_score", 0.0))
    new_conf = min(200.0, current_conf + float(data.bonus))
    db["users"].update_one({"_id": user["_id"]}, {"$set": {"confidence_score": new_conf}})
    return {"confidence_score": round(new_conf, 1), "message": f"+{data.bonus} Confidence Score awarded!"}


@app.get("/api/leaderboard")
def leaderboard(user: dict = Depends(get_current_user), db=Depends(get_db)):
    students = list(db["users"].find({"role": "student"}))
    complaints = list(db["complaints"].find())

    counts: dict = {}
    for c in complaints:
        sid = c.get("student_id")
        counts[sid] = counts.get(sid, 0) + 1
        for co_id in (c.get("co_submitters") or []):
            counts[co_id] = counts.get(co_id, 0) + 1

    rows = []
    for s in students:
        sid = s.get("_id")
        conf_score = round(float(s.get("confidence_score", 0.0)), 1)
        r_count = counts.get(sid, 0)
        if conf_score >= 130:
            badge = "Champion"
        elif conf_score >= 80:
            badge = "Sentinel"
        elif conf_score >= 40:
            badge = "Guardian"
        elif conf_score >= 10:
            badge = "Scout"
        else:
            badge = "Rookie"
        rows.append({
            "id": sid,
            "name": s.get("name", "Student"),
            "reports": r_count,
            "confidence_score": conf_score,
            "badge": badge,
            "avatar_url": s.get("avatar_url") or s.get("avatar") or s.get("profile_pic"),
        })

    # Primary sort by confidence_score, secondary by reports count
    rows.sort(key=lambda r: (r["confidence_score"], r["reports"]), reverse=True)

    my_id = user["_id"]
    my_rank = next((i + 1 for i, r in enumerate(rows) if r["id"] == my_id), None)
    my_conf = round(float(user.get("confidence_score", 0.0)), 1)

    return {
        "leaderboard": rows[:10],
        "my_reports": counts.get(my_id, 0),
        "my_confidence_score": my_conf,
        "my_rank": my_rank,
    }


@app.get("/api/complaints")
def all_complaints(user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    refresh_days_open(db)
    complaints = list(db["complaints"].find().sort("priority_score", -1))
    results = []
    for complaint in complaints:
        student = db["users"].find_one({"_id": complaint["student_id"]}) if not complaint.get("anonymous") else None
        results.append(serialize_complaint(complaint, include_student=True, student=student))
    return results


@app.post("/api/complaints/{complaint_id}/start")
def start_complaint(complaint_id: int, user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    complaint = db["complaints"].find_one({"id": complaint_id})
    if not complaint:
        raise HTTPException(404, "Not found")
    db["complaints"].update_one({"_id": complaint["_id"]}, {"$set": {"status": "in_progress", "updated_at": datetime.datetime.utcnow()}})
    complaint = db["complaints"].find_one({"_id": complaint["_id"]})
    if not complaint.get("anonymous"):
        notify(db, complaint["student_id"], complaint_id, f"Your report \"{complaint['description'][:60]}\" is now In Progress.")
    return serialize_complaint(complaint)


@app.post("/api/complaints/{complaint_id}/draft")
def draft(complaint_id: int, user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    complaint = db["complaints"].find_one({"id": complaint_id})
    if not complaint:
        raise HTTPException(404, "Not found")
    student = db["users"].find_one({"_id": complaint["student_id"]})
    message = draft_resolution_message(student["name"], complaint.get("category"), complaint["description"])
    db["complaints"].update_one({"_id": complaint["_id"]}, {"$set": {"draft_message": message, "admin_approved": False, "updated_at": datetime.datetime.utcnow()}})
    return {"draft_message": message}


@app.post("/api/complaints/{complaint_id}/resolve")
def resolve(complaint_id: int, data: ResolveIn, user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    complaint = db["complaints"].find_one({"id": complaint_id})
    if not complaint:
        raise HTTPException(404, "Not found")
    if not data.approved:
        raise HTTPException(400, "Resolution message was not approved; complaint left open")

    student = db["users"].find_one({"_id": complaint["student_id"]})
    student_name = student.get("name", "Student") if student else "Student"
    student_email = student.get("email") if student else None

    message = data.message or complaint.get("draft_message") or draft_resolution_message(student_name, complaint.get("category"), complaint["description"])
    db["complaints"].update_one(
        {"_id": complaint["_id"]},
        {"$set": {"status": "resolved", "admin_approved": True, "draft_message": message, "resolved_at": datetime.datetime.utcnow(), "updated_at": datetime.datetime.utcnow()}},
    )

    # Increase student confidence score automatically on resolution
    if student:
        current_conf = float(student.get("confidence_score", 0.0))
        new_conf = min(200.0, current_conf + 5.0)
        db["users"].update_one({"_id": student["_id"]}, {"$set": {"confidence_score": new_conf}})
        student["confidence_score"] = new_conf

    email_record = None
    if student_email and not complaint.get("anonymous"):
        email_body = f"Hi {student_name},\n\nYour complaint regarding '{complaint['description']}' has been reviewed and resolved by the administrative team.\n\nResolution Summary:\n{message}\n\nYour Student Confidence Score has increased to {student.get('confidence_score', 80.0)}%. Thank you for helping keep our campus running smoothly.\n\nBest regards,\nCompass Administration"
        email_record = send_email(
            to=student_email,
            subject=f"Compass: Your complaint #{complaint_id} has been resolved",
            body=email_body,
        )

    resolved = db["complaints"].find_one({"_id": complaint["_id"]})
    if not complaint.get("anonymous") and student:
        notify(db, student["_id"], complaint_id, f"Your report \"{complaint['description'][:60]}\" was marked Resolved.")
    return {"complaint": serialize_complaint(resolved, include_student=True, student=student), "email": email_record}


@app.delete("/api/complaints/{complaint_id}")
def delete_complaint(complaint_id: int, user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    complaint = db["complaints"].find_one({"id": complaint_id})
    if not complaint:
        raise HTTPException(404, "Complaint not found")
    db["complaints"].delete_one({"_id": complaint["_id"]})
    return {"message": f"Complaint #{complaint_id} deleted successfully."}


def make_watch_list(complaints: list[dict], *, weeks_back: int = 4) -> list[dict]:
    if not complaints:
        return []

    now = datetime.datetime.utcnow()
    buckets = {"category": {}, "location": {}}
    for complaint in complaints:
        created = complaint.get("created_at")
        if not isinstance(created, datetime.datetime):
            continue
        age_days = (now - created).days
        if age_days > weeks_back * 7:
            continue
        category = complaint.get("category") or "other"
        location = complaint.get("location") or "unknown"
        buckets["category"][category] = buckets["category"].get(category, 0) + 1
        buckets["location"][location] = buckets["location"].get(location, 0) + 1

    watch_list = []
    for label, values in buckets.items():
        for item, count in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
            watch_list.append({"type": label, "name": item, "count": count, "trend": "up" if count > 0 else "stable"})
    return watch_list[:8]


def build_admin_summary(db):
    complaints = list(db["complaints"].find())
    by_status = {"submitted": 0, "in_progress": 0, "resolved": 0}
    by_category = {}
    for complaint in complaints:
        status = complaint.get("status")
        if status in by_status:
            by_status[status] += 1
        category = complaint.get("category") or "uncategorized"
        by_category[category] = by_category.get(category, 0) + 1
    top_locations = {}
    for complaint in complaints:
        location = complaint.get("location") or "unknown"
        top_locations[location] = top_locations.get(location, 0) + 1
    top_location = max(top_locations, key=top_locations.get) if top_locations else None
    return {
        "total_complaints": len(complaints),
        "by_status": by_status,
        "by_category": by_category,
        "most_reported_location": top_location,
        "avg_priority": round(sum(float(c.get("priority_score") or 0) for c in complaints) / len(complaints), 1) if complaints else 0,
        "watch_list": make_watch_list(complaints),
    }


@app.get("/api/admin/summary")
def summary(user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    return build_admin_summary(db)


def send_weekly_summary():
    db = get_db()
    admin_users = list(db["users"].find({"role": "admin"}))
    summary = build_admin_summary(db)
    body = (
        "Compass weekly summary\n\n"
        f"Total complaints: {summary['total_complaints']}\n"
        f"Submitted: {summary['by_status'].get('submitted', 0)}\n"
        f"In progress: {summary['by_status'].get('in_progress', 0)}\n"
        f"Resolved: {summary['by_status'].get('resolved', 0)}\n"
        f"Most reported location: {summary.get('most_reported_location') or 'n/a'}\n"
        f"Average priority: {summary.get('avg_priority', 0)}\n"
    )
    for user in admin_users:
        send_email(user.get("email"), "Compass weekly summary", body)
    return summary


@app.post("/api/admin/summary/send-now")
def send_summary_now(user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    summary = send_weekly_summary()
    return {"message": "Weekly summary sent to admins.", "summary": summary}


@app.post("/api/admin/query")
def admin_query(data: AdminQueryIn, user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    query = (data.query or "").strip()
    if not query:
        raise HTTPException(400, "Query is required")
    complaints = list(db["complaints"].find())
    filtered = []
    if query:
        q = query.lower()
        for complaint in complaints:
            haystack = " ".join([
                complaint.get("description") or "",
                complaint.get("location") or "",
                complaint.get("category") or "",
                complaint.get("status") or "",
            ]).lower()
            if q in haystack or any(token in haystack for token in q.split() if len(token) > 2):
                filtered.append(complaint)
        if not filtered:
            filtered = complaints
    answer = answer_admin_query(query, [serialize_complaint(c) for c in filtered])
    return {
        "query": query,
        "count": len(filtered),
        "matches": [serialize_complaint(c) for c in filtered[:10]],
        "answer": answer,
    }


@app.post("/api/admin/chatbot")
def admin_chatbot(data: ChatIn, user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    message = (data.message or "").strip()
    if not message:
        raise HTTPException(400, "message is required")
    complaints = [serialize_complaint(c) for c in db["complaints"].find()]
    return chat_answer(message, complaints, history=data.history)


@app.get("/api/notifications")
def list_notifications(user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "student")
    items = list(db["notifications"].find({"student_id": user["_id"]}).sort("created_at", -1))
    return [serialize_notification(n) for n in items[:30]]


@app.post("/api/notifications/read-all")
def read_all_notifications(user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "student")
    db["notifications"].update_many({"student_id": user["_id"], "read": False}, {"$set": {"read": True}})
    return {"message": "All notifications marked as read"}


@app.post("/api/complaints/{complaint_id}/feedback")
def submit_feedback(complaint_id: int, data: FeedbackIn, user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "student")
    complaint = db["complaints"].find_one({"id": complaint_id})
    if not complaint or complaint.get("student_id") != user["_id"]:
        raise HTTPException(404, "Not found")
    if complaint.get("status") != "resolved":
        raise HTTPException(400, "You can only leave feedback once a complaint is resolved")
    if data.rating < 1 or data.rating > 5:
        raise HTTPException(400, "Rating must be between 1 and 5")
    db["complaints"].update_one(
        {"_id": complaint["_id"]},
        {"$set": {"feedback_rating": data.rating, "feedback_comment": (data.comment or "").strip(),
                   "feedback_at": datetime.datetime.utcnow()}},
    )
    return {"message": "Thanks for the feedback!"}


@app.get("/api/community/feed")
def community_feed(user: dict = Depends(get_current_user), db=Depends(get_db)):
    """Anonymized recent-activity feed — returns real submitted complaints."""
    complaints = list(db["complaints"].find().sort("created_at", -1))
    feed = []
    for c in complaints:
        created_at = c.get("created_at")
        resolved_at = c.get("resolved_at")
        student = db["users"].find_one({"_id": c.get("student_id")}) if not c.get("anonymous") else None
        student_name = "Anonymous" if c.get("anonymous") else (student.get("name", "Student") if student else "Student")
        feed.append({
            "id": c.get("id") or c.get("_id"),
            "title": c.get("title") or (c.get("description", "")[:50] + ("..." if len(c.get("description", "")) > 50 else "")),
            "description": c.get("description") or "",
            "location": c.get("location") or "Campus",
            "category": c.get("category") or "other",
            "category_source": c.get("category_source") or "text",
            "department": c.get("department") or infer_department(c.get("category"), c.get("description")),
            "status": c.get("status"),
            "urgency": round(float(c.get("urgency") or 0), 2),
            "frequency": c.get("frequency") or 1,
            "days_open": c.get("days_open") or 0,
            "priority_score": round(float(c.get("priority_score") or 0), 1),
            "photo_path": c.get("photo_path"),
            "anonymous": bool(c.get("anonymous")),
            "student_name": student_name,
            "created_at": format_iso_utc(created_at),
            "resolved_at": format_iso_utc(resolved_at),
        })
    return feed


@app.get("/api/leaderboard")
def leaderboard(user: dict = Depends(get_current_user), db=Depends(get_db)):
    complaints = list(db["complaints"].find())
    counts: dict = {}
    for c in complaints:
        sid = c.get("student_id")
        counts[sid] = counts.get(sid, 0) + 1

    rows = []
    for sid, count in counts.items():
        student = db["users"].find_one({"_id": sid})
        if not student:
            continue
        badge = "Newcomer"
        if count >= 10:
            badge = "Problem Solver"
        elif count >= 5:
            badge = "Active Reporter"
        elif count >= 1:
            badge = "First Reporter"
        rows.append({"name": student.get("name"), "reports": count, "badge": badge})

    rows.sort(key=lambda r: r["reports"], reverse=True)
    is_me = user["_id"]
    my_count = counts.get(is_me, 0)
    my_rank = next((i + 1 for i, r in enumerate(rows) if r["name"] == user.get("name") and r["reports"] == my_count), None)
    return {"leaderboard": rows[:10], "my_reports": my_count, "my_rank": my_rank}


@app.get("/api/admin/run-forecast")
def run_forecast(user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "admin")
    complaints = list(db["complaints"].find())
    result = run_predictive_pipeline(complaints, location=None, months_ahead=2, top_n=5)
    return {"location": "Campus-wide", "predictions": result.get("predictions", [])}


@app.get("/api/predictions/nearby")
def predictions_nearby(location: Optional[str] = None, user: dict = Depends(get_current_user), db=Depends(get_db)):
    complaints = list(db["complaints"].find())
    predictions = predict_upcoming_issues(complaints, location=location)
    return {"location": location or "Campus-wide", "predictions": predictions}


@app.get("/api/profile")
def get_profile(user: dict = Depends(get_current_user)):
    return {
        "id": user["_id"],
        "name": user.get("name"),
        "email": user.get("email"),
        "role": user.get("role"),
        "block": user.get("block"),
        "email_notifications": user.get("email_notifications", True),
        "streak_count": user.get("streak_count") or 0,
    }


@app.put("/api/profile")
def update_profile(data: ProfileUpdateIn, user: dict = Depends(get_current_user), db=Depends(get_db)):
    updates = {}
    if data.name:
        updates["name"] = data.name.strip()
    if data.block is not None:
        updates["block"] = data.block.strip()
    if data.email_notifications is not None:
        updates["email_notifications"] = data.email_notifications
    if updates:
        db["users"].update_one({"_id": user["_id"]}, {"$set": updates})
    updated = db["users"].find_one({"_id": user["_id"]})
    return {
        "id": updated["_id"], "name": updated.get("name"), "email": updated.get("email"),
        "block": updated.get("block"), "email_notifications": updated.get("email_notifications", True),
    }


def refresh_days_open(db):
    now = datetime.datetime.utcnow()
    for complaint in db["complaints"].find({"status": {"$ne": "resolved"}}):
        new_days = (now - complaint["created_at"]).days
        if new_days != complaint.get("days_open"):
            db["complaints"].update_one(
                {"_id": complaint["_id"]},
                {"$set": {"days_open": new_days, "priority_score": compute_priority(complaint["urgency"], complaint.get("frequency") or 1, new_days), "updated_at": now}},
            )


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")