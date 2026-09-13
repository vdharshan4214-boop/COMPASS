import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Date
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "student" or "admin"

    streak_count = Column(Integer, default=0)
    last_active_date = Column(Date, nullable=True)

    complaints = relationship("Complaint", back_populates="student")


class Complaint(Base):
    __tablename__ = "complaints"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    anonymous = Column(Boolean, default=False)

    description = Column(String, nullable=False)
    title = Column(String, nullable=True)
    location = Column(String, nullable=False)
    photo_path = Column(String, nullable=True)

    category = Column(String, nullable=True)
    urgency = Column(Float, default=0.0)
    frequency = Column(Integer, default=1)  # how many similar reports merged into this one
    days_open = Column(Integer, default=0)
    priority_score = Column(Float, default=0.0)

    status = Column(String, default="submitted")  # submitted -> in_progress -> resolved
    draft_message = Column(String, nullable=True)
    admin_approved = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    student = relationship("User", back_populates="complaints")
