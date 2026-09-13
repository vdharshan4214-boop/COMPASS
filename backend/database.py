"""MongoDB configuration for Compass.
The application expects a MongoDB connection string from the environment, but the
app also needs to run in a no-Mongo local demo environment. We therefore keep the
real pymongo path as the default and gracefully fall back to a small in-memory
Mongo-compatible implementation when there is no reachable server.
"""
import csv
import datetime
import os
from dotenv import load_dotenv
from pymongo import MongoClient, ReturnDocument

from auth import hash_password

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/compass")


class _MemoryCursor:
    def __init__(self, docs, query=None):
        self._docs = docs
        self._query = query or {}
        self._sort = None

    def _matches(self, doc):
        for key, expected in self._query.items():
            if key == "$ne":
                continue
            if key not in doc:
                return False
            if isinstance(expected, dict):
                for op, value in expected.items():
                    if op == "$ne" and doc.get(key) == value:
                        return False
                    if op == "$in" and doc.get(key) not in value:
                        return False
                    if op == "$gt" and not (doc.get(key, None) is not None and doc.get(key) > value):
                        return False
                    if op == "$lt" and not (doc.get(key, None) is not None and doc.get(key) < value):
                        return False
                    if op == "$gte" and not (doc.get(key, None) is not None and doc.get(key) >= value):
                        return False
                    if op == "$lte" and not (doc.get(key, None) is not None and doc.get(key) <= value):
                        return False
            elif doc.get(key) != expected:
                return False
        return True

    def sort(self, field, direction=-1):
        self._sort = (field, direction)
        return self

    def __iter__(self):
        docs = [d for d in self._docs if self._matches(d)]
        if self._sort:
            field, direction = self._sort
            docs.sort(key=lambda d: d.get(field) if d.get(field) is not None else 0, reverse=(direction == -1))
        return iter(docs)

    def __len__(self):
        return len(list(self.__iter__()))


class _MemoryCollection:
    def __init__(self, name, store):
        self.name = name
        self.store = store

    def _docs(self):
        return self.store.setdefault(self.name, [])

    def insert_one(self, doc):
        self._docs().append(doc)
        return type("InsertResult", (), {"inserted_id": doc.get("_id")})()

    def find_one(self, query=None):
        query = query or {}
        for doc in self._docs():
            ok = True
            for key, expected in query.items():
                if isinstance(expected, dict):
                    if "$ne" in expected and doc.get(key) == expected["$ne"]:
                        ok = False
                    elif "$in" in expected and doc.get(key) not in expected["$in"]:
                        ok = False
                elif doc.get(key) != expected:
                    ok = False
            if ok:
                return doc
        return None

    def find(self, query=None):
        return _MemoryCursor(self._docs(), query)

    def find_one_and_update(self, filter_, update, upsert=False, return_document=None):
        docs = self._docs()
        doc = self.find_one(filter_)
        if doc is None:
            if upsert:
                new_doc = dict(filter_)
                if "$set" in update:
                    new_doc.update(update["$set"])
                if "$inc" in update:
                    for key, value in update["$inc"].items():
                        new_doc[key] = value
                documents = docs
                documents.append(new_doc)
                if return_document == "after":
                    return new_doc
                return new_doc
            return None

        if "$set" in update:
            for key, value in update["$set"].items():
                doc[key] = value
        if "$inc" in update:
            for key, value in update["$inc"].items():
                doc[key] = doc.get(key, 0) + value
        return doc

    def update_one(self, filter_, update):
        doc = self.find_one(filter_)
        if doc is None:
            return type("UpdateResult", (), {"matched_count": 0, "modified_count": 0})()
        if "$set" in update:
            for key, value in update["$set"].items():
                doc[key] = value
        if "$inc" in update:
            for key, value in update["$inc"].items():
                doc[key] = doc.get(key, 0) + value
        return type("UpdateResult", (), {"matched_count": 1, "modified_count": 1})()

    def update_many(self, filter_, update):
        for doc in self._docs():
            if all(doc.get(k) == v for k, v in filter_.items()):
                self.update_one(filter_, update)
        return type("UpdateResult", (), {"matched_count": 1, "modified_count": 1})()

    def delete_one(self, filter_):
        doc = self.find_one(filter_)
        if doc is not None:
            self._docs().remove(doc)
            return type("DeleteResult", (), {"deleted_count": 1})()
        return type("DeleteResult", (), {"deleted_count": 0})()

    def delete_many(self, filter_):
        removed = 0
        docs_to_remove = [doc for doc in self._docs() if all(doc.get(k) == v for k, v in filter_.items())]
        for doc in docs_to_remove:
            self._docs().remove(doc)
            removed += 1
        return type("DeleteResult", (), {"deleted_count": removed})()


class _MemoryDatabase:
    def __init__(self):
        self._collections = {}

    def __getitem__(self, name):
        return _MemoryCollection(name, self._collections)


class _FallbackClient:
    def __init__(self):
        self._db = _MemoryDatabase()

    def get_default_database(self):
        return self._db

    def __getitem__(self, name):
        return self._db[name]


def _build_client():
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=1500, connectTimeoutMS=1500)
        client.admin.command("ping")
        return client
    except Exception:
        return _FallbackClient()


try:
    client = _build_client()
    db = client.get_default_database() if hasattr(client, "get_default_database") else client["compass"]
except Exception:
    client = _FallbackClient()
    db = client.get_default_database()


def get_db():
    return db


def get_users_collection():
    return db["users"]


def get_complaints_collection():
    return db["complaints"]


def next_id(collection_name: str) -> int:
    counters = db["counters"]
    counter = counters.find_one_and_update(
        {"_id": collection_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(counter["seq"])


def seed_demo_data():
    """Populate the local demo database with default admin & student accounts when empty."""
    users = db["users"]
    if users.find_one({"email": "admin@compass.local"}) is None:
        admin_id = next_id("users")
        users.insert_one({
            "_id": admin_id,
            "id": admin_id,
            "name": "Compass Admin",
            "email": "admin@compass.local",
            "password_hash": hash_password("admin123"),
            "role": "admin",
            "streak_count": 0,
            "last_active_date": None,
        })

    if users.find_one({"email": "student@compass.local"}) is None:
        student_id = next_id("users")
        users.insert_one({
            "_id": student_id,
            "id": student_id,
            "name": "Demo Student",
            "email": "student@compass.local",
            "password_hash": hash_password("student123"),
            "role": "student",
            "streak_count": 5,
            "confidence_score": 82.5,
            "last_active_date": None,
        })

