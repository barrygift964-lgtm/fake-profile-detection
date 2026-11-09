"""End-to-end tests of the Fake Profile Detection System.  Run:  python -m pytest -q"""
import io
import os
import re
import shutil
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import create_app  # noqa: E402
from werkzeug.security import generate_password_hash  # noqa: E402
import database as db  # noqa: E402

SAMPLE = os.path.join(ROOT, "data", "sample_profiles_DEMO.csv")
FAKE = dict(account_name="@daily_promo_24", statuses_count=35, followers_count=1, friends_count=2001,
            favourites_count=0, listed_count=0, FFratio=0.0005, URLratio=0.92, average_mentions=0.04,
            average_hashtags=0.12, average_favorites=0.01, average_retweet=0.85, average_reply=0.01)
GENUINE = dict(account_name="@ada_okoro", statuses_count=4210, followers_count=612, friends_count=498,
               favourites_count=1880, listed_count=14, FFratio=1.229, URLratio=0.18, average_mentions=0.71,
               average_hashtags=0.66, average_favorites=0.85, average_retweet=0.20, average_reply=0.42)


@pytest.fixture()
def app():
    tmp = tempfile.mkdtemp()
    a = create_app({"DATABASE": os.path.join(tmp, "t.sqlite3"), "MEDIA_DIR": os.path.join(tmp, "media"),
                    "TESTING": True})
    for sub in ("datasets", "models", "charts"):
        os.makedirs(os.path.join(tmp, "media", sub), exist_ok=True)
    with a.app_context():
        db.init_db()
        db.execute("INSERT INTO users (full_name,email,password_hash,role) VALUES (?,?,?,'admin')",
                   ("Admin", "admin@test.com", generate_password_hash("Admin12345")))
    yield a
    shutil.rmtree(tmp)


def token(client, path):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="_csrf" value="([0-9a-f]+)"', html).group(1)


def login(client, email, pw):
    return client.post("/login", data={"email": email, "password": pw, "_csrf": token(client, "/login")},
                       follow_redirects=True)


def upload_and_train(client):
    t = token(client, "/upload")
    with open(SAMPLE, "rb") as f:
        r = client.post("/upload", data={"dataset": (f, "sample.csv"), "_csrf": t},
                        content_type="multipart/form-data", follow_redirects=True)
    assert b"File uploaded: 2,798 records" in r.data
    r = client.post("/train", data={"n_trees": "100", "_csrf": token(client, "/upload")}, follow_redirects=True)
    assert b"Training complete" in r.data


def test_register_login_logout(app):
    c = app.test_client()
    t = token(c, "/register")
    r = c.post("/register", data={"full_name": "Ada Okoro", "email": "ada@test.com", "password": "secret123",
                                  "confirm_password": "secret123", "_csrf": t}, follow_redirects=True)
    assert b"Registration successful" in r.data
    t = token(c, "/register")
    r = c.post("/register", data={"full_name": "Ada", "email": "ada@test.com", "password": "secret123",
                                  "confirm_password": "secret123", "_csrf": t}, follow_redirects=True)
    assert b"Email already registered" in r.data
    t = token(c, "/register")
    r = c.post("/register", data={"full_name": "Ben Obi", "email": "ben@test.com", "password": "secret123",
                                  "confirm_password": "other1234", "_csrf": t}, follow_redirects=True)
    assert b"Passwords do not match" in r.data
    assert b"Invalid email or password" in login(c, "ada@test.com", "wrongpass1").data
    r = login(c, "ada@test.com", "secret123")
    assert b"Logout" in r.data
    r = c.get("/logout", follow_redirects=True)
    assert b"You have been logged out" in r.data


def test_access_control_and_csrf(app):
    c = app.test_client()
    assert c.get("/upload").status_code == 302                       # not logged in
    assert c.post("/login", data={"email": "x", "password": "y"}).status_code == 403   # no CSRF token
    t = token(c, "/register")
    c.post("/register", data={"full_name": "Ada Okoro", "email": "ada@test.com", "password": "secret123",
                              "confirm_password": "secret123", "_csrf": t})
    login(c, "ada@test.com", "secret123")
    r = c.get("/upload", follow_redirects=True)
    assert b"Access denied" in r.data                                 # normal user cannot upload


def test_upload_validation(app):
    c = app.test_client()
    login(c, "admin@test.com", "Admin12345")
    r = c.post("/upload", data={"dataset": (io.BytesIO(b"hello"), "notes.docx"), "_csrf": token(c, "/upload")},
               content_type="multipart/form-data", follow_redirects=True)
    assert b"Only CSV files are allowed" in r.data
    r = c.post("/upload", data={"dataset": (io.BytesIO(b"a,b\n1,2\n"), "bad.csv"), "_csrf": token(c, "/upload")},
               content_type="multipart/form-data", follow_redirects=True)
    assert b"Missing columns" in r.data


def test_detect_before_training(app):
    c = app.test_client()
    login(c, "admin@test.com", "Admin12345")
    r = c.post("/detect", data={**FAKE, "_csrf": token(c, "/detect")}, follow_redirects=True)
    assert b"No trained model available" in r.data


def test_full_flow_train_detect_dashboard(app):
    c = app.test_client()
    login(c, "admin@test.com", "Admin12345")
    upload_and_train(c)
    r = c.post("/detect", data={**FAKE, "_csrf": token(c, "/detect")})
    assert b"is a FAKE PROFILE" in r.data
    r = c.post("/detect", data={**GENUINE, "_csrf": token(c, "/detect")})
    assert b"is a GENUINE PROFILE" in r.data
    bad = {**GENUINE, "followers_count": "-5", "URLratio": "3"}
    r = c.post("/detect", data={**bad, "_csrf": token(c, "/detect")})
    assert b"Enter a valid number of zero or more" in r.data and b"between 0 and 1" in r.data
    r = c.get("/history")
    assert b"@daily_promo_24" in r.data and b"@ada_okoro" in r.data
    r = c.get("/performance")
    for word in (b"Accuracy", b"Precision", b"Recall", b"F1-score", b"Error rate", b"Confusion Matrix",
                 b"Feature Importance", b"Detections Made"):
        assert word in r.data
    assert c.get("/users").status_code == 200


def test_xss_escaped(app):
    c = app.test_client()
    login(c, "admin@test.com", "Admin12345")
    upload_and_train(c)
    r = c.post("/detect", data={**GENUINE, "account_name": "<script>alert(1)</script>", "_csrf": token(c, "/detect")})
    assert b"<script>alert(1)</script>" not in r.data and b"&lt;script&gt;" in r.data
