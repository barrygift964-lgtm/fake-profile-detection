"""Fake Profile Detection System for the X Social Media Platform (Random Forest).

Run:  python app.py      then open http://127.0.0.1:5000
"""
import json
import os
import secrets
from functools import wraps

from flask import (Flask, abort, flash, g, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import database as db
import ml_engine as ml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_FAILED_LOGINS = 5


def create_app(test_config=None):
    app = Flask(__name__, instance_path=os.path.join(BASE_DIR, "instance"))
    app.config.update(
        SECRET_KEY=os.environ.get("FPD_SECRET_KEY") or _load_or_create_secret(app.instance_path),
        DATABASE=os.path.join(app.instance_path, "fpd.sqlite3"),
        MEDIA_DIR=os.path.join(BASE_DIR, "static", "media"),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,           # 10 MB upload limit
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)
    os.makedirs(app.instance_path, exist_ok=True)
    for sub in ("datasets", "models", "charts"):
        os.makedirs(os.path.join(app.config["MEDIA_DIR"], sub), exist_ok=True)

    app.teardown_appcontext(db.close_db)
    with app.app_context():
        db.init_db()
    _register_routes(app)
    return app


def _load_or_create_secret(instance_path):
    os.makedirs(instance_path, exist_ok=True)
    path = os.path.join(instance_path, "secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


# ---------------------------------------------------------------- helpers
def current_user():
    uid = session.get("user_id")
    if uid is None:
        return None
    return db.query("SELECT * FROM users WHERE user_id = ? AND is_active = 1", (uid,), one=True)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("login", next=request.path))
        if g.user["role"] != "admin":
            flash("Access denied: only the administrator can use this page.", "error")
            return redirect(url_for("home"))
        return view(*args, **kwargs)
    return wrapped


def active_model():
    return db.query("SELECT * FROM trained_models WHERE is_active = 1 "
                    "ORDER BY model_id DESC LIMIT 1", one=True)


def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


# ---------------------------------------------------------------- routes
def _register_routes(app):

    @app.before_request
    def load_user_and_check_csrf():
        g.user = current_user()
        if request.method == "POST" and not app.config.get("TESTING_NO_CSRF"):
            sent = request.form.get("_csrf")
            if not sent or sent != session.get("_csrf"):
                abort(403)

    @app.context_processor
    def inject():
        return {"user": g.get("user"), "csrf_token": csrf_token}

    @app.route("/")
    def home():
        return render_template("home.html", model=active_model())

    @app.route("/register", methods=["GET", "POST"])
    def register():
        form = {}
        if request.method == "POST":
            form = request.form
            name = form.get("full_name", "").strip()
            email = form.get("email", "").strip().lower()
            pw, pw2 = form.get("password", ""), form.get("confirm_password", "")
            error = None
            if len(name) < 3:
                error = "Enter your full name (at least 3 characters)."
            elif "@" not in email or "." not in email.split("@")[-1]:
                error = "Enter a valid email address."
            elif len(pw) < 8 or pw.isalpha() or pw.isdigit():
                error = "Password must be at least 8 characters and contain letters and numbers."
            elif pw != pw2:
                error = "Passwords do not match."
            elif db.query("SELECT 1 FROM users WHERE email = ?", (email,), one=True):
                error = "Email already registered."
            if error:
                flash(error, "error")
            else:
                db.execute("INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                           (name, email, generate_password_hash(pw)))
                flash("Registration successful. Please log in.", "success")
                return redirect(url_for("login"))
        return render_template("register.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            user = db.query("SELECT * FROM users WHERE email = ?", (email,), one=True)
            if user and user["failed_logins"] >= MAX_FAILED_LOGINS:
                flash("Account locked after too many failed attempts. Contact the administrator.", "error")
            elif user and user["is_active"] and check_password_hash(user["password_hash"],
                                                                    request.form.get("password", "")):
                db.execute("UPDATE users SET failed_logins = 0 WHERE user_id = ?", (user["user_id"],))
                session.clear()
                session["user_id"] = user["user_id"]
                session.permanent = bool(request.form.get("keep_signed_in"))
                nxt = request.args.get("next", "")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//")
                                else url_for("dashboard"))
            else:
                if user:
                    db.execute("UPDATE users SET failed_logins = failed_logins + 1 WHERE user_id = ?",
                               (user["user_id"],))
                flash("Invalid email or password.", "error")
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "success")
        return redirect(url_for("login"))

    # ------------------------------------------------------- upload / train
    @app.route("/upload", methods=["GET", "POST"])
    @admin_required
    def upload():
        if request.method == "POST":
            f = request.files.get("dataset")
            if not f or not f.filename:
                flash("Choose a CSV file first (click Browse).", "error")
            elif not f.filename.lower().endswith(".csv"):
                flash("Only CSV files are allowed.", "error")
            else:
                name = secure_filename(f.filename)
                path = os.path.join(app.config["MEDIA_DIR"], "datasets", name)
                f.save(path)
                try:
                    total, genuine, fake = ml.count_classes(path)
                except ml.DatasetError as err:
                    os.remove(path)
                    flash(str(err), "error")
                else:
                    db.execute("INSERT INTO datasets (uploaded_by, file_name, file_path, total_records, "
                               "genuine_count, fake_count) VALUES (?, ?, ?, ?, ?, ?)",
                               (g.user["user_id"], name, path, total, genuine, fake))
                    flash(f"File uploaded: {total:,} records ({genuine:,} genuine, {fake:,} fake).",
                          "success")
            return redirect(url_for("upload"))
        latest = db.query("SELECT * FROM datasets ORDER BY dataset_id DESC LIMIT 1", one=True)
        return render_template("upload.html", latest=latest, model=active_model())

    @app.route("/train", methods=["POST"])
    @admin_required
    def train():
        ds = db.query("SELECT * FROM datasets ORDER BY dataset_id DESC LIMIT 1", one=True)
        if ds is None:
            flash("Upload a dataset before clicking Train|Test.", "error")
            return redirect(url_for("upload"))
        trees = int(request.form.get("n_trees", 100) or 100)
        trees = max(10, min(trees, 500))
        model_path = os.path.join(app.config["MEDIA_DIR"], "models", "rf_model.pkl")
        m = ml.train_and_test(ds["file_path"], model_path,
                              os.path.join(app.config["MEDIA_DIR"], "charts"), n_trees=trees)
        db.execute("UPDATE trained_models SET is_active = 0")
        db.execute("INSERT INTO trained_models (dataset_id, model_path, n_estimators, train_size, "
                   "test_size, accuracy, precision_score, recall_score, f1_score, error_rate, "
                   "tp, tn, fp, fn, importances) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (ds["dataset_id"], model_path, trees, m["train_size"], m["test_size"],
                    m["accuracy"], m["precision"], m["recall"], m["f1"], m["error_rate"],
                    m["tp"], m["tn"], m["fp"], m["fn"], json.dumps(m["importances"])))
        flash(f"Training complete: {m['train_size']:,} records | Testing: {m['test_size']:,} records "
              f"| Accuracy: {m['accuracy']}%", "success")
        return redirect(url_for("upload"))

    # ------------------------------------------------------- detection
    @app.route("/detect", methods=["GET", "POST"])
    @login_required
    def detect():
        values, errors, result = {}, {}, None
        model = active_model()
        if request.method == "POST":
            values = {k: request.form.get(k, "").strip() for k in ["account_name"] + ml.FEATURES}
            clean = {}
            for f in ml.FEATURES:
                try:
                    v = int(values[f]) if f in ml.INT_FEATURES else float(values[f])
                    if v < 0 or (f == "URLratio" and v > 1):
                        raise ValueError
                    clean[f] = v
                except ValueError:
                    errors[f] = ("Enter a value between 0 and 1." if f == "URLratio"
                                 else "Enter a valid number of zero or more.")
            if len(values["account_name"]) > 100:
                errors["account_name"] = "Maximum 100 characters."
            if model is None:
                flash("No trained model available. The administrator must upload a dataset "
                      "and click Train|Test.", "error")
            elif not errors:
                label, conf, votes, n = ml.detect(clean, model["model_path"])
                db.execute(
                    "INSERT INTO detections (user_id, model_id, account_name, statuses_count, "
                    "followers_count, friends_count, favourites_count, listed_count, ff_ratio, "
                    "url_ratio, average_mentions, average_hashtags, average_favorites, "
                    "average_retweet, average_reply, predicted_label, confidence) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (g.user["user_id"], model["model_id"], values["account_name"] or None,
                     *[clean[f] for f in ml.FEATURES], label, conf))
                result = {"name": values["account_name"] or "This account", "label": label,
                          "confidence": conf, "votes": votes, "trees": n}
        return render_template("detect.html", values=values, errors=errors, result=result,
                               features=ml.FEATURES, int_features=ml.INT_FEATURES, model=model)

    @app.route("/history")
    @login_required
    def history():
        if g.user["role"] == "admin":
            rows = db.query("SELECT d.*, u.full_name FROM detections d JOIN users u "
                            "ON u.user_id = d.user_id ORDER BY d.detection_id DESC")
        else:
            rows = db.query("SELECT d.*, u.full_name FROM detections d JOIN users u "
                            "ON u.user_id = d.user_id WHERE d.user_id = ? "
                            "ORDER BY d.detection_id DESC", (g.user["user_id"],))
        return render_template("history.html", rows=rows)

    # ------------------------------------------------------- dashboard
    @app.route("/performance")
    @login_required
    def dashboard():
        model = active_model()
        ds = None
        importances = []
        if model:
            ds = db.query("SELECT * FROM datasets WHERE dataset_id = ?", (model["dataset_id"],), one=True)
            importances = json.loads(model["importances"] or "[]")
        stats = db.query("SELECT COUNT(*) AS total, "
                         "SUM(predicted_label = 'Fake') AS fake, "
                         "SUM(predicted_label = 'Genuine') AS genuine FROM detections", one=True)
        recent = db.query("SELECT d.*, u.full_name FROM detections d JOIN users u "
                          "ON u.user_id = d.user_id ORDER BY d.detection_id DESC LIMIT 6")
        users = db.query("SELECT COUNT(*) AS n FROM users", one=True)["n"]
        return render_template("performance.html", m=model, ds=ds, importances=importances,
                               stats=stats, recent=recent, users=users,
                               existing=ml.EXISTING_SYSTEM)

    @app.route("/users", methods=["GET", "POST"])
    @admin_required
    def users():
        if request.method == "POST":
            uid = int(request.form["user_id"])
            action = request.form.get("action")
            if uid == g.user["user_id"]:
                flash("You cannot change your own account here.", "error")
            elif action == "toggle":
                row = db.query("SELECT * FROM users WHERE user_id = ?", (uid,), one=True)
                if row and row["failed_logins"] >= MAX_FAILED_LOGINS:     # unlock
                    db.execute("UPDATE users SET is_active = 1, failed_logins = 0 WHERE user_id = ?", (uid,))
                elif row:
                    db.execute("UPDATE users SET is_active = 1 - is_active WHERE user_id = ?", (uid,))
            elif action == "role":
                db.execute("UPDATE users SET role = CASE role WHEN 'admin' THEN 'user' ELSE 'admin' END "
                           "WHERE user_id = ?", (uid,))
            return redirect(url_for("users"))
        rows = db.query("SELECT u.*, (SELECT COUNT(*) FROM detections d WHERE d.user_id = u.user_id) "
                        "AS detections FROM users u ORDER BY u.user_id")
        return render_template("users.html", rows=rows)

    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("error.html", code=403,
                               message="The form has expired. Go back, refresh the page and try again."), 403

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(_e):
        flash("File is larger than 10 MB.", "error")
        return redirect(url_for("upload"))


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
