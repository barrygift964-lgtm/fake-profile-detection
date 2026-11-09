"""Create the administrator account and (optionally) train the first model.

Examples:
  python seed.py --email admin@fpd.local --password Admin12345
  python seed.py --email admin@fpd.local --password Admin12345 --dataset data/your_dataset.csv
"""
import argparse
import os
import shutil

from werkzeug.security import generate_password_hash

import database as db
import ml_engine as ml
from app import app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="System Administrator")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--dataset", help="CSV file to upload and train on")
    p.add_argument("--trees", type=int, default=100)
    a = p.parse_args()

    with app.app_context():
        email = a.email.strip().lower()
        user = db.query("SELECT * FROM users WHERE email = ?", (email,), one=True)
        if user:
            db.execute("UPDATE users SET role='admin', is_active=1, failed_logins=0, password_hash=? "
                       "WHERE user_id=?", (generate_password_hash(a.password), user["user_id"]))
            uid = user["user_id"]
            print(f"Updated existing user {email} as administrator.")
        else:
            uid = db.execute("INSERT INTO users (full_name, email, password_hash, role) VALUES (?,?,?,'admin')",
                             (a.name, email, generate_password_hash(a.password)))
            print(f"Created administrator {email}.")

        if a.dataset:
            media = app.config["MEDIA_DIR"]
            dest = os.path.join(media, "datasets", os.path.basename(a.dataset))
            if os.path.abspath(a.dataset) != os.path.abspath(dest):
                shutil.copy(a.dataset, dest)
            total, genuine, fake = ml.count_classes(dest)
            ds_id = db.execute("INSERT INTO datasets (uploaded_by, file_name, file_path, total_records, "
                               "genuine_count, fake_count) VALUES (?,?,?,?,?,?)",
                               (uid, os.path.basename(dest), dest, total, genuine, fake))
            model_path = os.path.join(media, "models", "rf_model.pkl")
            m = ml.train_and_test(dest, model_path, os.path.join(media, "charts"), n_trees=a.trees)
            import json
            db.execute("UPDATE trained_models SET is_active = 0")
            db.execute("INSERT INTO trained_models (dataset_id, model_path, n_estimators, train_size, test_size, "
                       "accuracy, precision_score, recall_score, f1_score, error_rate, tp, tn, fp, fn, importances) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (ds_id, model_path, a.trees, m["train_size"], m["test_size"], m["accuracy"],
                        m["precision"], m["recall"], m["f1"], m["error_rate"], m["tp"], m["tn"], m["fp"], m["fn"],
                        json.dumps(m["importances"])))
            print(f"Trained on {total} records ({genuine} genuine, {fake} fake). "
                  f"Accuracy {m['accuracy']}%, precision {m['precision']}%, recall {m['recall']}%, "
                  f"F1 {m['f1']}%, error rate {m['error_rate']}%.")
            print(f"Confusion matrix: TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")


if __name__ == "__main__":
    main()
