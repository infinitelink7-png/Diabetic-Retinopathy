Database setup for Diabetic-Retinopathy backend
=============================================

This project uses SQLite by default for quick local development. The Flask
app initializes and creates `backend/assessments.db` automatically via
`backend/database.py` when the app first runs.

If you prefer to create the DB and seed it manually (recommended for CI or
to avoid starting heavy model initialization during `app.py` import), use the
included script:

Create and seed locally
-----------------------

1. Activate your virtual environment and install requirements:

   Windows PowerShell:

       python -m venv .venv
       .venv\Scripts\Activate.ps1
       pip install -r requirements.txt

2. Run the seeder:

   Windows PowerShell:

       python backend/create_db_and_seed.py

This creates `backend/assessments.db` with 2 sample records.

Using on Render
---------------

- Render runs your web command (from the Procfile). The `backend/database.py`
  file configures SQLite and calls `db.create_all()` during `init_db(app)` so
  the database file will be created automatically in the deployment container
  at `backend/assessments.db`.
- If you want pre-populated data on Render, run the seeder locally and commit
  the `backend/assessments.db` file to the repository (small databases are
  fine), or upload the DB file as a deployment artifact.

Notes
-----
- Avoid importing `backend/app.py` from scripts that only need the DB, because
  `app.py` currently instantiates ML models which can be heavy. Use
  `create_db_and_seed.py` to create the DB without triggering model loading.
