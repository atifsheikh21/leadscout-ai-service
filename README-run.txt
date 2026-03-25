Run locally (Windows):

1) Create venv:
   py -m venv .venv

2) Activate:
   .venv\Scripts\activate

3) Install:
   pip install -r requirements.txt

4) Start server:
   uvicorn app:app --host 127.0.0.1 --port 8008

Laravel should have:
AI_IMAGE_SERVICE_URL=http://127.0.0.1:8008

Test:
GET http://127.0.0.1:8008/health
