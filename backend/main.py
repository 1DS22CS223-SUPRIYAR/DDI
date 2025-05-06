from fastapi import FastAPI
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000",  # Also include this just in case
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can specify more specific domains here if needed, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

@app.get("/drugs")
async def get_drugs():
    # Read CSV
    df = pd.read_csv("./resources/drugs.csv")
    df = df[['DrugBank ID', 'Drug Name']]
    # Convert to list of dicts: [{"id":1, "name":"Aspirin"}, ...]
    drugs_dict = dict(zip(df['DrugBank ID'], df['Drug Name']))
    
    return {"drugs": drugs_dict}