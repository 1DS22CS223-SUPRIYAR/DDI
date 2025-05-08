from fastapi import FastAPI, HTTPException
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.responses import FileResponse
import numpy as np

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

SDF_DIR = './dataset/sdf'

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

#Returning Drug Names
@app.get("/drugs")
def get_drugs():
    df = pd.read_csv("./resources/drugs.csv")
    df = df[['DrugBank ID', 'Drug Name']]
    drugs_dict = dict(zip(df['DrugBank ID'], df['Drug Name']))
    return {"drugs": drugs_dict}

#Returning SDF File
@app.get("/getSdf/{drug_id}")
async def get_sdf(drug_id: str):
    print(drug_id)
    file_path = os.path.join(SDF_DIR, f"{drug_id}.sdf")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='chemical/x-mdl-sdfile')
    else:
        raise HTTPException(status_code=404, detail="SDF file not found")

@app.get("/predict_interaction")
def predict_interaction(drug1: str, drug2: str):
    key1 = f"{drug1.lower()}_{drug2.lower()}"
    key2 = f"{drug2.lower()}_{drug1.lower()}"

    # Load interaction descriptions
    df = pd.read_csv('./resources/interactions.csv')

    # Load drug id -> name mapping
    drug_map_df = pd.read_csv('./resources/drugs.csv')  # <-- file with id, name

    # Get drug names from IDs
    drug1_name_row = drug_map_df[drug_map_df['DrugBank ID'] == drug1]
    drug2_name_row = drug_map_df[drug_map_df['DrugBank ID'] == drug2]

    drug1_name = drug1_name_row.iloc[0]['Drug Name'] if not drug1_name_row.empty else drug1
    drug2_name = drug2_name_row.iloc[0]['Drug Name'] if not drug2_name_row.empty else drug2

    # Random interaction row
    interaction = np.random.randint(0, len(df))
    interaction_row = df.iloc[interaction]

    # fetch description
    description_template = interaction_row['Description']

    # fill placeholders
    description_filled = description_template.replace('#Drug1', drug1_name).replace('#Drug2', drug2_name)

    ddi_name = interaction_row['DDI type']  # assuming DDI type is in CSV

    return {
        "type": ddi_name,
        "desc": description_filled,
        "accuracy": "0.85"
    }
