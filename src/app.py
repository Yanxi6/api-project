from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Import feature engineering from your ML project:
# For now, we paste the same add_features logic here (later we can package it).
import re

MODEL_PATH = Path("models/titanic_logreg.joblib")

app = FastAPI(title="Titanic Model API")

_model = None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int).astype(str)
    df["CabinKnown"] = df["Cabin"].notna().astype(int).astype(str)

    def extract_title(name: str) -> str:
        if not isinstance(name, str):
            return "Unknown"
        m = re.search(r",\s*([^\.]+)\.", name)
        if not m:
            return "Unknown"
        title = m.group(1).strip()
        rare = {"Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"}
        if title in rare:
            return "Rare"
        if title in {"Mlle", "Ms"}:
            return "Miss"
        if title == "Mme":
            return "Mrs"
        return title

    df["Title"] = df["Name"].apply(extract_title)
    return df


class Passenger(BaseModel):
    PassengerId: int = 9999
    Pclass: int
    Name: str
    Sex: str
    Age: float | None = None
    SibSp: int = 0
    Parch: int = 0
    Ticket: str = "UNKNOWN"
    Fare: float | None = None
    Cabin: str | None = None
    Embarked: str | None = None


@app.on_event("startup")
def load_model():
    global _model
    _model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Passenger):
    df = pd.DataFrame([p.model_dump()])
    df = add_features(df)
    pred = int(_model.predict(df)[0])
    proba = float(_model.predict_proba(df)[0].max())
    return {"survived": pred, "confidence": proba}
