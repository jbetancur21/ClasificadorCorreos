import fastapi
from pydantic import BaseModel, Field
import joblib


app = fastapi.FastAPI()
class EmailData(BaseModel):
    subject: str = Field(..., example="Important Meeting Tomorrow")
@app.get("/predict")
def predict(getData: EmailData):
    
    model01C = joblib.load( 'models/modelRandomForest.pkl') # Carga del modelo.
    vectorizer = joblib.load( 'models/vectorizer.pkl') # Carga del vectorizador.
    X = vectorizer.transform([getData.subject])
    predRandom = model01C.predict(X)
    return predRandom.tolist()