import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('Best_model.pkl')

df = pd.read_csv('X_test.csv')

thresh = 0.35


# defining the request body
class ID(BaseModel):
    IDclient: int


# defining the endpoint
@app.post('/prediction')
def predict_loan_repayment(data: ID):
    received = data.dict()
#    print(received) #ok: {'IDclient': 60667}
    IDclient = received['IDclient']
#    print(IDclient) #ok: 60667
    if IDclient not in df['SK_ID_CURR'].tolist():
        raise HTTPException(status_code=404, detail="There is no client with such ID")

    data_ID = df[df['SK_ID_CURR'] == IDclient]
#    print(data_ID) #ok
    pred_proba = model.predict_proba(data_ID)[:,1]
#    print(pred_proba) #ok: [0.3104069]
    if (pred_proba[0] < thresh):
        pred = 'Loan granted!'
    else:
        pred = 'Sorry, loan ungranted...'
    return {
        'predicted probability': pred_proba[0],
        'prediction': pred
    }

# Run the API - locally
#if __name__ == "__main__":
#    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn modelAPI:app -- reload # to type in cmd to run it (--reload for dev mode)
