from fastapi import FastAPI
from . import models
from .database import engine
from .routers import user,email,phonenumber
from .config import settings

#Comment out when creating a new database
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user.router)
app.include_router(email.router)
app.include_router(phonenumber.router)

@app.get("/")
def root():
    return{"message":"Hello World"}