from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserInput(BaseModel):
  email: str
  password: str

@app.post("/login")
def login(user: UserInput):
  if user.email == "ekundayogabrieloluwadamilare@gmail.com" and user.password == "gab":
    return "Login successful"
  return "Access denied"