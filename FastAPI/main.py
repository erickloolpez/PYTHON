from fastapi import FastAPI
from fastapi.responses import HTMLResponse , JSONResponse
from pydantic import BaseModel

from jwt_manager import create_token

from config.database import  engine, Base

from middlewares.error_handler import ErrorHandler

from routers.movie import movie_router

app = FastAPI()
app.title = "Mi first Back with FastAPI"
app.version = "0.0.1"

app.add_middleware(ErrorHandler)
app.include_router(movie_router)

Base.metadata.create_all(bind=engine)


class User(BaseModel):
    email: str
    password: str

movies = [
    {
        "id": 1,
        "title": "Avatar",
        "overview": "Es de unos monos que son de color azul.",
        "year": 2019,
        "rating": 8,
        "category": "Accion"
    },
    {
        "id": 2,
        "title": "Avatar Ang",
        "overview": "Es de unos monos que son de color azul.",
        "year": 2019,
        "rating": 8,
        "category": "Accion"
    }
]

#Metodo POST de Usuarios
@app.post('/login', tags=['auth'])
def login(user: User):
    if user.email == "admin@gmail.com" and user.password == 'admin':
        token:str = create_token(user.dict())
        return JSONResponse(content=token)



#Metodo GET Pruebas
@app.get('/', tags=['Home'])
def message():
    return HTMLResponse('<h1>Hellow World!</h1>')


