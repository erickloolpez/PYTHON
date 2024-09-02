
from fastapi import APIRouter
from fastapi import  Path, Query, Depends
from fastapi.responses import   JSONResponse
from typing import  List

from config.database import Session
from models.movie import Movie as MovieModel

from fastapi.encoders import jsonable_encoder
from middlewares.jwt_bearer import JWTBearer

from services.movie import MovieService

from schemas.movie import Movie

movie_router = APIRouter()



#Metodo GET general
@movie_router.get('/movies', tags=['Movies'],response_model=List[Movie], status_code=200, dependencies=[Depends(JWTBearer())])
def get_movies() -> List[Movie]:
    db = Session()
    result = MovieService(db).get_movies()
    return JSONResponse(status_code=200,content= jsonable_encoder(result))



#Metodo GET por id
@movie_router.get('/movies/{id}', tags=['Movies'])
def get_movie(id: int = Path(ge=1, le=2000)):
    db = Session()
    result = MovieService(db).get_movie(id)

    if not result :
        return JSONResponse(status_code=404, content={'message':"No encontrado"})
    return JSONResponse(content=jsonable_encoder(result))


#Metodo GET por Categoria
@movie_router.get('/movies/', tags=['Movies'])
def get_movie_by_category(category: str = Query(min_length=5) ):
    return JSONResponse(content={'message':'wenas'})



#Metodo POST
@movie_router.post('/movies', tags=['Movies'], response_model=dict)
def create_movie(movie: Movie) -> dict:
    db = Session()
    MovieService(db).create_movie(movie)
    return JSONResponse(content={"message":"Se ha registrado la pelicula." })



#Metodo PUT
@movie_router.put('/movies/{id}', tags=['Movies'])
def update_movie(id: int, movie: Movie):
    db = Session()
    result = MovieService(db).get_movie(id)

    if not result:
        return JSONResponse(content={'message':"ID no encontrado."})
    
    MovieService(db).update_movie(id, movie)

    return JSONResponse(content={"message":"Movie Actualizada"})


#Metodo DELETE
@movie_router.delete('/movies/{id}', tags=['Movies'])
def delete_movie(id: int):
    db = Session()
    result = db.query(MovieModel).filter(MovieModel.id == id).first()

    if not result:
        return JSONResponse(content={'message':"ID no encontrado, no fue posible borrar"})
    
    db.delete(result)
    db.commit()

    return JSONResponse(content={'message':'Movie Eliminada.'})
    
