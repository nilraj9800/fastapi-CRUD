from ..  import models,schemas
from fastapi import FastAPI,status,Depends,HTTPException,APIRouter
from sqlalchemy.orm import Session
from ..database import engine,get_db

router = APIRouter(
    prefix = "/users"
)

@router.post("/",status_code=status.HTTP_201_CREATED,response_model=schemas.PostResponse)
def create_user(user:schemas.CreateUser,db:Session = Depends(get_db)):
    new_post = models.Users(**user.dict())
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    return new_post

@router.get("/{id}",response_model=schemas.PostResponse)
def get_user_id(id:int, db: Session = Depends(get_db)):
    user = db.query(models.Users).filter(models.Users.id == id).first()
    if not user:
        raise HTTPException(status_code= status.HTTP_404_NOT_FOUND,
                            detail=f"post with id: {id} was not found")
    return user

# As the GET method searches the first get route in the code,automatically the above code will be implemented
# To get around this we add a /name in the url
@router.get("/name/{name}")
def get_user_name(name:str,db: Session = Depends(get_db)):
    post = db.query(models.Users).filter(models.Users.firstName == name).first()
    if not post:
        raise HTTPException(status_code= status.HTTP_404_NOT_FOUND,
                            detail=f"post with id: {name} was not found")
    return post


@router.delete("/{id}",status_code=status.HTTP_204_NO_CONTENT)
def delete_post(id: int, db:Session = Depends(get_db)):
    delpost = db.query(models.Users).filter(models.Users.id == id)
    if delpost.first() == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"post with id {id} does not exist ")
    delpost.delete(synchronize_session = False)
    db.commit()