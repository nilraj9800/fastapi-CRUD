from ..  import models,schemas
from fastapi import FastAPI,status,Depends,HTTPException,APIRouter
from sqlalchemy.orm import Session
from ..database import engine,get_db

router = APIRouter(
    prefix = "/email"
)

@router.post("/",status_code=status.HTTP_201_CREATED,response_model=schemas.CreateEmail)
def create_email(email:schemas.CreateEmail,db:Session = Depends(get_db)):
    additionalemail = models.email(**email.dict())
    db.add(additionalemail)
    db.commit()
    db.refresh(additionalemail)
    return additionalemail

@router.put("/{id}",response_model=schemas.CreateEmail)
def update_email(id:int,updated_email:schemas.CreateEmail,db:Session = Depends(get_db)):
    update_post = db.query(models.email).filter(models.email.id == id)
    
    if update_post == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"post with id {id} does not exist")
    update_post.update(updated_email.dict(),synchronize_session=False)
    db.commit()
    return update_post.first()