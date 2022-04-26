from ..  import models,schemas
from fastapi import FastAPI,status,Depends,HTTPException,APIRouter
from sqlalchemy.orm import Session
from ..database import engine,get_db

router = APIRouter(
    prefix = "/phonenumber"
)

@router.post("/",status_code=status.HTTP_201_CREATED,
response_model=schemas.CreatePhoneNumber)
def create_phone_number(number:schemas.CreatePhoneNumber,db:Session = Depends(get_db)):
    additionalnumber = models.phonenumber(**number.dict())
    db.add(additionalnumber)
    db.commit()
    db.refresh(additionalnumber)
    return additionalnumber

@router.put("/{id}",response_model=schemas.CreatePhoneNumber)
def update_phone_number(id:int,updated_number:schemas.CreatePhoneNumber,
db:Session = Depends(get_db)):
    update_post = db.query(models.phonenumber).filter(models.phonenumber.id == id)

    if update_post == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"post with id {id} does not exist")

    update_post.update(updated_number.dict(),synchronize_session=False)
    db.commit()
    
    return update_post.first()