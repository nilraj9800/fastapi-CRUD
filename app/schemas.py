from pydantic import BaseModel,EmailStr


class CreateUser(BaseModel):
    lastName: str
    firstName: str
    email:list[EmailStr]
    phonenumber:list[int]
    
    #returns the response as JSON
    class Config:
        orm_mode =True

class PostResponse(CreateUser):
    class Config:
        orm_mode =True


class CreateEmail(BaseModel):
    email:EmailStr

    class Config:
        orm_mode =True

class CreatePhoneNumber(BaseModel):
    phonenumber:int

    class Config:
        orm_mode =True