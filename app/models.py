from .database import Base
from sqlalchemy import Column,Integer,String,ARRAY,BIGINT

class Users(Base):
    __tablename__ = "user"
    id = Column(Integer,primary_key=True,nullable= False)
    lastName = Column(String,nullable=False)
    firstName = Column(String,nullable=False)
    email = Column(ARRAY(String(30)),nullable=False,unique=True)
    phonenumber = Column(ARRAY(BIGINT),nullable=False)


class email(Base):
    __tablename__ = "email"
    id = Column(Integer,primary_key= True,nullable= False)
    email = Column(String,nullable=False)

class phonenumber(Base):
    __tablename__ = "phonenumber"
    id = Column(Integer,primary_key= True,nullable= False)
    phonenumber = Column(BIGINT,nullable=False)
    