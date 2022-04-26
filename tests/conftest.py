from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.main import app

from app.config import settings
from app.database import get_db
from app.database import Base
from app import models

SQLALCHEMY_DATABASE_URL = f'postgresql://postgres:nilraj123@localhost:5432/testapi'

# Engine is responsible for connection to database
engine = create_engine(SQLALCHEMY_DATABASE_URL)

TestingSessionLocal = sessionmaker(
    autocommit=False,autoflush=False,bind=engine)


@pytest.fixture()
def session():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(session):
    def overrid_get_db():
        try:
            yield session
        finally:
            session.close()
    app.dependency_overrides[get_db] = overrid_get_db
    yield TestClient(app)

@pytest.fixture()
def test_user(session):
    user_data = [{
        "lastName": "last1",
        "firstName": "first1",
        "email": ["email1@gmail.com"],
        "phonenumber": [123456789]
    }, {
        "lastName": "last2",
        "firstName": "first2",
        "email": ["email2@gmail.com"],
        "phonenumber": [23456789]
    },
        {
        "lastName": "last3",
        "firstName": "first3",
        "email": ["email3@gmail.com"],
        "phonenumber": [3456789]
    }, {
        "lastName": "last4",
        "firstName": "first4",
        "email": ["email4@gmail.com"],
        "phonenumber": [456789]
     }]
    def create_user_model(users):
        return models.Users(**users)

    post_map = map(create_user_model, user_data)
    user = list(post_map)
    session.add_all(user)
   
    session.commit()

    user = session.query(models.Users).all()
    return user
        
@pytest.fixture()
def test_user_name(client):
    user_data = {"lastName": "last1",
                "firstName": "first1",
                "email": ["email1@gmail.com"],
                "phonenumber": [123456789]}
    res = client.post("/users/", json=user_data)
    new_user = res.json()
    return new_user

@pytest.fixture
def test_email(session):
    user_data = [{
        "id":"1",
        "email": "email1@gmail.com"
        
    }, {
        "id":"2",
        "email": "email2@gmail.com"
            
            }]
    def create_user_model(users):
        return models.email(**users)

    post_map = map(create_user_model, user_data)
    user = list(post_map)
    session.add_all(user)
   
    session.commit()

    user = session.query(models.email).all()
    return user

@pytest.fixture
def test_number(session):
    user_data = [{
        "id":"1",
        "phonenumber": "12345678"
        
    }, {
        "id":"2",
        "phonenumber": "2345678"
            },
        {
        "id":"3",
        "phonenumber": "3456789"
    }]
    def create_user_model(users):
        return models.phonenumber(**users)

    post_map = map(create_user_model, user_data)
    user = list(post_map)
    session.add_all(user)
   
    session.commit()

    user = session.query(models.phonenumber).all()
    return user