from socket import getnameinfo
from app import schemas
from app.config import settings
from .conftest import client,session

def test_create_user(client):
    res = client.post(
        "/users/", json={"lastName":"doe","firstName":"john","email":["johndoe@gmail.com"],"phonenumber":["123456789"]})

    new_user = schemas.CreateUser(**res.json())

    assert new_user.lastName == "doe"
    assert new_user.firstName == "john"
    assert new_user.email == ["johndoe@gmail.com"]
    assert new_user.phonenumber == [123456789]
    assert res.status_code == 201

def test_get_user(client,test_user):
    res = client.get(f"/users/{test_user[0].id}")

    getid = schemas.CreateUser(**res.json())

    assert getid.lastName == test_user[0].lastName
    assert getid.firstName == test_user[0].firstName
    assert getid.email == test_user[0].email
    assert getid.phonenumber == test_user[0].phonenumber

def test_get_user_name(client,test_user_name):
    res = client.get(f"/users/{test_user_name['firstName']}")

    getname = schemas.CreateUser(**res.json())

    assert getname.lastName == test_user_name['lastName']
    assert getname.firstName == test_user_name['firstName']
    assert getname.email == test_user_name['email']
    assert getname.phonenumber == test_user_name['phonenumber']


def test_user_non_exist(client):
    res = client.get(
        f"/users/8000000")
    assert res.status_code == 404


def test_delete_user_success(client, test_user):
    res = client.delete(
        f"/users/{test_user[0].id}")
    assert res.status_code == 204

def test_delete_user_non_exist(client):
    res = client.delete(
        f"/posts/8000000")
    assert res.status_code == 404




