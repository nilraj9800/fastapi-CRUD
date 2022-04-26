from app import schemas
from app.config import settings
from .conftest import client,session

def test_create_phonenumber(client):
    res = client.post(
        "/phonenumber/", json={"phonenumber":12345678})

    new_number = schemas.CreatePhoneNumber(**res.json())

    assert new_number.phonenumber == 12345678
    assert res.status_code == 201


def test_update_number(client,test_number):
    data = {
        "phonenumber": 2345678,
        "id": test_number[0].id
    }
    res = client.put(f"/phonenumber/{test_number[0].id}", json=data)
    updated_number = schemas.CreatePhoneNumber(**res.json())
    assert res.status_code == 200
    assert updated_number.phonenumber == data['phonenumber']

