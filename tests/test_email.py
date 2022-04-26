from app import schemas
from app.config import settings
from .conftest import client,session

def test_create_email(client):
    res = client.post(
        "/email/", json={"email":"johndoe@gmail.com"})

    new_email = schemas.CreateEmail(**res.json())

    assert new_email.email == "johndoe@gmail.com"
    assert res.status_code == 201


def test_update_email(client,test_email):
    data = {
        "email": "1122nilraj@gmail.com",
        "id": test_email[0].id
    }
    res = client.put(f"/email/{test_email[0].id}", json=data)
    updated_email= schemas.CreateEmail(**res.json())
    assert res.status_code == 200
    assert updated_email.email == data['email']