from fastapi.testclient import TestClient

from .app import app

client = TestClient(app)


def test_health_not_ready():
    response = client.get("/health")
    assert response.status_code == 421

def test_health_ready():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

def test_preadict():
    with TestClient(app) as client:
        with open("./test_data/X_test_tree.npy", 'rb') as f:
            file = f.read()
        response = client.post(
            "/predict", 
            files={'file':('file', file, "multipart/form-data")}
        )
        assert response.status_code == 200
        assert type(response.json()) == dict
        assert type(response.json()['predict']) == list

def test_preadict_exception():
    with TestClient(app) as client:
        with open("./test_data/not_array.npy", 'rb') as f:
            file = f.read()
        response = client.post(
            "/predict", 
            files={'file':('file', file, "multipart/form-data")}
        )
        assert response.status_code == 400
