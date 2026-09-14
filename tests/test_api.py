from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model"] == "naive_bayes"


def test_prediction_endpoint_product_classification_laptop():
    response = client.post(
        "/api/v1/predict",
        json={"text": "Apple laptop with 16GB RAM and M3 processor"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["model"] == "naive_bayes"
    assert payload["data"]["prediction"] == "laptop"
    assert payload["data"]["probability"] > 0.5


def test_prediction_endpoint_product_classification_phone():
    response = client.post(
        "/api/v1/predict",
        json={"text": "iPhone 15 Pro Max 256GB titanium blue"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["prediction"] == "phone"
    assert payload["data"]["probability"] > 0.5


def test_prediction_endpoint_product_classification_tablet():
    response = client.post(
        "/api/v1/predict",
        json={"text": "iPad Air 11-inch with M2 chip and 128GB storage"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["prediction"] == "tablet"
    assert payload["data"]["probability"] > 0.5


def test_prediction_endpoint_empty_text_rejected():
    response = client.post(
        "/api/v1/predict",
        json={"text": ""},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "features cannot be empty"
