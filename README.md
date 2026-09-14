# Local Server AI Strategy - Phân loại sản phẩm theo loại

This project provides a compact local deployment of a Naive Bayes classifier behind a FastAPI service for product type classification. The API accepts a JSON payload with text describing a product, preprocesses it, and returns the predicted product category and posterior probability.

## Features

- Separate preprocessing and prediction logic
- Naive Bayes implementation with Laplace smoothing
- REST endpoint for prediction and health checks
- Docker support for local and production-like deployment
- Basic command-line validation scripts
- Product categories: `phone`, `laptop`, `tablet`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

## Test the API

### Correct classification examples

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Apple laptop with 16GB RAM and M3 processor"}'
```

Expected response:

```json
{
  "success": true,
  "status": 200,
  "message": "Naive Bayes prediction successful",
  "data": {
    "model": "naive_bayes",
    "endpoint": "/api/v1/predict",
    "prediction": "laptop",
    "probability": 0.9,
    "health_status": "healthy"
  }
}
```

Other valid examples:

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"iPhone 15 Pro Max 256GB titanium blue"}'
```

Expected prediction: `phone`

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"iPad Air 11-inch with M2 chip and 128GB storage"}'
```

Expected prediction: `tablet`

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Dell XPS 13 laptop, 32GB RAM, Intel Core i7"}'
```

Expected prediction: `laptop`

### Incorrect / invalid examples

These inputs should be rejected or return the wrong category if the classification logic is not followed.

#### 1. Empty input

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":""}'
```

Expected:

```json
{
  "detail": "features cannot be empty"
}
```

#### 2. Wrong field type

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[123,456]}'
```

Expected:

```json
{
  "detail": "all feature values must be strings"
}
```

#### 3. Wrong prediction should not pass

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Samsung Galaxy S24 Ultra with 512GB storage"}'
```

Expected correct label: `phone`

This case is considered incorrect if the API returns `laptop` or `tablet`.

## Docker

```bash
docker compose up --build
```

## Endpoint summary

- `GET /health` — service health status
- `POST /api/v1/predict` — classify a product description or feature vector

## Test case summary

### Correct test cases

| No. | Input text | Expected prediction | Expected probability |
|---|---|---|---|
| 1 | "Apple laptop with 16GB RAM and M3 processor" | laptop | near 0.90 |
| 2 | "iPhone 15 Pro Max 256GB titanium blue" | phone | near 0.85 |
| 3 | "iPad Air 11-inch with M2 chip and 128GB storage" | tablet | near 0.85 |
| 4 | "Dell XPS 13 laptop, 32GB RAM, Intel Core i7" | laptop | near 0.90 |

### Incorrect test cases

| No. | Input text | Expected result |
|---|---|---|
| 1 | "" | 400 error: features cannot be empty |
| 2 | {"features":[123,456]} | 400 error: all feature values must be strings |
| 3 | "Samsung Galaxy S24 Ultra with 512GB storage" | should be phone, not laptop/tablet |

## Model notes

The implementation uses a categorical Naive Bayes model with Laplace smoothing. Prior probabilities are computed from the class distribution in the training dataset, while likelihoods are estimated via feature value counts for each class. The model is trained to detect product categories such as `phone`, `laptop`, and `tablet` based on product names and technical keywords.
