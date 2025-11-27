import json
import random

from locust import HttpUser, task, between


class ModelUser(HttpUser):
    """
    Locust load test user that hits the /predict endpoint.
    """
    wait_time = between(1, 3)  

    @task
    def predict(self):
        # Generate some random test data – adapt to your feature space
        payload = {
            "data": {
                "feature_0": random.uniform(0, 10),
                "feature_1": random.uniform(0, 100),
                "feature_2": random.uniform(-5, 5),
            }
        }

        headers = {"Content-Type": "application/json"}

        with self.client.post(
            "/predict",
            data=json.dumps(payload),
            headers=headers,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status code: {response.status_code}")