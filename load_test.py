import requests, random, time, concurrent.futures

API = "http://localhost:8000"

def make_prediction():
    features = {
        "Age":                random.randint(20, 85),
        "Gender":             random.randint(0, 1),
        "Family History":     random.randint(0, 1),
        "Prior Fractures":    random.randint(0, 1),
        "Calcium Intake":     random.randint(0, 2),
        "Physical Activity":  random.randint(0, 2),
        "Smoking":            random.randint(0, 1),
        "Alcohol Consumption":random.randint(0, 1),
        "Hormonal Changes":   random.randint(0, 1),
        "Body Weight":        random.randint(0, 1),
        "Vitamin D Intake":   random.randint(0, 1),
        "Medical Conditions": random.randint(0, 1),
        "Medications":        random.randint(0, 1),
        "Race/Ethnicity":     random.randint(0, 4),
    }
    resp = requests.post(f"{API}/predict", json={
        "features":      features,
        "experiment_id": "exp_001"
    })
    return resp.json()

def run_load_test(n_requests=200, concurrency=10):
    print(f"🚀 Sending {n_requests} requests "
          f"with concurrency {concurrency}...")
    start  = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [
            executor.submit(make_prediction)
            for _ in range(n_requests)
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    elapsed = time.time() - start
    a_count = sum(1 for r in results if r.get("variant") == "A")
    b_count = sum(1 for r in results if r.get("variant") == "B")
    avg_lat = sum(r.get("latency_ms", 0) for r in results) / len(results)

    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"   Throughput: {n_requests/elapsed:.1f} req/s")
    print(f"   Model A:    {a_count} requests ({a_count/n_requests:.0%})")
    print(f"   Model B:    {b_count} requests ({b_count/n_requests:.0%})")
    print(f"   Avg latency:{avg_lat:.1f}ms")

if __name__ == "__main__":
    run_load_test(200, 10)