from flask import Flask, jsonify, request
from flask_caching import Cache
from demand_sense.inference_module.infer_model import infer

"""
Server using Flask to handle the api requests.
A cache is used in the server to improve the speed of handling redundant 
requests. 
This a synchornous server handling sequential requests and can be improved by 
using threads and more workers depending on the device.
Additionally to improve, a load balancer can be incorporated to handle 
different servers. 
However, the best way to create endpoints and host them is to use hosting 
services such as Amazon Sagemaker and Azure App Services.
These services offer automatic scaling options with dynamic servers.
"""

config = {
    "DEBUG": False,  # Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300,
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)


@app.route("/api/predict")
# creates cache entry for each unique attribute combination of requests
@cache.cached(
    key_prefix=lambda: request.full_path,
    make_cache_key=lambda *args, **kwargs: str(request.args),
)
def predict():
    """
    Function handling /api/predict request with date tag

    :return response: json containing date and corresponding sales information
    """
    date = request.args.get("date")
    sales = infer(test_date=date, infer_level="day")
    prediction = {
        "date": date,
        "sales": sales,
    }
    return jsonify(prediction)


@app.route("/api/predict_customer_sales")
@cache.cached(
    key_prefix=lambda: request.full_path,
    make_cache_key=lambda *args, **kwargs: str(request.args),
)
def predict_customer_sales():
    """
    Function handling /api/predict_customer_sales request with date and customer tag

    :return response: json containing date, customer_id, and corresponding customer
    sales information on the specified date
    """
    date = request.args.get("date")
    customer_id = request.args.get("customer_id")
    sales = infer(test_date=date, customer_id=customer_id, infer_level="customer")
    prediction = {
        "date": date,
        "customer_id": customer_id,
        "sales": sales,
    }
    return jsonify(prediction)


@app.route("/api/predict_product_sales")
@cache.cached(
    key_prefix=lambda: request.full_path,
    make_cache_key=lambda *args, **kwargs: str(request.args),
)
def predict_product_sales():
    """
    Function handling /api/predict_product_sales request with date and
    product tag

    :return response: json containing date, product_id, and corresponding
    product sales information on the specified date
    """
    date = request.args.get("date")
    product_id = request.args.get("product_id")
    sales = infer(test_date=date, product_id=product_id, infer_level="product")
    prediction = {
        "date": date,
        "product_id": product_id,
        "sales": sales,
    }
    return jsonify(prediction)


@app.route("/api/predict_customer_product_sales")
@cache.cached(
    key_prefix=lambda: request.full_path,
    make_cache_key=lambda *args, **kwargs: str(request.args),
)
def predict_customer_product_sales():
    """
    Function handling /api/predict_customer_product_sales request with date,
    customer, and product tag

    :return response: json containing date, customer_id, product_id, and
    corresponding product sales information in the customer specified on the
    specified date
    """
    date = request.args.get("date")
    customer_id = request.args.get("customer_id")
    product_id = request.args.get("product_id")
    sales = infer(
        test_date=date,
        customer_id=customer_id,
        product_id=product_id,
        infer_level="customer_product",
    )
    prediction = {
        "date": date,
        "product_id": product_id,
        "customer_id": customer_id,
        "sales": sales,
    }
    return jsonify(prediction)


if __name__ == "__main__":
    app.run()
