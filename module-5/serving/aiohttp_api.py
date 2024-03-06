import logging

from aiohttp import web

logger = logging.getLogger()

from serving.predictor import Predictor


async def handle_predict(request: web.Request) -> web.Response:
    predictor: Predictor = request.app["predictor"]
    payload = await request.json()
    text = payload["text"]
    print(text)
    # This is cpu_bound operation, should be run with ProcessPoolExecutor
    result = predictor.predict(text=text)
    return web.json_response(result.tolist())


app = web.Application()
app["predictor"] = Predictor.default_from_model_registry()
app.router.add_post("/predict", handle_predict)


if __name__ == "__main__":
    web.run_app(app)
