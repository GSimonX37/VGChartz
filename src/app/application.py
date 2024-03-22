import pandas as pd

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response

from app.model import Model


model = Model()

app = FastAPI()


@app.post("/")
async def send_response(request: Request) -> Response:
    """
    Обрабатывает запрос клиента, который должен содержать:

    - data: данные видеоигры;

    Отправляет ответ на запрос клиента, который содержит:

    - data: данные видеоигры;
    - sales: количество проданных копий видеоигры;

    :param request: запрос от клиента;
    :return: ответ на запрос клиента.
    """

    data: dict = await request.json()

    body = {}

    game = pd.DataFrame(
        data=[[*data.values()]],
        index=[0],
        columns=[*data.keys()]
    )

    sales = model.result(
        data=game
    )

    body.update(data)
    body['sales'] = sales

    return Response(
        status_code=200,
        media_type='application/json',
        content=f'{body}'
    )
