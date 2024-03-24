FROM python:3.10

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./models/lgbmr ./models/lgbmr
COPY ./src/app ./app
COPY ./src/run.py .

CMD ["python", "run.py", "lgbmr"]