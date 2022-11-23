FROM ubuntu

USER root

RUN apt-get update && apt-get install -y python3 \
    python3-pip

RUN mkdir -p /app/
RUN mkdir -p /app/weights
RUN mkdir -p /app/tmp_for_upload

COPY ./model /app/model
COPY ./test_data /app/test_data

COPY ./__init__.py /app/
COPY ./app.py /app/
COPY ./requirements.txt /app/
COPY ./setup.py /app/
COPY ./test_app.py /app/

RUN apt-get update && apt-get install -y curl

WORKDIR /app

RUN pip3 install .

RUN pytest

CMD ["uvicorn", "app:app",  "--host", "0.0.0.0", "--port", "8000"]
