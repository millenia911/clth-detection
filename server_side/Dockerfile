FROM bitnami/pytorch:latest

USER root

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && \
    apt-get install build-essential \
    pip install --no-cache-dir -r requirements.txt

RUN python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html


EXPOSE 8080

COPY . .

CMD [ "python", "server.py" ]
