# Clothing Detection with Pytorch

## Description
Detect items of person outfit using deep learning in Pytorch with Detectron2(cpu). The application is served over network and can be accessed with Web UI or just the API. It consume an image and spit the annotated image and the annotation dict as the response.

It was trained with clothing dataset from https://github.com/seralexger/clothing-detection-dataset

Trained weights is available here https://drive.google.com/drive/folders/1EAqzV06-23-bjiUm15RTUPHkU8V2tQ24?usp=sharing

| Warning: The model is not reliable yet (until updated), further improvement with the dataset and training setup is needed.|
| --- |

## How to Run w/o Docker
The main script of this project is `server.py` inside `server_side` folder. When it runs, it will serve the inference algorithm at `http://{hostname}:8080/api/test`, with method `POST` to request inference process. Request payload must be an image.

To test the app, run the web interface with `app.py` inside `client` folder. It will serve the interface at `http://{hostname}:5000/home` .

note: don't forget to install the requirements first.

## Run with docker
To run with docker, just type command:
``` 
$ docker-compose.up
```
in docker-compose.yml directory. It will start both `server.py` and `app.py` as services.

As an alternative, you can run them in Docker swarm mode. It enables you to create duplicates of container with same docker image instantly. 

Initiate the swarm first:
```
$ docker swarm init
```
then 
```
$ docker stack deploy -c docker-compose.yml <your-stack-name>
```
to stop the stack
```
$ docker stack rm <your-stack-name>
```

--------------------------------------


