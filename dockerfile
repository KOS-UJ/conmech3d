FROM python:3.10.12-slim-buster

WORKDIR /usr/src/app

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install -y git
RUN apt-get install -y python3-gmsh
RUN apt-get install -y g++

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements-deep.txt

ENV PYTHONPATH "${PYTHONPATH}:."
