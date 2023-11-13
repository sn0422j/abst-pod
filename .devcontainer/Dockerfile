FROM python:3.10.13-bullseye

RUN apt update && \
    apt install build-essential
RUN pip install --upgrade pip && \
    pip install cmake --upgrade
RUN pip install "espnet[all]" pyopenjtalk
RUN apt update && \
    apt install ffmpeg -y
RUN pip install pybars3 black isort 
