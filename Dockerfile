FROM tensorflow/tensorflow:2.2.0rc2-gpu-jupyter
#FROM tensorflow/tensorflow:2.2.0rc0-jupyter

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

COPY src/ /app/
WORKDIR /app


RUN useradd -rm -d /home/docker -s /bin/bash -g root -G sudo -u 1000 docker

USER docker