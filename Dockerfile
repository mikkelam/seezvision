FROM tensorflow/tensorflow:2.2.0rc2-gpu-jupyter
#FROM tensorflow/tensorflow:2.2.0rc0-jupyter

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

COPY . /app/
WORKDIR /app


RUN useradd -rm -d /home/docker -s /bin/bash -g root -G sudo -u 1000 docker
RUN wget https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment.hu
USER docker