FROM tensorflow/tensorflow:2.9.1-gpu
# ^ For GPU tf

# FROM tensorflow/tensorflow:2.9.1
# ^ For CPU tf

RUN apt-get update -qq \
    && apt-get install -yq python3-venv a2ps parallel\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m venv --system-site-packages /env
ENV VIRTUAL_ENV /env
ENV PATH "/env/bin:$PATH"

ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

FROM    nvidia/cuda:11.4.0-runtime-ubuntu20.04
RUN     apt-get update
RUN    apt-get install black -y

RUN apt-get install flake8 -y

WORKDIR /app

ENTRYPOINT ["bash", "-c"]
