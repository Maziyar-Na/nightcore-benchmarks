FROM zjia/nightcore:asplos-ae as nightcore
FROM python:3.10 as builder

COPY --from=nightcore /src/nightcore       /src/nightcore
COPY ./python/ /src/nightcore/worker/python/

FROM ubuntu:22.04
RUN apt update && apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html

COPY --from=builder   /src/nightcore/worker/python  /nightcore/worker/python
COPY --from=nightcore /nightcore/launcher           /nightcore/launcher
COPY main.py          /app/main.py
ENV  PYTHONPATH       /nightcore/worker/python
WORKDIR /app

ENTRYPOINT [ "/nightcore/launcher", \
             "--fprocess=python3 main.py", \
             "--fprocess_working_dir=/app", \
             "--fprocess_mode=python" ]
