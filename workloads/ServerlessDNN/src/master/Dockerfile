FROM python:3.10
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app
COPY resnet-master.py ./
ENTRYPOINT ["python3", "-u", "/app/resnet-master.py"]