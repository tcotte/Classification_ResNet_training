FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.9
LABEL authors="Tristan_COTTE"

RUN pip3 config set global.trusted-host "download.pytorch.org"

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .

RUN pip3 install -r requirements.txt
#RUN pip3 install torchvision --upgrade


WORKDIR /src


COPY src .

RUN mkdir -p saved_models && chmod 777 saved_models
RUN mkdir -p dataset && chmod 777 dataset
RUN chmod 777 ./

#ENTRYPOINT ["python3.9", "train.py"]