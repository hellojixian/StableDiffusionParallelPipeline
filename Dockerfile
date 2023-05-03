# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
WORKDIR /app
COPY . .
ENV PYHTONUNBUFFERED=1
RUN apt update
RUN pip3 install -r requirements.txt
ENTRYPOINT ["sh", "./benchmark.sh"]%