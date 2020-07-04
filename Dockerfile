FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install -y python3-pip

COPY requirements.txt /anomaly_detection_app/requirements.txt

COPY . /anomaly_detection_app
WORKDIR /anomaly_detection_app

RUN pip3 install -r requirements.txt

RUN python3 main.py -p vulnerable_robot_challenge.csv -sr -sm -nv -fi
RUN python3 main.py -p vulnerable_robot_challenge.csv -lv -sr -sm -nv -fi
