FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install -y python3-pip

RUN pip3 install numpy
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install Flask
RUN pip3 install flask-restful

COPY . /anomaly_detection_app
WORKDIR /anomaly_detection_app

RUN python3 main.py -p vulnerable_robot_challenge.csv -sr -sm -nv -fi
RUN python3 main.py -p vulnerable_robot_challenge.csv -lv -sr -sm -nv -fi
