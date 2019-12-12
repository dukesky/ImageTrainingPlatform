FROM "python:3.6"
WORKDIR /cift_trainserver

COPY requirement.txt /cift_trainserver
RUN pip install -r ./requirement.txt

#EXPOSE 8000

COPY ciftServer.py /cift_trainserver
CMD ["/usr/bin/python", "ciftServer.py"]~ 
