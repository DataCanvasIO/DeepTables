FROM ubuntu:18.04
USER root
ENV LANG C.UTF-8
ENV NotebookToken ''

RUN  sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list \
     && sed -i "s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list \
     && cat /etc/apt/sources.list \
     && apt-get update \
     && apt-get install -y python3 python3-pip git \
     && pip3 install -i https://pypi.doubanio.com/simple --no-cache-dir --upgrade pip \
     && apt-get clean

RUN mkdir -p /opt/datacanvas

RUN git clone https://github.com/DataCanvasIO/deeptables.git  /opt/datacanvas/deeptables

RUN pip3 install -i https://pypi.doubanio.com/simple --no-cache-dir jupyter
RUN pip3 install -i https://pypi.doubanio.com/simple --no-cache-dir -r /opt/datacanvas/deeptables/requirements.txt

ENV PYTHONPATH /opt/datacanvas/deeptables

EXPOSE 8888

RUN echo "#!/bin/bash\njupyter notebook --notebook-dir=/opt/datacanvas/deeptables/examples --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=\$NotebookToken" > /entrypoint.sh \
    && chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]

# docker run --rm --name deeptable -p 8830:8888 -e NotebookToken=your-token datacanvas/deeptables-example
