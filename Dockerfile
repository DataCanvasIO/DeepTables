From python:3.8-buster

# ARG PIP_PKGS="tensorflow==2.4.2 hypernets[all] deeptables shap"
ARG PIP_PKGS=" featuretools==0.27.0 matplotlib==3.1.0 shap jupyterlab"
ARG PIP_OPTS="--disable-pip-version-check --no-cache-dir"
# ARG PIP_OPTS="--disable-pip-version-check --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/"
RUN python -m pip install $PIP_OPTS --upgrade pip

RUN python -m pip install $PIP_OPTS numpy==1.20
RUN python -m pip install $PIP_OPTS tensorflow==2.8.2

RUN python -m pip install $PIP_OPTS $PIP_PKGS\
    && python -m pip install $PIP_OPTS "deeptables"\
    && mkdir -p /opt/datacanvas \
    && cp -r /usr/local/lib/python3.8/site-packages/deeptables/examples /opt/datacanvas/ \
    && echo "#!/bin/bash\njupyter lab --notebook-dir=/opt/datacanvas --ip=0.0.0.0 --port=\$NotebookPort --no-browser --allow-root --NotebookApp.token=\$NotebookToken" > /entrypoint.sh \
    && chmod +x /entrypoint.sh \
    && rm -rf /tmp/*

EXPOSE 8888

ENV TF_CPP_MIN_LOG_LEVEL=3 \
    NotebookToken="" \
    NotebookPort=8888

CMD ["/entrypoint.sh"]

# docker run --rm --name deeptables -p 8830:8888 -e NotebookToken=your-token datacanvas/deeptables
