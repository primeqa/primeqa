FROM python:3.9-slim as sphinx

# Pyserini needs Java
ENV JAVA_HOME=/usr/local/openjdk-11
ENV PATH=${PATH}:${JAVA_HOME}/bin
COPY --from=openjdk:11.0-jdk-slim ${JAVA_HOME} ${JAVA_HOME}
RUN update-alternatives --install /usr/bin/java java ${JAVA_HOME}/bin/java 1
RUN update-alternatives --install /usr/bin/javac javac ${JAVA_HOME}/bin/javac 1

# Based on https://github.com/sphinx-doc/docker/blob/master/base/Dockerfile
# If latex is needed in the future use https://github.com/sphinx-doc/docker/blob/master/latexpdf/Dockerfile instead
WORKDIR /docs
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      graphviz \
      imagemagick \
      make \
 && apt-get autoremove \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip
COPY . .
RUN python3 -m pip install --no-cache-dir .[docs]
WORKDIR docs
RUN make clean html


FROM alpine:latest as redbean

# Based on https://github.com/kissgyorgy/redbean-docker
ARG DOWNLOAD_FILENAME=redbean-original-1.4.com

RUN apk add --update zip bash
RUN wget https://justine.lol/redbean/${DOWNLOAD_FILENAME} -O redbean.com
RUN chmod +x redbean.com

# This will normalize the binary to ELF
RUN zip -d redbean.com .ape
RUN bash /redbean.com -h

# Add your files here
COPY --from=sphinx /docs/docs/_build/html /assets
WORKDIR /assets
RUN zip -r /redbean.com *

# just for debugging purposes
RUN ls -la /redbean.com
RUN zip -sf /redbean.com


FROM scratch

COPY --from=redbean /redbean.com /
CMD ["/redbean.com", "-vv", "-p", "80"]