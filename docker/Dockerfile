FROM debian:jessie

RUN echo "deb http://deb.debian.org/debian jessie-backports main contrib" >> /etc/apt/sources.list
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -t jessie-backports cmake make gcc g++ curl git  -y
RUN apt-get install libopenmpi-dev libtbb-dev openmpi-bin libgtest-dev libmuparser-dev locales -y

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

ADD bootstrap.sh /root/bootstrap.sh
RUN chmod a+x /root/bootstrap.sh && /root/bootstrap.sh
