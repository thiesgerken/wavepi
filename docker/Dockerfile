FROM debian:jessie

# RUN echo "deb http://deb.debian.org/debian jessie-backports main contrib" >> /etc/apt/sources.list
RUN echo "deb [check-valid-until=no] http://archive.debian.org/debian jessie-backports main contrib" >> /etc/apt/sources.list
RUN echo "Acquire::Check-Valid-Until \"false\";" >> /etc/apt/apt.conf

RUN apt-get update && apt-get upgrade -y
RUN apt-get install libopenmpi-dev libtbb-dev openmpi-bin locales -y
RUN apt-get install -t jessie-backports cmake make gcc g++ curl git libgtest-dev libmuparser-dev -y

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

ADD boost.sh /root/boost.sh
RUN bash /root/boost.sh

ADD dealII.sh /root/dealII.sh
RUN bash /root/dealII.sh

ADD gtest.sh /root/gtest.sh
RUN bash /root/gtest.sh
