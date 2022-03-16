FROM nvidia/cuda:11.0-runtime-ubuntu20.04

WORKDIR /home/user

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip curl wget git pkg-config cmake swig
 
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Set default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --set python3 /usr/bin/python3.7

# Install PDF converter
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz && \
    tar -xvf xpdf-tools-linux-4.03.tar.gz && cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin

RUN apt-get install libpoppler-cpp-dev pkg-config -y --fix-missing

# Install packages
# Install pytorch with gpu support
RUN pip3 install numpy scipy Cython
RUN pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Install faiss separately as building latest versions can cause trouble with swig
RUN pip3 install faiss-cpu==1.7.0

COPY ./api/requirements.txt /home/user/
RUN pip3 install -r requirements.txt

# Copy API code
COPY ./api /home/user/api

EXPOSE 8000

# cmd for running the API (note: "--preload" is not working with cuda)
CMD ["api/init_container.sh"]
