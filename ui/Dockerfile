FROM python:3.7.4-stretch

WORKDIR /home/user

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# install as a package
COPY ./ui/requirements.txt /home/user/
RUN pip install -r requirements.txt

# copy UI code
COPY ./ui /home/user/ui

EXPOSE 8501

# cmd for running the API
CMD ["streamlit", "run", "ui/webapp.py"]
