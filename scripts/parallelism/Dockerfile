# pull the official docker image with python version
FROM python:3.11.1-slim
# set work directory
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

CMD ["bash"]