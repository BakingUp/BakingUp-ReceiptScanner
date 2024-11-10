FROM python:3.13.0-bookworm

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8001"]