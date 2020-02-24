FROM chrisai/haiku-scalable-example-base:latest

WORKDIR /home/circleci

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "actor_client.py"]