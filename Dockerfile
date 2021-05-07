FROM python:3.7.9

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

RUN python3 --version
RUN pip3 --version

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 
EXPOSE 5000
