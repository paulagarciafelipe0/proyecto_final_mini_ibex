FROM public.ecr.aws/lambda/python:3.11

# Copiar archivos al contenedor
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt -t .

COPY lambda_function_volatilidad.py .

# Establecer el manejador de Lambda
CMD ["lambda_function_volatilidad.lambda_handler"]


