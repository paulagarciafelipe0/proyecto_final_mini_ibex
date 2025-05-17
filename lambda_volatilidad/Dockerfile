FROM public.ecr.aws/lambda/python:3.11

# Copiar requirements.txt y tu función
COPY requirements.txt .
COPY lambda_function_volatilidad.py .

# Instalar dependencias
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Comando de inicio de Lambda
CMD ["lambda_function_volatilidad.lambda_handler"]

