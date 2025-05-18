# tests/test_scraping.py

from lambda_scraping.lambda_function_scraping import handler as scraping_handler
from lambda_volatilidad.lambda_function_volatilidad import handler as volatilidad_handler

def test_scraping_lambda_mock():
    # Aquí podrías simular un evento de scraping básico y comprobar que no falla
    event = {}
    context = None
    result = scraping_handler(event, context)
    assert result is not None

def test_volatilidad_lambda_mock():
    # Simulación básica
    event = {}
    context = None
    result = volatilidad_handler(event, context)
    assert result is not None
