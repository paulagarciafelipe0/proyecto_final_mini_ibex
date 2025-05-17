import boto3
import pandas as pd
import mibian
from decimal import Decimal
from datetime import datetime, timezone

dynamodb = boto3.resource('dynamodb', region_name='eu-west-1')

# Tablas DynamoDB
tabla_opciones = dynamodb.Table('OpcionesMiniIBEX')
tabla_futuros = dynamodb.Table('FuturosMiniIBEX')
tabla_volatilidad = dynamodb.Table('VolatilidadMiniIBEX')

def calcular_volatilidad(opcion_row, precio_futuro):
    try:
        if pd.isna(opcion_row['strike']) or pd.isna(opcion_row['precio_anterior']):
            return None

        strike = float(opcion_row['strike'])
        precio_opcion = float(opcion_row['precio_anterior'])

        fecha_vencimiento = datetime.strptime(opcion_row['fecha_vencimiento'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        dias_restantes = (fecha_vencimiento - datetime.now(timezone.utc)).days

        if dias_restantes <= 0:
            return None

        tipo = opcion_row['tipo']
        if tipo == 'CALL':
            bs = mibian.BS([precio_futuro, strike, 0, dias_restantes], callPrice=precio_opcion)
            vol = bs.impliedVolatility
        elif tipo == 'PUT':
            bs = mibian.BS([precio_futuro, strike, 0, dias_restantes], putPrice=precio_opcion)
            vol = bs.impliedVolatility
        else:
            return None

        if vol is None or pd.isna(vol) or vol < 0:
            return None

        return round(vol, 5)

    except Exception as e:
        print(f"Error calculando volatilidad Strike {strike} {tipo}: {e}")
        return None

def obtener_futuro_mas_proximo():
    futuros = tabla_futuros.scan()['Items']
    df_futuros = pd.DataFrame(futuros)

    if df_futuros.empty:
        print("No se encontraron futuros.")
        return None

    df_futuros['fecha_vencimiento'] = pd.to_datetime(df_futuros['fecha_vencimiento'], utc=True, errors='coerce')

    # Usa correctamente la columna 'último' como precio del futuro
    df_futuros['precio_anterior'] = pd.to_numeric(df_futuros['último'], errors='coerce')

    df_valid = df_futuros.dropna(subset=['fecha_vencimiento', 'precio_anterior'])
    df_valid = df_valid[df_valid['precio_anterior'] > 0]

    if df_valid.empty:
        print("No hay futuros válidos con precio 'último'.")
        return None

    df_valid = df_valid.sort_values(by='fecha_vencimiento')
    precio = df_valid.iloc[0]['precio_anterior']
    print(f"Futuro más próximo: {df_valid.iloc[0]['fecha_vencimiento'].date()} (Precio: {precio})")
    return precio

def lambda_handler(event, context):
    opciones = tabla_opciones.scan()['Items']
    if not opciones:
        print("No se encontraron opciones.")
        return {'statusCode': 404, 'body': 'No hay opciones para procesar.'}

    precio_futuro = obtener_futuro_mas_proximo()
    if precio_futuro is None:
        return {'statusCode': 404, 'body': 'No se encontró futuro válido.'}

    items_procesados = 0

    for row in opciones:
        vol = calcular_volatilidad(row, float(precio_futuro))
        if vol is None:
            continue

        item_vol = {
            'fecha_vencimiento': row['fecha_vencimiento'],
            'strike_tipo_timestamp': row['strike_tipo_timestamp'],
            'volatilidad_implicita': Decimal(str(vol)),
            'timestamp': datetime.utcnow().isoformat()
        }

        tabla_volatilidad.put_item(Item=item_vol)
        items_procesados += 1

    print(f"{items_procesados} volatilidades calculadas y guardadas.")
    return {
        'statusCode': 200,
        'body': f'{items_procesados} volatilidades calculadas y guardadas correctamente'
    }
