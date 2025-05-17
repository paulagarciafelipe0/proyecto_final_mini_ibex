import boto3 
import pandas as pd
import logging
from decimal import Decimal
from datetime import datetime, timezone
from scraping import obtener_datos_meff
import json

# Configura logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb', region_name='eu-west-1')
tabla_opciones = dynamodb.Table('OpcionesMiniIBEX')
tabla_futuros = dynamodb.Table('FuturosMiniIBEX')

def guardar_opciones(df_opciones):
    for _, row in df_opciones.iterrows():
        if pd.isna(row['Strike']) or pd.isna(row['Anterior']):
            continue

        item = {
            'fecha_vencimiento': row['Vencimiento'].strftime('%Y-%m-%d'),
            'strike_tipo_timestamp': f"{row['Strike']}_{row['Tipo']}_{datetime.now(timezone.utc).isoformat()}",
            'strike': Decimal(str(row['Strike'])),
            'tipo': row['Tipo'],
            'precio_anterior': Decimal(str(row['Anterior']))
        }

        tabla_opciones.put_item(Item=item)

def guardar_futuros(df_futuros):
    for _, row in df_futuros.iterrows():
        item = {
            'fecha_vencimiento': row['Vencimiento'].strftime('%Y-%m-%d'),
            'tipo_contrato': row['Tipo'],
            'timestamp': datetime.utcnow().isoformat()
        }

        for k in ['Anterior', 'Último', 'Volumen', 'Apertura', 'Máximo', 'Mínimo']:
            val = row.get(k)
            if val and isinstance(val, str):
                val = val.replace('.', '').replace(',', '.')
                try:
                    item[k.lower()] = Decimal(val)
                except:
                    pass

        tabla_futuros.put_item(Item=item)

def publicar_evento_fin_scraping():
    client = boto3.client('events', region_name='eu-west-1')
    response = client.put_events(
        Entries=[
            {
                'Source': 'custom.miniibex.scraping',
                'DetailType': 'Scraping Completed',
                'Detail': json.dumps({'message': 'Scraping y guardado finalizado correctamente.'}),
                'EventBusName': 'custom-mini-ibex-bus'
            }
        ]
    )
    logger.info(f"Evento lanzado a EventBridge: {response}")

def lambda_handler(event, context):
    logger.info("Lambda scrapingGuardarDynamo ha iniciado.")

    df_opciones, df_futuros = obtener_datos_meff()

    guardar_opciones(df_opciones)
    guardar_futuros(df_futuros)

    logger.info(f"{len(df_opciones)} opciones y {len(df_futuros)} futuros guardados en DynamoDB.")

    # Lanzar evento custom que desencadene la Lambda calcularVolatilidad
    publicar_evento_fin_scraping()

    return {
        'statusCode': 200,
        'body': 'Scraping, guardado y evento lanzado correctamente.'
    }
