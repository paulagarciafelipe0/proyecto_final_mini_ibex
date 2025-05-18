import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from decimal import Decimal

# --- Inicializar conexión a DynamoDB ---
dynamodb = boto3.resource('dynamodb', region_name='eu-west-1')
tabla_opciones = dynamodb.Table('OpcionesMiniIBEX')
tabla_futuros = dynamodb.Table('FuturosMiniIBEX')
tabla_volatilidad = dynamodb.Table('VolatilidadMiniIBEX')

# --- Funciones estadísticas sin scipy ---
def norm_cdf(x):
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    L = abs(x)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - (1.0 / np.sqrt(2 * np.pi)) * np.exp(-L**2 / 2.0) * (a1*K + a2*K**2 + a3*K**3 + a4*K**4 + a5*K**5)
    return w if x >= 0 else 1.0 - w

def norm_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)

def bs_price(S, K, T, sigma, tipo):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == 'CALL':
        return S * norm_cdf(d1) - K * norm_cdf(d2)
    elif tipo == 'PUT':
        return K * norm_cdf(-d2) - S * norm_cdf(-d1)
    return None

def calcular_volatilidad(S, K, T, tipo, precio_opcion, tol=1e-5, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        price = bs_price(S, K, T, sigma, tipo)
        if price is None:
            return None
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        vega = S * norm_pdf(d1) * np.sqrt(T)
        if vega == 0:
            return None
        increment = (price - precio_opcion) / vega
        sigma -= increment
        if abs(increment) < tol:
            return round(sigma * 100, 2)
    return None

# --- Lambda principal optimizado ---
def lambda_handler(event, context):
    try:
        opciones = tabla_opciones.scan()['Items']
        futuros = tabla_futuros.scan()['Items']

        if not opciones or not futuros:
            print("No hay datos.")
            return {'statusCode': 400, 'body': 'No hay datos suficientes en las tablas.'}

        df_opciones = pd.DataFrame(opciones)
        df_futuros = pd.DataFrame(futuros)

        df_opciones['strike'] = df_opciones['strike'].astype(float)
        df_opciones['precio_anterior'] = df_opciones['precio_anterior'].astype(float)
        df_opciones['fecha_vencimiento'] = pd.to_datetime(df_opciones['fecha_vencimiento'])

        hoy_iso = datetime.utcnow().date().isoformat()
        df_opciones = df_opciones[df_opciones['strike_tipo_timestamp'].str.contains(hoy_iso)]

        if df_opciones.empty:
            print("No hay nuevas opciones hoy.")
            return {'statusCode': 200, 'body': 'Sin registros nuevos hoy.'}

        df_futuros['fecha_vencimiento'] = pd.to_datetime(df_futuros['fecha_vencimiento'])
        df_futuros = df_futuros[df_futuros['fecha_vencimiento'] > datetime.utcnow()]
        S = float(df_futuros.sort_values('fecha_vencimiento').iloc[0]['último'])

        hoy = datetime.utcnow().date()
        contador = 1

        for row in df_opciones.itertuples(index=False):
            try:
                timestamp_str = row.strike_tipo_timestamp.split('_')[-1]
                if datetime.fromisoformat(timestamp_str.replace('Z', '')).date() != hoy:
                    continue

                T = (row.fecha_vencimiento - datetime.utcnow()).days / 365
                if T <= 0:
                    continue

                K = row.strike
                tipo = row.tipo.strip().upper()
                precio = row.precio_anterior

                vol = calcular_volatilidad(S, K, T, tipo, precio)
                if vol is None or vol <= 0 or vol > 300:
                    continue

                item = {
                    'fecha_vencimiento': row.fecha_vencimiento.strftime('%Y-%m-%d'),
                    'strike_tipo_timestamp': f"{K}_{tipo}_{datetime.now(timezone.utc).isoformat()}",
                    'timestamp': datetime.utcnow().isoformat(),
                    'volatilidad_implicita': Decimal(str(vol))
                }

                tabla_volatilidad.put_item(Item=item)
                print(f"Volatilidad {contador} calculada y guardada - Strike {K}, Tipo {tipo}, Vol: {vol:.2f}%")
                contador += 1

            except Exception as e:
                print(f"Error fila: {e}")

        return {'statusCode': 200, 'body': f'{contador - 1} volatilidades calculadas y guardadas.'}

    except Exception as e:
        print(f"Error general: {e}")
        return {'statusCode': 500, 'body': str(e)}




