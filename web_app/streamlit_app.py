import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import boto3
import io
import numpy as np
from datetime import datetime
import base64
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from scipy.interpolate import griddata

# Estilo matplotlib oscuro
plt.style.use('dark_background')

# Configuración de la página
st.set_page_config(
    page_title="Skew de Volatilidad MINI IBEX",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilo CSS profesional
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }
    .main-header { font-size: 2.5rem; font-weight: 800; color: #FF6B8B; margin-bottom: 1.5rem; }
    .sub-header { font-size: 1.4rem; font-weight: 600; color: #FF6B8B; margin-top: 1rem; margin-bottom: 1rem; }
    .stButton>button {
        background-color: #FF6B8B;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #FF4066;
        transform: scale(1.02);
    }
    .stat-box {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-label {
        font-size: 0.8rem;
        color: #AAAAAA;
        margin-bottom: 2px;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-box {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #AAAAAA;
    }
    .metric-value {
        font-size: 0.9rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .dataframe th {
        background-color: #1E1E1E !important;
        color: #FF6B8B !important;
    }
    .dataframe td {
        background-color: #121212 !important;
        color: #FFFFFF !important;
    }
    .date-selector {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .download-btn {
        display: inline-block;
        padding: 8px 12px;
        background-color: #FF6B8B;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        margin: 5px 5px 5px 0px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
    }
    .download-btn:hover {
        background-color: #FF4066;
        transform: scale(1.02);
    }
    .stRadio > div {
        padding: 5px 10px;
        background-color: #1E1E1E;
        border-radius: 8px;
    }
    .stRadio [role=radiogroup] {
        display: flex;
        justify-content: space-between;
    }
    </style>
""", unsafe_allow_html=True)

# --- Obtener datos desde DynamoDB ---
@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def cargar_datos_dynamodb():
    # Inicializa una sesión de boto3 usando los secrets de Streamlit
    session = boto3.Session(
        aws_access_key_id     = st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"],
        region_name           = st.secrets["aws"]["region_name"]
    )
    dynamodb = session.resource("dynamodb")

    # Opciones
    tabla_opciones = dynamodb.Table('OpcionesMiniIBEX')
    opciones = tabla_opciones.scan()['Items']
    df_opciones = pd.DataFrame(opciones)
    df_opciones['strike'] = df_opciones['strike'].astype(float)
    df_opciones['precio_anterior'] = df_opciones['precio_anterior'].astype(float)
    df_opciones['fecha_vencimiento'] = pd.to_datetime(df_opciones['fecha_vencimiento'])
    df_opciones['tipo'] = df_opciones['tipo'].astype(str)

    # Volatilidad
    tabla_vol = dynamodb.Table('VolatilidadMiniIBEX')
    vol = tabla_vol.scan()['Items']
    df_vol = pd.DataFrame(vol)
    df_vol['strike'] = df_vol['strike_tipo_timestamp'].apply(lambda x: float(x.split('_')[0]))
    df_vol['tipo'] = df_vol['strike_tipo_timestamp'].apply(lambda x: x.split('_')[1])
    df_vol['volatilidad_implicita'] = df_vol['volatilidad_implicita'].astype(float)
    df_vol['fecha_vencimiento'] = pd.to_datetime(df_vol['fecha_vencimiento'])
    df_vol['scrap_datetime'] = pd.to_datetime(
        df_vol['strike_tipo_timestamp'].apply(lambda s: s.split('_', 2)[2])
    )
    df_vol['scrap_date'] = df_vol['scrap_datetime'].dt.date

    # Futuros
    tabla_futuros = dynamodb.Table('FuturosMiniIBEX')
    futuros = tabla_futuros.scan()['Items']
    df_futuros = pd.DataFrame(futuros)
    df_futuros['fecha_vencimiento'] = pd.to_datetime(df_futuros['fecha_vencimiento'])
    df_futuros['apertura'] = df_futuros['apertura'].astype(float)

    return df_opciones, df_vol, df_futuros


# --- Cálculo de estadísticas de volatilidad ---
def calcular_estadisticas(df_vol_fecha):
    stats = {}
    if not df_vol_fecha.empty:
        stats['vol_min'] = df_vol_fecha['volatilidad_implicita'].min()
        stats['vol_max'] = df_vol_fecha['volatilidad_implicita'].max()
        stats['vol_media'] = df_vol_fecha['volatilidad_implicita'].mean()
        stats['vol_mediana'] = df_vol_fecha['volatilidad_implicita'].median()

        df_calls = df_vol_fecha[df_vol_fecha['tipo'] == 'CALL']
        df_puts = df_vol_fecha[df_vol_fecha['tipo'] == 'PUT']
        if not df_calls.empty:
            stats['call_min'] = df_calls['volatilidad_implicita'].min()
            stats['call_max'] = df_calls['volatilidad_implicita'].max()
            stats['call_media'] = df_calls['volatilidad_implicita'].mean()
        if not df_puts.empty:
            stats['put_min'] = df_puts['volatilidad_implicita'].min()
            stats['put_max'] = df_puts['volatilidad_implicita'].max()
            stats['put_media'] = df_puts['volatilidad_implicita'].mean()

        # ATM skew
        atm_call = df_calls.iloc[(df_calls['strike'] - precio_futuro).abs().argsort()[:1]]
        atm_put = df_puts.iloc[(df_puts['strike'] - precio_futuro).abs().argsort()[:1]]
        if not atm_call.empty and not atm_put.empty:
            stats['atm_call_vol'] = atm_call['volatilidad_implicita'].values[0]
            stats['atm_put_vol'] = atm_put['volatilidad_implicita'].values[0]
            stats['put_call_diff'] = stats['atm_put_vol'] - stats['atm_call_vol']

        # Skew 25-delta
        if len(df_calls) > 3 and len(df_puts) > 3:
            df_calls_sorted = df_calls.sort_values('strike')
            df_puts_sorted = df_puts.sort_values('strike')
            otm_call_vol = df_calls_sorted.iloc[2]['volatilidad_implicita']
            itm_put_vol = df_puts_sorted.iloc[2]['volatilidad_implicita']
            stats['skew_25d'] = itm_put_vol - otm_call_vol

    return stats

# --- Lógica principal ---
with st.spinner("Cargando datos desde DynamoDB..."):
    df_opciones, df_vol, df_futuros = cargar_datos_dynamodb()

# Filtrar por fechas futuras
hoy = pd.to_datetime(datetime.utcnow().date())
df_opciones = df_opciones[df_opciones['fecha_vencimiento'] >= hoy]
df_vol      = df_vol[df_vol['fecha_vencimiento'] >= hoy]
df_futuros  = df_futuros[df_futuros['fecha_vencimiento'] >= hoy]

# Precio futuro más cercano
precio_futuro = df_futuros.sort_values('fecha_vencimiento').iloc[0]['apertura']

# Fechas disponibles
fechas_disponibles = df_opciones['fecha_vencimiento'].drop_duplicates().sort_values()

# Funciones de descarga
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', transparent=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def download_button(obj, filename, text):
    if isinstance(obj, plt.Figure):
        img_str = fig_to_base64(obj)
        href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 {text}</a>'
    else:
        csv = obj.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {text}</a>'
    st.markdown(f"<div class='download-btn'>{href}</div>", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Skew de Volatilidad Implícita - MINI IBEX</div>', unsafe_allow_html=True)

# Selector de fechas
st.markdown('<div class="date-selector">', unsafe_allow_html=True)
fechas_seleccionadas = st.multiselect(
    "🗓️ Selecciona fechas de vencimiento para comparar:",
    options=fechas_disponibles,
    default=[fechas_disponibles.iloc[0]] if not fechas_disponibles.empty else []
)
if not fechas_seleccionadas and not fechas_disponibles.empty:
    fechas_seleccionadas = [fechas_disponibles.iloc[0]]
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    mostrar_calls  = st.checkbox("Mostrar CALLs",     value=True)
with col2:
    mostrar_puts   = st.checkbox("Mostrar PUTs",      value=True)
with col3:
    mostrar_futuro = st.checkbox("Mostrar futuro",    value=True)
with col4:
    tipo_grafico   = st.radio("Tipo de gráfico:", ["Skew 2D", "Superficie 3D"], horizontal=True)
with col5:
    st.markdown(f"**Futuro:** {precio_futuro}")
st.markdown('</div>', unsafe_allow_html=True)

# Layout
g1, g2 = st.columns([3, 1])

with g1:
    if tipo_grafico == "Skew 2D":
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#1E1E1E')

        if mostrar_futuro:
            ax.axvline(precio_futuro, linestyle='--', color='#FF4066', alpha=0.7, label=f'Futuro ({precio_futuro})')

        # colormaps
        total_lines = sum(len(df_vol[df_vol['fecha_vencimiento']==fv]['scrap_date'].unique()) for fv in fechas_seleccionadas)
        cmap_calls = cm.get_cmap('Reds', total_lines+1)
        cmap_puts  = cm.get_cmap('Blues', total_lines+1)
        line_idx = 0

        for fv in fechas_seleccionadas:
            df_fv = df_vol[df_vol['fecha_vencimiento']==fv]
            fecha_str = fv.strftime('%d/%m/%Y')
            for scrap in sorted(df_fv['scrap_date'].unique()):
                df_day = df_fv[df_fv['scrap_date']==scrap]
                scrap_str = scrap.strftime('%d/%m/%Y')

                if mostrar_calls:
                    df_c = df_day[df_day['tipo']=='CALL']
                    ax.plot(df_c['strike'], df_c['volatilidad_implicita'],
                            label=f"CALL {fecha_str} @ {scrap_str}",
                            marker='o', linewidth=2, markersize=5,
                            color=cmap_calls(line_idx), alpha=0.8)
                if mostrar_puts:
                    df_p = df_day[df_day['tipo']=='PUT']
                    ax.plot(df_p['strike'], df_p['volatilidad_implicita'],
                            label=f"PUT {fecha_str} @ {scrap_str}",
                            marker='s', linewidth=2, markersize=5,
                            color=cmap_puts(line_idx), alpha=0.8)
                line_idx += 1

        ax.set_title("Comparativa de Volatility Skew por Vencimiento y Día", fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel("Strike", fontsize=12, color='white')
        ax.set_ylabel("Volatilidad Implícita (%)", fontsize=12, color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='#444444')
        ax.legend(fontsize=8, loc='best', facecolor='#1E1E1E', edgecolor='#444444', labelcolor='white')
        plt.tight_layout()
        st.pyplot(fig)
        download_button(fig, "volatility_skew.png", "Descargar gráfico")

    else:
        # Superficie 3D (igual que antes)...
        # [mantener tu lógica de superficie 3D sin cambios]
        if len(fechas_seleccionadas) > 1:
            strikes_all, vols_all, dates_all, types_all = [], [], [], []
            for fv in fechas_seleccionadas:
                df_fv = df_vol[df_vol['fecha_vencimiento']==fv]
                for t in ['CALL','PUT']:
                    if (t=='CALL' and mostrar_calls) or (t=='PUT' and mostrar_puts):
                        df_t = df_fv[df_fv['tipo']==t]
                        strikes_all.extend(df_t['strike'])
                        vols_all.extend(df_t['volatilidad_implicita'])
                        dates_all.extend([mdates.date2num(fv.to_pydatetime())]*len(df_t))
                        types_all.extend([t]*len(df_t))

            if strikes_all:
                fig = plt.figure(figsize=(10,6))
                fig.patch.set_facecolor('#121212')
                ax3 = fig.add_subplot(111, projection='3d')
                ax3.set_facecolor('#1E1E1E')

                ci = [i for i,t in enumerate(types_all) if t=='CALL']
                pi = [i for i,t in enumerate(types_all) if t=='PUT']
                if mostrar_calls and ci:
                    ax3.scatter([strikes_all[i] for i in ci],
                                [dates_all[i] for i in ci],
                                [vols_all[i] for i in ci],
                                c='red', label='CALL', alpha=0.7)
                if mostrar_puts and pi:
                    ax3.scatter([strikes_all[i] for i in pi],
                                [dates_all[i] for i in pi],
                                [vols_all[i] for i in pi],
                                c='blue', label='PUT', alpha=0.7)

                xi = np.linspace(min(strikes_all), max(strikes_all), 100)
                yi = np.linspace(min(dates_all), max(dates_all), 100)
                X, Y = np.meshgrid(xi, yi)
                if mostrar_calls and ci:
                    Z1 = griddata(( [strikes_all[i] for i in ci], [dates_all[i] for i in ci] ),
                                  [vols_all[i] for i in ci], (X,Y), method='cubic')
                    ax3.plot_surface(X,Y,Z1, cmap='Reds', alpha=0.3, linewidth=0)
                if mostrar_puts and pi:
                    Z2 = griddata(( [strikes_all[i] for i in pi], [dates_all[i] for i in pi] ),
                                  [vols_all[i] for i in pi], (X,Y), method='cubic')
                    ax3.plot_surface(X,Y,Z2, cmap='Blues', alpha=0.3, linewidth=0)

                ax3.set_xlabel('Strike', color='white')
                ax3.set_ylabel('Fecha', color='white')
                ax3.set_zlabel('Vol Impl (%)', color='white')
                ax3.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
                ax3.tick_params(colors='white')
                ax3.set_title("Superficie de Volatilidad", color='white', fontsize=14, fontweight='bold')
                ax3.view_init(elev=30, azim=-45)
                plt.tight_layout()
                st.pyplot(fig)
                download_button(fig, "volatility_surface.png", "Descargar superficie")
            else:
                st.warning("No hay suficientes datos para la superficie 3D")
        else:
            st.warning("Selecciona al menos dos vencimientos para la superficie 3D")

with g2:
    if fechas_seleccionadas:
        fecha_main = fechas_seleccionadas[0]
        df_main = df_vol[df_vol['fecha_vencimiento']==fecha_main]
        stats = calcular_estadisticas(df_main)

        st.markdown(f'<div class="sub-header">📊 Estadísticas {fecha_main.strftime("%d/%m/%Y")}</div>', unsafe_allow_html=True)
        if stats:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-box"><span class="metric-label">Vol Media:</span>'
                            f'<span class="metric-value">{stats["vol_media"]:.2f}%</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box"><span class="metric-label">Vol Min:</span>'
                            f'<span class="metric-value">{stats["vol_min"]:.2f}%</span></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-box"><span class="metric-label">Vol Max:</span>'
                            f'<span class="metric-value">{stats["vol_max"]:.2f}%</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box"><span class="metric-label">Vol Mediana:</span>'
                            f'<span class="metric-value">{stats["vol_mediana"]:.2f}%</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="sub-header">📉 Métricas de Skew</div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            if 'put_call_diff' in stats:
                st.markdown(f'<div class="metric-box"><span class="metric-label">Put-Call ATM:</span>'
                            f'<span class="metric-value">{stats["put_call_diff"]:.2f}%</span></div>', unsafe_allow_html=True)
            if 'skew_25d' in stats:
                st.markdown(f'<div class="metric-box"><span class="metric-label">Skew 25Δ:</span>'
                            f'<span class="metric-value">{stats["skew_25d"]:.2f}%</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="sub-header">🔄 CALL vs PUT</div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            cc, pp = st.columns(2)
            with cc:
                if 'call_media' in stats:
                    st.markdown(f"<div style='text-align:center; color:#FF6347;'><strong>CALL</strong><br>{stats['call_media']:.2f}%</div>", unsafe_allow_html=True)
            with pp:
                if 'put_media' in stats:
                    st.markdown(f"<div style='text-align:center; color:#00BFFF;'><strong>PUT</strong><br>{stats['put_media']:.2f}%</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"<div style='font-size:0.8rem; color:#888888;'>Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>", unsafe_allow_html=True)
        else:
            st.info("No hay datos suficientes para estadísticas.")

# Tabla comparativa
st.markdown('<div class="sub-header">📋 Tabla de Datos</div>', unsafe_allow_html=True)
if fechas_seleccionadas:
    dfs = []
    for fv in fechas_seleccionadas:
        tmp = df_vol[df_vol['fecha_vencimiento']==fv].copy()
        tmp['fecha_str'] = tmp['fecha_vencimiento'].dt.strftime('%d/%m/%Y')
        dfs.append(tmp)
    df_comb = pd.concat(dfs) if dfs else pd.DataFrame()
    if not df_comb.empty:
        pivot = pd.pivot_table(
            df_comb,
            values='volatilidad_implicita',
            index=['strike','tipo'],
            columns='fecha_str',
            aggfunc='first'
        ).reset_index()
        fmt = pivot.copy()
        for col in fmt.columns:
            if col not in ['strike','tipo']:
                fmt[col] = fmt[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        st.dataframe(fmt.sort_values(['tipo','strike']), use_container_width=True, hide_index=True)
        download_button(pivot, "volatility_data.csv", "Descargar datos CSV")
    else:
        st.info("No hay datos para mostrar.")
else:
    st.info("Selecciona al menos una fecha.")

# Footer
st.markdown("<div style='text-align:center; color:#888888; margin-top:20px;'>© 2025 - Dashboard de Volatilidad MINI IBEX by PaulaG</div>", unsafe_allow_html=True)
