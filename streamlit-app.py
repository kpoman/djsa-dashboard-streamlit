import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# For time series decomposition
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st
from io import BytesIO

url_partediario = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQKK0HtdtrMX7fT9X0ZdOhZ8LZwFKkPKi_NaGbZgSk1SeFq0kz5H2tK48ne-wN4_YUF7Vg3ViX70aMe/pub?output=xlsx'

@st.cache_data
def get_partediario(url):
    df_partediario = pd.read_excel(url,
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        names=['cat', 'tarde_vo', 'tarde_tanque', 'tarde_prod', 'maniana_vo', 'maniana_tanque', 'maniana_prod', 'diaria_total', 'diaria_ltvo', 'entregado'],
        sheet_name=None)
    keys = list(df_partediario.keys())
    tabs = []
    for i in range(len(keys)):
        tab_name = keys[len(keys)-i-1]
        if len(tab_name.split('-')) == 2:
            tabs.append(df_partediario[tab_name])
    df_prodtambo = pd.concat(tabs, ignore_index=True)
    df_dates = df_prodtambo[df_prodtambo['cat'] == 'La Merced']['diaria_total']
    dates = df_dates.to_list()
    df_total = df_prodtambo[df_prodtambo['cat'] == 'Total'].copy()
    # Clean dates: remove '#REF!', empty, or obviously invalid values
    def is_valid_date(val):
        if pd.isnull(val):
            return False
        s = str(val).strip()
        if s == '' or s.startswith('#') or s.lower() in ['nan', 'none']:
            return False
        try:
            float(s)
            return True
        except ValueError:
            # If not a number, try parsing as date
            try:
                pd.to_datetime(s, errors='raise')
                return True
            except Exception:
                return False
    clean_dates = [d for d in dates if is_valid_date(d)]
    min_len = min(len(df_total), len(clean_dates))
    df_total = df_total.iloc[:min_len].copy()
    df_total['date'] = pd.to_datetime(clean_dates[:min_len], errors='coerce')
    df_total = df_total[df_total['diaria_total'] > 0]
    df_total['roll'] = df_total['diaria_total'].rolling(7).mean()
    df_total['ltvo_roll'] = df_total['diaria_ltvo'].rolling(7).mean()
    df_total.reset_index(drop=True, inplace=True)
    return df_total

st.title("Panel de Producción de Leche del Tambo")

df = get_partediario(url_partediario)

# Selector de rango de fechas
min_date = df['date'].min().date()
max_date = df['date'].max().date()
start_date, end_date = st.date_input(
    "Selecciona el rango de fechas:",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
df_filtrado = df.loc[mask].copy()

# KPIs
st.subheader("Indicadores Clave")
col1, col2, col3 = st.columns(3)
col1.metric("Litros Totales", f"{df_filtrado['diaria_total'].sum():,.0f} L")
col2.metric("Promedio Diario", f"{df_filtrado['diaria_total'].mean():,.0f} L")
col3.metric("Cantidad de Días", f"{df_filtrado.shape[0]}")

# Gráfico de tendencia
st.subheader("Tendencia de Producción Diaria de Leche")
fig_trend = px.line(df_filtrado, x='date', y='diaria_total', title='Producción Total Diaria de Leche', labels={'date':'Fecha', 'diaria_total':'Litros'})
fig_trend.add_scatter(x=df_filtrado['date'], y=df_filtrado['roll'], mode='lines', name='Media móvil 7 días')
st.plotly_chart(fig_trend, use_container_width=True)

# --- Descomposición de la serie temporal ---
st.subheader("Descomposición de la Serie Temporal (Tendencia, Estacionalidad, Residuo)")
if df_filtrado.shape[0] > 30:
    # Descomponer solo si hay suficientes datos
    ts = df_filtrado.groupby('date')['diaria_total'].mean().asfreq('D')
    ts = ts.interpolate()  # Completar días faltantes si los hay
    result = seasonal_decompose(ts, model='additive', period=365 if len(ts) > 365 else 30)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    result.observed.plot(ax=axes[0], title='Observado')
    result.trend.plot(ax=axes[1], title='Tendencia')
    result.seasonal.plot(ax=axes[2], title='Estacionalidad')
    result.resid.plot(ax=axes[3], title='Residuo')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    st.image(buf.getvalue(), caption='Descomposición de la producción diaria de leche', use_column_width=True)
else:
    st.info("No hay suficientes datos para la descomposición (se requieren más de 30 días)")

# Gráfico de estacionalidad (por mes)
st.subheader("Estacionalidad (Promedio Mensual)")
df_filtrado = df_filtrado.assign(month=df_filtrado['date'].dt.month)
monthly_avg = df_filtrado.groupby('month')['diaria_total'].mean().reset_index()
fig_season = px.bar(monthly_avg, x='month', y='diaria_total', labels={'month':'Mes', 'diaria_total':'Promedio de Litros'}, title='Promedio de Producción por Mes')
st.plotly_chart(fig_season, use_container_width=True)

# Descargar datos filtrados
st.download_button(
    label="Descargar datos filtrados como CSV",
    data=df_filtrado.to_csv(index=False),
    file_name='produccion_leche_filtrada.csv',
    mime='text/csv'
)
