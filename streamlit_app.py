"""
Dashboard de Precios Agrícolas
Visualización simple de precios históricos de productos
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Precios Agrícolas",
    page_icon="🌾",
    layout="wide"
)

# Título
st.title("🌾 Precios Históricos de Productos Agrícolas")
st.markdown("Visualiza la evolución de precios de productos agrícolas en el tiempo")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw_prices.csv')
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    return df

try:
    df = load_data()

    # Sidebar - Filtros
    st.sidebar.header("Filtros")

    # Selector de productos
    productos_disponibles = sorted(df['variety'].unique())
    productos_seleccionados = st.sidebar.multiselect(
        "Selecciona productos",
        options=productos_disponibles,
        default=['tomate', 'papa_blanca', 'cebolla_blanca'] if 'tomate' in productos_disponibles else [productos_disponibles[0]]
    )

    # Filtro de fechas
    min_date = df['publication_date'].min()
    max_date = df['publication_date'].max()

    fecha_inicio = st.sidebar.date_input(
        "Fecha inicial",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    fecha_fin = st.sidebar.date_input(
        "Fecha final",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    # Filtro de unidad
    unidades = sorted(df['unit'].dropna().unique())
    unidad_seleccionada = st.sidebar.selectbox(
        "Unidad",
        options=['Todas'] + unidades,
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    **Información de datos:**
    - Total registros: {len(df):,}
    - Productos: {df['variety'].nunique()}
    - Período: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}
    """)

    # Filtrar datos
    df_filtered = df.copy()

    if productos_seleccionados:
        df_filtered = df_filtered[df_filtered['variety'].isin(productos_seleccionados)]

    df_filtered = df_filtered[
        (df_filtered['publication_date'] >= pd.to_datetime(fecha_inicio)) &
        (df_filtered['publication_date'] <= pd.to_datetime(fecha_fin))
    ]

    if unidad_seleccionada != 'Todas':
        df_filtered = df_filtered[df_filtered['unit'] == unidad_seleccionada]

    # Verificar si hay datos
    if len(df_filtered) == 0:
        st.warning("No hay datos para los filtros seleccionados")
    else:
        # Métricas resumidas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Registros", f"{len(df_filtered):,}")

        with col2:
            precio_promedio = df_filtered['price'].mean()
            st.metric("Precio Promedio", f"₡{precio_promedio:,.0f}")

        with col3:
            precio_min = df_filtered['price'].min()
            st.metric("Precio Mínimo", f"₡{precio_min:,.0f}")

        with col4:
            precio_max = df_filtered['price'].max()
            st.metric("Precio Máximo", f"₡{precio_max:,.0f}")

        st.markdown("---")

        # Gráfico principal - Serie de tiempo
        st.subheader("📈 Evolución de Precios")

        if productos_seleccionados:
            # Agregar por fecha y producto para promediar múltiples registros del mismo día
            df_plot = df_filtered.groupby(['publication_date', 'variety'])['price'].mean().reset_index()

            fig = px.line(
                df_plot,
                x='publication_date',
                y='price',
                color='variety',
                title='Precio por Fecha',
                labels={
                    'publication_date': 'Fecha',
                    'price': 'Precio (₡)',
                    'variety': 'Producto'
                },
                height=500
            )

            fig.update_layout(
                hovermode='x unified',
                xaxis_title='Fecha',
                yaxis_title='Precio (₡)',
                legend_title='Producto'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecciona al menos un producto para ver la gráfica")

        # Tabla de estadísticas por producto
        if productos_seleccionados:
            st.subheader("📊 Estadísticas por Producto")

            stats = df_filtered.groupby('variety')['price'].agg([
                ('Promedio', 'mean'),
                ('Mediana', 'median'),
                ('Mínimo', 'min'),
                ('Máximo', 'max'),
                ('Desv. Estándar', 'std'),
                ('Registros', 'count')
            ]).round(2)

            # Formatear los valores monetarios
            for col in ['Promedio', 'Mediana', 'Mínimo', 'Máximo', 'Desv. Estándar']:
                stats[col] = stats[col].apply(lambda x: f"₡{x:,.2f}")

            stats['Registros'] = stats['Registros'].astype(int)

            st.dataframe(stats, use_container_width=True)

        # Sección de datos crudos (colapsable)
        with st.expander("🔍 Ver datos crudos"):
            st.dataframe(
                df_filtered[['publication_date', 'variety', 'price', 'unit', 'NOMBRE']].sort_values('publication_date', ascending=False),
                use_container_width=True
            )

            # Botón de descarga
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Descargar datos filtrados (CSV)",
                data=csv,
                file_name=f"precios_agricolas_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

except FileNotFoundError:
    st.error("❌ No se encontró el archivo de datos 'data/raw_prices.csv'")
    st.info("Asegúrate de que el archivo existe en la ubicación correcta")
except Exception as e:
    st.error(f"❌ Error al cargar los datos: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Dashboard de Precios Agrícolas | Datos reales sin procesamiento ML
</div>
""", unsafe_allow_html=True)
