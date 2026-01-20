# Análisis de Precios Agrícolas - CNP

> **Un análisis visual e interactivo de 4 años de datos históricos (2021-2024) del Consejo Nacional de Producción.**

## Acerca de Este Proyecto

Este dashboard presenta un **análisis transparente y riguroso de los precios agrícolas** registrados por el CNP entre 2021 y 2024. Explora patrones reales, estacionalidad y anomalías en 9,184 registros de 56 productos diferentes.

✅ **Datos del CNP** - 9,184 registros históricos de 56 productos agrícolas  
✅ **Transparencia Total** - Visualización honesta sin predicciones especulativas  
✅ **Análisis Interactivo** - Filtros dinámicos para explorar subcategorías  
✅ **Datos Reales** - Directamente del Consejo Nacional de Producción  

##  Características

### Panel Principal
- **Gráfica de Serie Temporal** interactiva (2021-2024)
- **Filtros dinámicos**: productos, fechas
- **Tabla de estadísticas** por producto (media, mediana, mín, máx, desv. est.)

### Funcionalidades
- Descarga de datos filtrados a CSV
- Visualización interactiva con Plotly
- Estadísticas detalladas por producto
- Acceso a datos crudos

## Tech Stack

```
Frontend:       Streamlit 1.52.1
Visualization:  Plotly (Express)
Data:           Pandas, NumPy
Language:       Python 3.12.3
Dataset:        9,184 registros | 56 productos | 2021-2024
```

## ⚡ Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run src/dashboard.py

# Visit → http://localhost:8501
```

## Estructura

```
market-price-ml/
├── src/
│   └── dashboard.py         # App principal
├── data/
│   └── raw_prices.csv       # Dataset CNP
├── requirements.txt
└── README.md
```

## Dataset

**Fuente:** Consejo Nacional de Producción (CNP) - Costa Rica

| Campo | Descripción |
|-------|------------|
| `publication_date` | Fecha (YYYY-MM-DD) |
| `variety` | Producto agrícola |
| `price` | Precio en ₡ |
| `unit` | Unidad de medida |

**Cobertura**: 2021-2024 | **Registros**: 9,184 | **Productos**: 56
