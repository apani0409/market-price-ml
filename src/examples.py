"""
EJEMPLOS PRÁCTICOS: Cómo usar y mejorar el modelo

Este archivo contiene ejemplos de código para:
1. Analizar resultados del modelo
2. Mejorar rendimiento
3. Usar el modelo en producción
"""

# ============================================================================
# 1. INSPECCIONAR PREDICCIONES
# ============================================================================

def example_1_inspect_predictions():
    """
    Cómo ver qué está prediciendo el modelo en detalle.
    """
    from src.data_loader import load_data
    from src.preprocessing import preprocess_pipeline
    from src.features import feature_engineering_pipeline
    from src.model import train_and_evaluate, prepare_features
    import pandas as pd
    
    # Cargar y procesar
    df = load_data()
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    
    # Entrenar modelo
    model, results, test_df = train_and_evaluate(df, model_type='random_forest')
    
    # Obtener predicciones
    X_test, y_test = prepare_features(test_df)
    predictions = model.predict(X_test)
    
    # Crear DataFrame con resultados
    results_df = test_df.copy()
    results_df['predicted_price'] = predictions
    results_df['error'] = abs(predictions - test_df['mean_price'].values)
    results_df['error_pct'] = (results_df['error'] / test_df['mean_price'].values) * 100
    
    print("=" * 80)
    print("EJEMPLOS DE PREDICCIONES")
    print("=" * 80)
    
    # Las mejores predicciones
    print("\n✅ TOP 10 MEJORES PREDICCIONES:")
    best = results_df.nsmallest(10, 'error')[
        ['variety', 'year', 'week', 'mean_price', 'predicted_price', 'error', 'error_pct']
    ]
    print(best.to_string(index=False))
    
    # Las peores predicciones
    print("\n❌ TOP 10 PEORES PREDICCIONES:")
    worst = results_df.nlargest(10, 'error')[
        ['variety', 'year', 'week', 'mean_price', 'predicted_price', 'error', 'error_pct']
    ]
    print(worst.to_string(index=False))
    
    # Análisis por producto
    print("\n🌾 ERROR PROMEDIO POR PRODUCTO:")
    by_variety = results_df.groupby('variety').agg({
        'error': 'mean',
        'error_pct': 'mean',
        'mean_price': ['mean', 'std', 'min', 'max']
    }).round(2).sort_values(('error', 'mean'), ascending=False)
    print(by_variety)
    
    return results_df


# ============================================================================
# 2. COMPARAR MODELOS
# ============================================================================

def example_2_compare_models():
    """
    Cómo comparar RandomForest vs LinearRegression vs otros modelos.
    """
    from src.data_loader import load_data
    from src.preprocessing import preprocess_pipeline
    from src.features import feature_engineering_pipeline
    from src.model import train_and_evaluate
    import pandas as pd
    
    df = load_data()
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    
    print("=" * 80)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 80)
    
    models_to_test = ['random_forest', 'linear_regression']
    results_list = []
    
    for model_type in models_to_test:
        print(f"\n⏳ Entrenando {model_type}...")
        model, results, test_df = train_and_evaluate(
            df, 
            model_type=model_type,
            test_size=0.2
        )
        
        results_list.append({
            'Modelo': model_type,
            'MAE': f"{results['mae']:.2f}",
            'RMSE': f"{results['rmse']:.2f}",
            'Ratio RMSE/MAE': f"{results['rmse']/results['mae']:.2f}",
            'Test Samples': len(test_df)
        })
    
    comparison_df = pd.DataFrame(results_list)
    print("\n📊 RESULTADOS:")
    print(comparison_df.to_string(index=False))
    
    # Determinar el mejor
    rf_mae = float(results_list[0]['MAE'])
    lr_mae = float(results_list[1]['MAE'])
    
    if rf_mae < lr_mae:
        print(f"\n✅ RandomForest es mejor ({(lr_mae-rf_mae)/lr_mae*100:.1f}% mejor)")
    else:
        print(f"\n✅ LinearRegression es mejor ({(rf_mae-lr_mae)/rf_mae*100:.1f}% mejor)")


# ============================================================================
# 3. MEJORAR EL MODELO
# ============================================================================

def example_3_improve_model():
    """
    Ideas para mejorar el rendimiento del modelo.
    """
    from src.data_loader import load_data
    from src.preprocessing import preprocess_pipeline
    from src.features import feature_engineering_pipeline
    from src.model import train_and_evaluate, prepare_features
    import pandas as pd
    import numpy as np
    
    df = load_data()
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df, rolling_window=4)
    
    # Modelo base
    model_base, results_base, _ = train_and_evaluate(df)
    print("=" * 80)
    print("MEJORAMIENTO DEL MODELO")
    print("=" * 80)
    print(f"\n📊 Baseline MAE: {results_base['mae']:.2f}")
    
    # Idea 1: Aumentar rolling window
    print("\n💡 IDEA 1: Aumentar rolling_window (detectar patrones a más largo plazo)")
    for window in [6, 8, 10]:
        df_test = feature_engineering_pipeline(df.iloc[:], rolling_window=window)
        _, results_test, _ = train_and_evaluate(df_test)
        improvement = (results_base['mae'] - results_test['mae']) / results_base['mae'] * 100
        print(f"   rolling_window={window}: MAE={results_test['mae']:.2f} ({improvement:+.1f}%)")
    
    # Idea 2: Agregar features adicionales
    print("\n💡 IDEA 2: Agregar features adicionales")
    print("   Posibles features:")
    print("   • lag_1_price: Precio de la semana anterior")
    print("   • lag_2_price: Precio de hace 2 semanas")
    print("   • price_trend: Tendencia (precio actual - promedio)")
    print("   • seasonal_multiplier: Factor estacional por producto")
    
    # Idea 3: Usar diferentes tamaños de test
    print("\n💡 IDEA 3: Ajustar train/test split")
    for test_size in [0.1, 0.2, 0.3, 0.4]:
        _, results_test, _ = train_and_evaluate(df, test_size=test_size)
        improvement = (results_base['mae'] - results_test['mae']) / results_base['mae'] * 100
        print(f"   test_size={test_size:.1%}: MAE={results_test['mae']:.2f} ({improvement:+.1f}%)")


# ============================================================================
# 4. PRODUCCIÓN: GUARDAR Y CARGAR MODELO
# ============================================================================

def example_4_save_model():
    """
    Cómo guardar el modelo entrenado para usar después.
    """
    from src.data_loader import load_data
    from src.preprocessing import preprocess_pipeline
    from src.features import feature_engineering_pipeline
    from src.model import train_and_evaluate
    import pickle
    from pathlib import Path
    
    print("=" * 80)
    print("GUARDAR MODELO ENTRENADO")
    print("=" * 80)
    
    # Entrenar
    df = load_data()
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    model, results, _ = train_and_evaluate(df)
    
    # Crear carpeta de modelos
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Guardar modelo
    model_path = model_dir / 'agricultural_price_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Modelo guardado en: {model_path}")
    print(f"   Tamaño del archivo: {model_path.stat().st_size / 1024:.1f} KB")
    
    # Guardar métricas
    metrics_path = model_dir / 'model_metrics.txt'
    from datetime import datetime
    with open(metrics_path, 'w') as f:
        f.write(f"RandomForest Agricultural Price Prediction Model\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"\nPerformance Metrics:\n")
        f.write(f"MAE: {results['mae']:.2f}\n")
        f.write(f"RMSE: {results['rmse']:.2f}\n")
    
    print(f"✅ Métricas guardadas en: {metrics_path}")
    
    # Cargar y usar
    print("\n📥 Cargando modelo guardado...")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    print("✅ Modelo cargado exitosamente!")
    return loaded_model


# ============================================================================
# 5. ANÁLISIS POR PRODUCTO: INVESTIGAR TOMATE
# ============================================================================

def example_5_analyze_product():
    """
    Cómo investigar un producto específico que tiene errores altos.
    """
    from src.data_loader import load_data
    import pandas as pd
    
    print("=" * 80)
    print("ANÁLISIS DETALLADO: TOMATE (Producto con 21% de error)")
    print("=" * 80)
    
    df = load_data()
    
    # Filtrar tomate
    tomate = df[df['variety'].str.contains('tomate', case=False, na=False)]
    
    print(f"\n📊 ESTADÍSTICAS DE TOMATE:")
    print(f"   Registros totales: {len(tomate)}")
    print(f"   Años cubiertos: {sorted(tomate['year'].unique())}")
    print(f"   Semanas por año: {tomate.groupby('year')['week'].nunique()}")
    
    # Análisis por año
    print(f"\n📈 PRECIO POR AÑO:")
    by_year = tomate.groupby('year').agg({
        'price': ['count', 'mean', 'std', 'min', 'max']
    }).round(2)
    print(by_year)
    
    # Variabilidad
    print(f"\n🔄 VARIABILIDAD TEMPORAL:")
    price_by_week = tomate.groupby('week')['price'].agg(['mean', 'std', 'count'])
    print(f"   Semanas con mayor variabilidad (std dev):")
    print(price_by_week.nlargest(5, 'std')[['mean', 'std', 'count']])
    
    # Variantes
    print(f"\n🌾 VARIANTES DE TOMATE:")
    tomate_varieties = tomate['variety'].value_counts()
    print(tomate_varieties)
    
    # Recomendaciones
    print("\n💡 POSIBLES CAUSAS DEL ALTO ERROR:")
    print("   1. Alta variabilidad estacional (precios cambian mucho)")
    print("   2. Múltiples variedades con diferentes patrones de precio")
    print("   3. Cambios abruptos (plagas, clima, demanda)")
    
    print("\n✅ SOLUCIONES:")
    print("   1. Entrenar modelo separado para tomate")
    print("   2. Agregar features de estacionalidad (mes, época del año)")
    print("   3. Agregar datos externos (clima, demanda)")


# ============================================================================
# 6. PREDICCIÓN EN TIEMPO REAL
# ============================================================================

def example_6_predict_new():
    """
    Cómo hacer predicciones en nuevos datos (en producción).
    """
    from src.data_loader import load_data
    from src.preprocessing import preprocess_pipeline
    from src.features import feature_engineering_pipeline
    from src.model import train_and_evaluate
    import pickle
    import numpy as np
    
    print("=" * 80)
    print("PREDICCIÓN EN TIEMPO REAL")
    print("=" * 80)
    
    # Entrenar modelo
    df = load_data()
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    model, _, _ = train_and_evaluate(df)
    
    # Simular nuevo dato para predicción
    new_data = np.array([
        [2024, 50, 50, 1200.5, 100.2]  # year, week, week_of_year, rolling_mean, rolling_std
    ])
    
    print("\n🎯 NUEVO DATO PARA PREDECIR:")
    feature_names = ['year', 'week', 'week_of_year', 'rolling_mean_price', 'rolling_std_price']
    for name, value in zip(feature_names, new_data[0]):
        print(f"   {name}: {value}")
    
    # Hacer predicción
    prediction = model.predict(new_data)[0]
    
    print(f"\n🔮 PREDICCIÓN:")
    print(f"   Precio esperado: ${prediction:.2f}")
    print(f"   Error esperado: ±$96.20 (MAE)")
    print(f"   Rango confiable: ${prediction-96.20:.2f} a ${prediction+96.20:.2f}")
    
    return prediction


# ============================================================================
# MAIN: Ejecutar todos los ejemplos
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("EJEMPLOS DE USO Y MEJORA DEL MODELO")
    print("=" * 80)
    
    import sys
    
    examples = {
        '1': ('Inspeccionar Predicciones', example_1_inspect_predictions),
        '2': ('Comparar Modelos', example_2_compare_models),
        '3': ('Mejorar el Modelo', example_3_improve_model),
        '4': ('Guardar/Cargar Modelo', example_4_save_model),
        '5': ('Analizar Tomate (Producto problema)', example_5_analyze_product),
        '6': ('Predicción en Tiempo Real', example_6_predict_new),
    }
    
    print("\nEjemplos disponibles:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            print(f"\n▶️  Ejecutando: {examples[choice][0]}")
            examples[choice][1]()
        else:
            print(f"Opción inválida: {choice}")
    else:
        print("\nUso: python -m src.examples <número>")
        print("Ejemplo: python -m src.examples 1")
