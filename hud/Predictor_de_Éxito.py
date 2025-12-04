import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Sephora Success Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Estilos generales */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Título principal */
    .main-title {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Predicción */
    .prediction-success {
        color: #2c3e50;
        padding: 0.4rem 0.6rem;
        text-align: center;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    .prediction-fail {
        color: #2c3e50;
        padding: 0.4rem 0.6rem;
        text-align: center;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    /* Consenso */
    .consensus-strong {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .consensus-moderate {
        background: linear-gradient(135deg, #FFB75E 0%, #ED8F03 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Sidebar */
    .sidebar-metric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Documentación - Estilo profesional oscuro minimalista */
    .doc-section {
        background: #1e1e1e;
        padding: 2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        border-left: 3px solid #404040;
        color: #d4d4d4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .doc-section h2 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.8rem;
        border-bottom: 2px solid #404040;
        padding-bottom: 0.5rem;
    }
    
    .doc-section h3 {
        color: #e0e0e0;
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        font-size: 1.3rem;
    }
    
    .doc-section p {
        color: #b0b0b0;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .doc-section ul, .doc-section ol {
        color: #b0b0b0;
        line-height: 1.8;
        padding-left: 1.5rem;
    }
    
    .doc-section li {
        color: #b0b0b0;
        margin-bottom: 0.5rem;
    }
    
    .doc-section b {
        color: #ffffff;
        font-weight: 600;
    }
    
    .doc-code {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 6px;
        font-family: 'Consolas', 'Courier New', monospace;
        margin: 1rem 0;
        color: #d4d4d4;
        border: 1px solid #404040;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Carga todos los modelos entrenados"""
    models = {
        'xgb_tuned': joblib.load('models3/xgboost_tuned.pkl'),
        'lgb_tuned': joblib.load('models3/lightgbm_tuned.pkl'),
        'rf_tuned': joblib.load('models3/random_forest_tuned.pkl'),
        'xgb_baseline': joblib.load('models3/xgboost_model.pkl'),
        'lgb_baseline': joblib.load('models3/lightgbm_model.pkl'),
        'rf_baseline': joblib.load('models3/random_forest_model.pkl'),
        'imputer': joblib.load('models3/imputer.pkl')
    }
    
    # Cargar features
    with open('models3/features_list.json', 'r') as f:
        models['features'] = json.load(f)
    
    return models

@st.cache_data
def load_metrics():
    """Carga las métricas de los modelos"""
    tuned = pd.read_csv('results/tuned_metrics.csv')
    baseline = pd.read_csv('results/baseline_metrics.csv', index_col=0)
    return tuned, baseline

@st.cache_data
def load_dataset():
    """Carga el dataset para muestreo"""
    return pd.read_csv('data/products_with_target.csv')

# Cargar todo
models = load_models()
tuned_metrics, baseline_metrics = load_metrics()
df_products = load_dataset()

def create_feature_dict(
    price_usd, rating, child_count,
    limited_edition, new, online_only, out_of_stock, sephora_exclusive,
    has_sale, has_value_price, discount_pct,
    brand_product_count, brand_avg_rating, brand_avg_loves, is_top_brand,
    has_variations, variation_price_range, has_ingredients, has_highlights, num_highlights,
    exclusivity_score, is_exclusive, category_depth
):
    """Crea diccionario de features ordenado"""
    return {
        'price_usd': price_usd,
        'rating': rating,
        'child_count': child_count,
        'limited_edition': limited_edition,
        'new': new,
        'online_only': online_only,
        'out_of_stock': out_of_stock,
        'sephora_exclusive': sephora_exclusive,
        'has_sale': has_sale,
        'has_value_price': has_value_price,
        'discount_pct': discount_pct,
        'brand_product_count': brand_product_count,
        'brand_avg_rating': brand_avg_rating,
        'brand_avg_loves': brand_avg_loves,
        'is_top_brand': is_top_brand,
        'has_variations': has_variations,
        'variation_price_range': variation_price_range,
        'has_ingredients': has_ingredients,
        'has_highlights': has_highlights,
        'num_highlights': num_highlights,
        'exclusivity_score': exclusivity_score,
        'is_exclusive': is_exclusive,
        'category_depth': category_depth
    }

def predict_product(features_dict):
    """Realiza predicciones con todos los modelos"""
    # Crear DataFrame
    X = pd.DataFrame([features_dict])
    
    # Imputar (aunque no debería haber nulos)
    X_imputed = models['imputer'].transform(X)
    
    # Predicciones
    predictions = {
        'xgb_tuned': {
            'pred': models['xgb_tuned'].predict(X_imputed)[0],
            'proba': models['xgb_tuned'].predict_proba(X_imputed)[0]
        },
        'lgb_tuned': {
            'pred': models['lgb_tuned'].predict(X_imputed)[0],
            'proba': models['lgb_tuned'].predict_proba(X_imputed)[0]
        },
        'rf_tuned': {
            'pred': models['rf_tuned'].predict(X_imputed)[0],
            'proba': models['rf_tuned'].predict_proba(X_imputed)[0]
        }
    }
    
    return predictions

def generate_random_product():
    """Genera un producto aleatorio del dataset (sesgo hacia exitosos)"""
    # Sesgar 60% hacia productos exitosos
    if np.random.random() < 0.6 and 'is_successful' in df_products.columns:
        successful_products = df_products[df_products['is_successful'] == 1]
        if len(successful_products) > 0:
            sample = successful_products.sample(1).iloc[0]
        else:
            sample = df_products.sample(1).iloc[0]
    else:
        sample = df_products.sample(1).iloc[0]
    
    # Calcular features derivadas
    has_sale = int(pd.notna(sample.get('sale_price_usd', np.nan)))
    has_value_price = int(pd.notna(sample.get('value_price_usd', np.nan)))
    
    discount_pct = 0
    if has_sale and sample['price_usd'] > 0:
        sale_price = sample.get('sale_price_usd', sample['price_usd'])
        discount_pct = ((sample['price_usd'] - sale_price) / sample['price_usd']) * 100
    
    # Variaciones
    has_variations = int(sample.get('child_count', 0) > 0)
    variation_price_range = 0
    if pd.notna(sample.get('child_max_price')) and pd.notna(sample.get('child_min_price')):
        variation_price_range = sample['child_max_price'] - sample['child_min_price']
    
    # Ingredientes y highlights
    has_ingredients = int(pd.notna(sample.get('ingredients')))
    has_highlights = int(pd.notna(sample.get('highlights')))
    num_highlights = 0
    if has_highlights:
        num_highlights = len(str(sample['highlights']).split(','))
    
    # Exclusividad
    exclusivity_score = (
        sample.get('sephora_exclusive', 0) * 3 +
        sample.get('limited_edition', 0) * 2 +
        sample.get('online_only', 0) * 1
    )
    is_exclusive = int((sample.get('sephora_exclusive', 0) == 1) or (sample.get('limited_edition', 0) == 1))
    
    # Categoría
    category_depth = (
        int(pd.notna(sample.get('primary_category'))) +
        int(pd.notna(sample.get('secondary_category'))) +
        int(pd.notna(sample.get('tertiary_category')))
    )
    
    # Brand features (calcular del dataset)
    brand_name = sample.get('brand_name', 'Unknown')
    brand_products = df_products[df_products['brand_name'] == brand_name]
    brand_product_count = len(brand_products)
    brand_avg_rating = brand_products['rating'].mean()
    brand_avg_loves = brand_products['loves_count'].mean()
    is_top_brand = int(brand_product_count >= df_products['brand_name'].value_counts().head(20).min())
    
    return {
        'price_usd': float(sample.get('price_usd', 0)),
        'rating': float(sample.get('rating', 0)),
        'child_count': int(sample.get('child_count', 0)),
        'limited_edition': int(sample.get('limited_edition', 0)),
        'new': int(sample.get('new', 0)),
        'online_only': int(sample.get('online_only', 0)),
        'out_of_stock': int(sample.get('out_of_stock', 0)),
        'sephora_exclusive': int(sample.get('sephora_exclusive', 0)),
        'has_sale': has_sale,
        'has_value_price': has_value_price,
        'discount_pct': float(discount_pct),
        'brand_product_count': int(brand_product_count),
        'brand_avg_rating': float(brand_avg_rating),
        'brand_avg_loves': float(brand_avg_loves),
        'is_top_brand': is_top_brand,
        'has_variations': has_variations,
        'variation_price_range': float(variation_price_range),
        'has_ingredients': has_ingredients,
        'has_highlights': has_highlights,
        'num_highlights': int(num_highlights),
        'exclusivity_score': int(exclusivity_score),
        'is_exclusive': is_exclusive,
        'category_depth': int(category_depth)
    }

with st.sidebar:
    st.markdown("#### Modelos")
    
    # Métricas de los modelos
    xgb_tuned_metrics = tuned_metrics[tuned_metrics['Model'] == 'XGBoost (Tuned)'].iloc[0]
    lgb_tuned_metrics = tuned_metrics[tuned_metrics['Model'] == 'LightGBM (Tuned)'].iloc[0]
    rf_tuned_metrics = tuned_metrics[tuned_metrics['Model'] == 'Random Forest (Tuned)'].iloc[0]
    xgb_base_metrics = baseline_metrics.loc['XGBoost']
    
    # Mostrar métricas de forma simple con separación
    st.markdown(f"• **XGBoost Optimizado**")
    st.text(f"F1: {xgb_tuned_metrics['F1-Score']*100:.1f}% | Acc: {xgb_tuned_metrics['Accuracy']*100:.1f}%")
    st.markdown("")  # Separación
    
    st.markdown(f"• **LightGBM Optimizado**")
    st.text(f"F1: {lgb_tuned_metrics['F1-Score']*100:.1f}% | Acc: {lgb_tuned_metrics['Accuracy']*100:.1f}%")
    st.markdown("")  # Separación
    
    st.markdown(f"• **Random Forest Optimizado**")
    st.text(f"F1: {rf_tuned_metrics['F1-Score']*100:.1f}% | Acc: {rf_tuned_metrics['Accuracy']*100:.1f}%")

st.markdown('<h1 class="main-title">Sephora Product Success Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predicción de éxito comercial de productos de skincare usando Machine Learning</p>', unsafe_allow_html=True)

# Tabs de navegación
tab1, tab2, tab3 = st.tabs(["Predicción", "Comparación de Modelos", "Documentación"])

with tab1:
    # Botón para generar producto aleatorio
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("Generar Producto Aleatorio", use_container_width=True, type="primary"):
            st.session_state['random_product'] = generate_random_product()
    
    # Inicializar con producto aleatorio si no existe
    if 'random_product' not in st.session_state:
        st.session_state['random_product'] = generate_random_product()
    
    st.markdown("---")
    
    # Formulario de entrada en 3 columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Producto")
        price_usd = st.number_input("Precio USD", min_value=0.0, value=st.session_state['random_product']['price_usd'], step=5.0)
        rating = st.slider("Rating Inicial", 0.0, 5.0, st.session_state['random_product']['rating'], 0.1)
        discount_pct = st.number_input("Descuento %", min_value=0.0, value=st.session_state['random_product']['discount_pct'], step=5.0)
        child_count = st.number_input("N° Variaciones", min_value=0, value=st.session_state['random_product']['child_count'], step=1)
        variation_price_range = st.number_input("Rango Precio", min_value=0.0, value=st.session_state['random_product']['variation_price_range'], step=5.0)
        num_highlights = st.number_input("N° Destacados", min_value=0, value=st.session_state['random_product']['num_highlights'], step=1)
    
    with col2:
        st.markdown("### Marca")
        brand_product_count = st.number_input("Productos", min_value=0, value=st.session_state['random_product']['brand_product_count'], step=10)
        brand_avg_rating = st.slider("Rating Marca", 0.0, 5.0, st.session_state['random_product']['brand_avg_rating'], 0.1)
        brand_avg_loves = st.number_input("Loves Marca", min_value=0, value=int(st.session_state['random_product']['brand_avg_loves']), step=1000)
        
        st.markdown("### Clasificación")
        is_top_brand = st.checkbox("Top Brand", value=bool(st.session_state['random_product']['is_top_brand']))
        exclusivity_score = st.slider("Score Exclusividad", 0, 6, st.session_state['random_product']['exclusivity_score'], step=1)
        category_depth = st.slider("Nivel Categoría", 1, 3, st.session_state['random_product']['category_depth'], step=1)
    
    with col3:
        st.markdown("### Indicadores")
        limited_edition = st.checkbox("Limited Edition", value=bool(st.session_state['random_product']['limited_edition']))
        new = st.checkbox("New", value=bool(st.session_state['random_product']['new']))
        online_only = st.checkbox("Online Only", value=bool(st.session_state['random_product']['online_only']))
        out_of_stock = st.checkbox("Out of Stock", value=bool(st.session_state['random_product']['out_of_stock']))
        sephora_exclusive = st.checkbox("Sephora Exclusive", value=bool(st.session_state['random_product']['sephora_exclusive']))
        has_value_price = st.checkbox("Has Value Price", value=bool(st.session_state['random_product']['has_value_price']))
        has_ingredients = st.checkbox("Has Ingredients", value=bool(st.session_state['random_product']['has_ingredients']))
        has_highlights = st.checkbox("Has Highlights", value=bool(st.session_state['random_product']['has_highlights']))
        is_exclusive = st.checkbox("Is Exclusive", value=bool(st.session_state['random_product']['is_exclusive']))
    
    st.markdown("---")
    st.markdown("## Resultados de la Predicción")
    
    # Crear features y predecir (inferir has_sale y has_variations automáticamente)
    has_sale = 1 if discount_pct > 0 else 0
    has_variations = 1 if child_count > 0 else 0
    
    features = create_feature_dict(
        price_usd, rating, child_count,
        int(limited_edition), int(new), int(online_only), int(out_of_stock), int(sephora_exclusive),
        has_sale, int(has_value_price), discount_pct,
        brand_product_count, brand_avg_rating, brand_avg_loves, int(is_top_brand),
        has_variations, variation_price_range, int(has_ingredients), int(has_highlights), num_highlights,
        exclusivity_score, int(is_exclusive), category_depth
    )
    
    predictions = predict_product(features)
    
    # Mostrar predicciones en 3 columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_xgb = predictions['xgb_tuned']
        prob_xgb = pred_xgb['proba'][1] * 100
        border_color = "#27ae60" if pred_xgb['pred'] == 1 else "#c0392b"
        
        st.markdown(f"""
        <h3 style="border-bottom: 3px solid {border_color}; padding-bottom: 0.5rem; margin-bottom: 1rem; font-size: 1.1rem;">XGBoost Tuned</h3>
        """, unsafe_allow_html=True)
        
        # Reducir ancho con columnas internas
        col_inner_left, col_inner_center, col_inner_right = st.columns([0.15, 1, 0.15])
        with col_inner_center:
            if pred_xgb['pred'] == 1:
                st.markdown(f"""
                <div class="prediction-success">
                    <h3 style="margin: 0.1rem 0; color: #27ae60; font-size: 1.05rem;">✅ EXITOSO</h3>
                    <h2 style="margin: 0.1rem 0; color: #2c3e50; font-size: 1.6rem;">{prob_xgb:.1f}%</h2>
                    <p style="text-align: left; margin: 0.3rem 0 0.15rem 0; font-size: 0.75rem; color: #7f8c8d;">Confianza:</p>
                    <div style="background: #ecf0f1; border-radius: 10px; height: 5px;">
                        <div style="background: #27ae60; height: 100%; width: {prob_xgb}%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-fail">
                    <h3 style="margin: 0.1rem 0; color: #c0392b; font-size: 1.05rem;">❌ NO EXITOSO</h3>
                    <h2 style="margin: 0.1rem 0; color: #2c3e50; font-size: 1.6rem;">{100-prob_xgb:.1f}%</h2>
                    <p style="text-align: left; margin: 0.3rem 0 0.15rem 0; font-size: 0.75rem; color: #7f8c8d;">Confianza:</p>
                    <div style="background: #ecf0f1; border-radius: 10px; height: 5px;">
                        <div style="background: #c0392b; height: 100%; width: {100-prob_xgb}%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        pred_lgb = predictions['lgb_tuned']
        prob_lgb = pred_lgb['proba'][1] * 100
        border_color = "#27ae60" if pred_lgb['pred'] == 1 else "#c0392b"
        
        st.markdown(f"""
        <h3 style="border-bottom: 3px solid {border_color}; padding-bottom: 0.5rem; margin-bottom: 1rem; font-size: 1.1rem;">LightGBM Tuned</h3>
        """, unsafe_allow_html=True)
        
        # Reducir ancho con columnas internas
        col_inner_left, col_inner_center, col_inner_right = st.columns([0.15, 1, 0.15])
        with col_inner_center:
            if pred_lgb['pred'] == 1:
                st.markdown(f"""
                <div class="prediction-success">
                    <h3 style="margin: 0.1rem 0; color: #27ae60; font-size: 1.05rem;">EXITOSO</h3>
                    <h2 style="margin: 0.1rem 0; color: #2c3e50; font-size: 1.6rem;">{prob_lgb:.1f}%</h2>
                    <p style="text-align: left; margin: 0.3rem 0 0.15rem 0; font-size: 0.75rem; color: #7f8c8d;">Confianza:</p>
                    <div style="background: #ecf0f1; border-radius: 10px; height: 5px;">
                        <div style="background: #27ae60; height: 100%; width: {prob_lgb}%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-fail">
                    <h3 style="margin: 0.1rem 0; color: #c0392b; font-size: 1.05rem;">NO EXITOSO</h3>
                    <h2 style="margin: 0.1rem 0; color: #2c3e50; font-size: 1.6rem;">{100-prob_lgb:.1f}%</h2>
                    <p style="text-align: left; margin: 0.3rem 0 0.15rem 0; font-size: 0.75rem; color: #7f8c8d;">Confianza:</p>
                    <div style="background: #ecf0f1; border-radius: 10px; height: 5px;">
                        <div style="background: #c0392b; height: 100%; width: {100-prob_lgb}%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        pred_rf = predictions['rf_tuned']
        prob_rf = pred_rf['proba'][1] * 100
        border_color = "#27ae60" if pred_rf['pred'] == 1 else "#c0392b"
        
        st.markdown(f"""
        <h3 style="border-bottom: 3px solid {border_color}; padding-bottom: 0.5rem; margin-bottom: 1rem; font-size: 1.1rem;">Random Forest Tuned</h3>
        """, unsafe_allow_html=True)
        
        # Reducir ancho con columnas internas
        col_inner_left, col_inner_center, col_inner_right = st.columns([0.15, 1, 0.15])
        with col_inner_center:
            if pred_rf['pred'] == 1:
                st.markdown(f"""
                <div class="prediction-success">
                    <h3 style="margin: 0.1rem 0; color: #27ae60; font-size: 1.05rem;">EXITOSO</h3>
                    <h2 style="margin: 0.1rem 0; color: #2c3e50; font-size: 1.6rem;">{prob_rf:.1f}%</h2>
                    <p style="text-align: left; margin: 0.3rem 0 0.15rem 0; font-size: 0.75rem; color: #7f8c8d;">Confianza:</p>
                    <div style="background: #ecf0f1; border-radius: 10px; height: 5px;">
                        <div style="background: #27ae60; height: 100%; width: {prob_rf}%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-fail">
                    <h3 style="margin: 0.1rem 0; color: #c0392b; font-size: 1.05rem;">NO EXITOSO</h3>
                    <h2 style="margin: 0.1rem 0; color: #2c3e50; font-size: 1.6rem;">{100-prob_rf:.1f}%</h2>
                    <p style="text-align: left; margin: 0.3rem 0 0.15rem 0; font-size: 0.75rem; color: #7f8c8d;">Confianza:</p>
                    <div style="background: #ecf0f1; border-radius: 10px; height: 5px;">
                        <div style="background: #c0392b; height: 100%; width: {100-prob_rf}%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Análisis de Confianza Detallado
    st.markdown("### Análisis de Confianza")
    
    # Crear DataFrame para mejor visualización
    confidence_data = pd.DataFrame({
        'Modelo': ['XGBoost Tuned', 'LightGBM Tuned', 'Random Forest Tuned'],
        'Confianza Éxito (%)': [prob_xgb, prob_lgb, prob_rf],
        'Confianza Fracaso (%)': [100-prob_xgb, 100-prob_lgb, 100-prob_rf],
        'Predicción': ['Éxito' if p['pred'] == 1 else 'No Éxito' 
                       for p in [pred_xgb, pred_lgb, pred_rf]]
    })
    
    # Gráfico de barras agrupadas con ancho reducido
    col_graph_left, col_graph_center, col_graph_right = st.columns([1, 2, 1])
    with col_graph_center:
        fig = go.Figure()
    
    # Barra de confianza de éxito
    fig.add_trace(go.Bar(
        name='Confianza Éxito',
        x=confidence_data['Modelo'],
        y=confidence_data['Confianza Éxito (%)'],
        text=[f"{v:.1f}%" for v in confidence_data['Confianza Éxito (%)']],
        textposition='auto',
        textfont=dict(color='white', size=14, family='Arial Black'),
        marker_color='#27ae60',
        hovertemplate='%{x}<br>Éxito: %{y:.1f}%<extra></extra>'
    ))
    
    # Barra de confianza de fracaso
    fig.add_trace(go.Bar(
        name='Confianza Fracaso',
        x=confidence_data['Modelo'],
        y=confidence_data['Confianza Fracaso (%)'],
        text=[f"{v:.1f}%" for v in confidence_data['Confianza Fracaso (%)']],
        textposition='auto',
        textfont=dict(color='white', size=14, family='Arial Black'),
        marker_color='#c0392b',
        hovertemplate='%{x}<br>Fracaso: %{y:.1f}%<extra></extra>'
    ))
    
    # Línea de umbral
    fig.add_hline(y=50, line_dash="dash", line_color="#34495e", line_width=2,
                  annotation_text="Umbral: 50%", 
                  annotation_position="left",
                  annotation_font_size=12,
                  annotation_font_color="#34495e")
    
    fig.update_layout(
        yaxis_title="Probabilidad (%)",
        yaxis_range=[0, 105],
        height=350,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=12),
        yaxis=dict(gridcolor='#e0e0e0')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla resumen centrada con separación y márgenes
    st.markdown("<div style='margin-top: 2rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    col_space1, col_table, col_space_mid, col_metrics, col_space2 = st.columns([0.3, 2, 0.3, 1.2, 0.3])
    with col_table:
        st.dataframe(
            confidence_data.style.format({
                'Confianza Éxito (%)': '{:.1f}',
                'Confianza Fracaso (%)': '{:.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    with col_metrics:
        st.metric("Confianza Promedio (Éxito)", f"{confidence_data['Confianza Éxito (%)'].mean():.1f}%")
        st.metric("Desviación Estándar", f"{confidence_data['Confianza Éxito (%)'].std():.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Consenso de modelos
    st.markdown("### Consenso de Modelos")
    
    success_count = sum([pred_xgb['pred'], pred_lgb['pred'], pred_rf['pred']])
    avg_prob = (prob_xgb + prob_lgb + prob_rf) / 3
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modelos que predicen ÉXITO", f"{success_count}/3")
    
    with col2:
        st.metric("Probabilidad Promedio", f"{avg_prob:.1f}%")
    
    with col3:
        consensus_level = "FUERTE" if success_count == 3 else "MODERADO" if success_count == 2 else "DÉBIL"
        st.metric("Nivel de Consenso", consensus_level)
    
    # Recomendación
    if success_count >= 2:
        st.success("RECOMENDACIÓN: Producto con alto potencial de éxito")
    else:
        st.warning("PRECAUCIÓN: Producto con bajo potencial de éxito")

with tab2:
    st.markdown("## Comparación de Modelos")
    
    # Tabla de métricas
    st.markdown("### Tabla de Métricas")
    
    # Cargar métricas baseline adicionales
    lgb_base_metrics = baseline_metrics.loc['LightGBM']
    rf_base_metrics = baseline_metrics.loc['Random Forest']
    
    # Crear tabla comparativa
    comparison_df = pd.DataFrame({
        'Modelo': ['XGBoost Tuned', 'LightGBM Tuned', 'Random Forest Tuned', 
                   'XGBoost Baseline', 'LightGBM Baseline', 'Random Forest Baseline'],
        'Accuracy': [
            xgb_tuned_metrics['Accuracy'],
            lgb_tuned_metrics['Accuracy'],
            rf_tuned_metrics['Accuracy'],
            xgb_base_metrics['test_accuracy'],
            lgb_base_metrics['test_accuracy'],
            rf_base_metrics['test_accuracy']
        ],
        'Precision': [
            xgb_tuned_metrics['Precision'],
            lgb_tuned_metrics['Precision'],
            rf_tuned_metrics['Precision'],
            xgb_base_metrics['precision'],
            lgb_base_metrics['precision'],
            rf_base_metrics['precision']
        ],
        'Recall': [
            xgb_tuned_metrics['Recall'],
            lgb_tuned_metrics['Recall'],
            rf_tuned_metrics['Recall'],
            xgb_base_metrics['recall'],
            lgb_base_metrics['recall'],
            rf_base_metrics['recall']
        ],
        'F1-Score': [
            xgb_tuned_metrics['F1-Score'],
            lgb_tuned_metrics['F1-Score'],
            rf_tuned_metrics['F1-Score'],
            xgb_base_metrics['f1'],
            lgb_base_metrics['f1'],
            rf_base_metrics['f1']
        ],
        'ROC-AUC': [
            xgb_tuned_metrics['ROC-AUC'],
            lgb_tuned_metrics['ROC-AUC'],
            rf_tuned_metrics['ROC-AUC'],
            xgb_base_metrics['roc_auc'],
            lgb_base_metrics['roc_auc'],
            rf_base_metrics['roc_auc']
        ]
    })
    
    # Formatear tabla con highlight de máximo (verde) y mínimo (rojo)
    def highlight_max_min(s):
        if s.name == 'Modelo':
            return [''] * len(s)
        is_max = s == s.max()
        is_min = s == s.min()
        colors = []
        for i in range(len(s)):
            if is_max[i]:
                colors.append('background-color: #a8d5a8; color: #000000')  # Verde más notorio con texto negro
            elif is_min[i]:
                colors.append('background-color: #f5b7b1; color: #000000')  # Rojo más notorio con texto negro
            else:
                colors.append('')
        return colors
    
    st.dataframe(
        comparison_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}'
        }).apply(highlight_max_min),
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("### Mejoras con Hyperparameter Tuning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### XGBoost: Baseline → Tuned")
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        baseline_vals = [xgb_base_metrics['test_accuracy'], xgb_base_metrics['precision'], 
                        xgb_base_metrics['recall'], xgb_base_metrics['f1']]
        tuned_vals = [xgb_tuned_metrics['Accuracy'], xgb_tuned_metrics['Precision'],
                     xgb_tuned_metrics['Recall'], xgb_tuned_metrics['F1-Score']]
        
        for metric, base, tuned in zip(metrics_names, baseline_vals, tuned_vals):
            improvement = ((tuned - base) / base) * 100
            st.metric(metric, f"{tuned:.4f}", delta=f"{improvement:+.2f}%")
    
    with col2:
        st.markdown("#### LightGBM: Baseline → Tuned")
        
        baseline_vals_lgb = [lgb_base_metrics['test_accuracy'], lgb_base_metrics['precision'], 
                            lgb_base_metrics['recall'], lgb_base_metrics['f1']]
        tuned_vals_lgb = [lgb_tuned_metrics['Accuracy'], lgb_tuned_metrics['Precision'],
                         lgb_tuned_metrics['Recall'], lgb_tuned_metrics['F1-Score']]
        
        for metric, base, tuned in zip(metrics_names, baseline_vals_lgb, tuned_vals_lgb):
            improvement = ((tuned - base) / base) * 100
            st.metric(metric, f"{tuned:.4f}", delta=f"{improvement:+.2f}%")
    
    with col3:
        st.markdown("#### Random Forest: Baseline → Tuned")
        
        baseline_vals_rf = [rf_base_metrics['test_accuracy'], rf_base_metrics['precision'], 
                           rf_base_metrics['recall'], rf_base_metrics['f1']]
        tuned_vals_rf = [rf_tuned_metrics['Accuracy'], rf_tuned_metrics['Precision'],
                        rf_tuned_metrics['Recall'], rf_tuned_metrics['F1-Score']]
        
        for metric, base, tuned in zip(metrics_names, baseline_vals_rf, tuned_vals_rf):
            improvement = ((tuned - base) / base) * 100
            st.metric(metric, f"{tuned:.4f}", delta=f"{improvement:+.2f}%")

with tab3:
    st.markdown("## Documentación Técnica")
    st.markdown("")
    
    # 1. Objetivo del Modelo
    st.markdown("""
    <div class="doc-section">
        <h2>Objetivo del Modelo</h2>
        <p>Este sistema utiliza algoritmos de Machine Learning para predecir el éxito comercial de productos de Sephora basándose en características tempranas conocidas al lanzamiento, para tomar decisiones de mercado</p>
        <p><b>Definición de Éxito:</b></p>
        <ul>
            <li><b>Loves Count</b> superior al percentil 75 <b>Y</b> Rating mayor o igual a 4.0</li>
            <li>Productos con alta popularidad y buena calificación de usuarios</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Dataset
    st.markdown("""
    <div class="doc-section">
        <h2>Dataset</h2>
        <div class="doc-code">
Fuente:             Sephora Products and Skincare Reviews (Kaggle)
Total Productos:    8,494 productos
Total Reviews:      602,130 reviews
Productos Exitosos: 1,669 (19.6% del total)
Desbalance:         Clase minoritaria tratada con SMOTE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Modelos
    st.markdown("""
    <div class="doc-section">
        <h2>Arquitectura de Modelos</h2>
        <p><b>Modelos Implementados:</b></p>
        <ol>
            <li><b>XGBoost Tuned</b>
                <ul>
                    <li>Optimización: Optuna con 100 trials (TPE Sampler)</li>
                    <li>Uso: Modelo principal en producción</li>
                </ul>
            </li>
            <li><b>LightGBM Tuned</b>
                <ul>
                    <li>Optimización: Optuna con 100 trials (TPE Sampler)</li>
                    <li>Uso: Modelo secundario en producción</li>
                </ul>
            </li>
            <li><b>Random Forest Tuned</b>
                <ul>
                    <li>Optimización: Optuna con 100 trials (TPE Sampler)</li>
                    <li>Uso: Modelo ensemble complementario</li>
                </ul>
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Features
    st.markdown("""
    <div class="doc-section">
        <h2>Ingeniería de Características</h2>
        <p><b>Total: 27 features organizadas en 7 categorías</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="doc-section">
            <p style="font-weight: 600; color: #ffffff; margin-bottom: 0.3rem;">Características Básicas</p>
            <ul style="margin-top: 0.2rem; margin-bottom: 1rem;">
                <li>price_usd</li>
                <li>rating</li>
                <li>child_count</li>
            </ul>
            <p style="font-weight: 600; color: #ffffff; margin-bottom: 0.3rem;">Indicadores Binarios</p>
            <ul style="margin-top: 0.2rem; margin-bottom: 1rem;">
                <li>limited_edition</li>
                <li>new</li>
                <li>online_only</li>
                <li>out_of_stock</li>
                <li>sephora_exclusive</li>
            </ul>
            <p style="font-weight: 600; color: #ffffff; margin-bottom: 0.3rem;">Precio y Descuentos</p>
            <ul style="margin-top: 0.2rem;">
                <li>has_sale</li>
                <li>has_value_price</li>
                <li>discount_pct</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="doc-section">
            <p style="font-weight: 600; color: #ffffff; margin-bottom: 0.3rem;">Características de Marca</p>
            <ul style="margin-top: 0.2rem; margin-bottom: 1rem;">
                <li>brand_product_count</li>
                <li>brand_avg_rating</li>
                <li>brand_avg_loves</li>
                <li>is_top_brand</li>
            </ul>
            <p style="font-weight: 600; color: #ffffff; margin-bottom: 0.3rem;">Características de Producto</p>
            <ul style="margin-top: 0.2rem;">
                <li>has_variations</li>
                <li>variation_price_range</li>
                <li>has_ingredients</li>
                <li>has_highlights</li>
                <li>num_highlights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="doc-section">
            <p style="font-weight: 600; color: #ffffff; margin-bottom: 0.3rem;">Exclusividad</p>
            <ul style="margin-top: 0.2rem; margin-bottom: 1rem;">
                <li>exclusivity_score</li>
                <li>is_exclusive</li>
            </ul>
            <p style="font-weight: 600; color: #ffffff; margin-bottom: 0.3rem;">Categorización</p>
            <ul style="margin-top: 0.2rem;">
                <li>category_depth</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 5. Feature Importance
    with st.container():
        st.markdown("""
        <div class="doc-section">
            <h2>Importancia de Características</h2>
            <p style="color: #b0b0b0; margin-bottom: 0.5rem;">Top 5 características según modelo XGBoost optimizado:</p>
            <ol style="color: #b0b0b0; line-height: 1.8;">
                <li><strong style="color: #ffffff;">has_variations (29.8%)</strong> - La existencia de variaciones del producto es el predictor más fuerte</li>
                <li><strong style="color: #ffffff;">brand_avg_loves (9.7%)</strong> - El prestigio de la marca medido por popularidad promedio</li>
                <li><strong style="color: #ffffff;">rating (7.7%)</strong> - La calificación inicial del producto</li>
                <li><strong style="color: #ffffff;">online_only (7.3%)</strong> - El canal de distribución exclusivo online</li>
                <li><strong style="color: #ffffff;">new (5.9%)</strong> - El factor novedad en el lanzamiento</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # 6. Métricas
    with st.container():
        st.markdown("""
        <div class="doc-section">
            <h2>Métricas de Evaluación</h2>
            <p style="color: #b0b0b0; margin-bottom: 0.8rem;"><strong style="color: #ffffff;">Accuracy:</strong> Proporción de predicciones correctas sobre el total de predicciones. Métrica general de rendimiento del modelo.</p>
            <p style="color: #b0b0b0; margin-bottom: 0.8rem;"><strong style="color: #ffffff;">Precision:</strong> De todos los productos predichos como exitosos, ¿qué proporción realmente lo son? Minimiza falsos positivos.</p>
            <p style="color: #b0b0b0; margin-bottom: 0.8rem;"><strong style="color: #ffffff;">Recall:</strong> De todos los productos realmente exitosos, ¿qué proporción detectamos? Minimiza falsos negativos.</p>
            <p style="color: #b0b0b0; margin-bottom: 0.8rem;"><strong style="color: #ffffff;">F1-Score:</strong> Media armónica entre Precision y Recall. Métrica balanceada para datasets desbalanceados.</p>
            <p style="color: #b0b0b0; margin-bottom: 0;"><strong style="color: #ffffff;">ROC-AUC:</strong> Área bajo la curva ROC. Mide la capacidad del modelo para discriminar entre clases.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 7. Casos de Uso
    with st.container():
        st.markdown("""
        <div class="doc-section">
            <h2>Aplicaciones Empresariales</h2>
            <p style="color: #b0b0b0; margin-bottom: 0.8rem;"><strong style="color: #ffffff;">Gestión de Inventario:</strong> Optimizar niveles de stock anticipando la demanda de productos basándose en predicciones de éxito.</p>
            <p style="color: #b0b0b0; margin-bottom: 0.8rem;"><strong style="color: #ffffff;">Estrategia de Marketing:</strong> Asignar presupuesto publicitario de manera eficiente priorizando productos con alto potencial de éxito.</p>
            <p style="color: #b0b0b0; margin-bottom: 0.8rem;"><strong style="color: #ffffff;">Desarrollo de Producto:</strong> Identificar características clave que contribuyen al éxito para guiar decisiones de diseño y desarrollo.</p>
            <p style="color: #b0b0b0; margin-bottom: 0;"><strong style="color: #ffffff;">Análisis de Marca:</strong> Evaluar el potencial de nuevas marcas o líneas de producto a su temprana incorporación al catálogo.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 8. Stack Tecnológico
    st.markdown("""
    <div class="doc-section">
        <h2>Stack Tecnológico</h2>
        <div class="doc-code">
Lenguaje:           Python 3.13.2
ML Frameworks:      XGBoost 2.0.3, LightGBM 4.1.0
Optimización:       Optuna 3.5.0 (Tree-structured Parzen Estimator)
Data Processing:    Pandas 2.1.4, NumPy 1.26.2
Visualization:      Matplotlib 3.8.2, Seaborn 0.13.0, Plotly 5.18.0
Frontend:           Streamlit 1.29.0
Imbalance:          imbalanced-learn 0.11.0 (SMOTE)
Environment:        venv (Python Virtual Environment)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 9. Información Académica
    st.markdown("""
    <div class="doc-section">
        <h2>Información del Proyecto</h2>
        <div class="doc-code">
Curso:              CC209 - Data Mining Tools
Institución:        Universidad
Periodo:            Ciclo 2025-2
Fecha:              Diciembre 2025
Tipo:               Trabajo Final - Proyecto de Machine Learning
Metodología:        CRISP-DM (Cross-Industry Standard Process for Data Mining)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("<p style='text-align: center; color: #95a5a6; font-size: 0.9rem; padding: 2rem 0;'>Sephora Product Success Predictor © 2025 | Desarrollado con Streamlit</p>", unsafe_allow_html=True)
