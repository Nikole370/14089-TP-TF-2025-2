# ============================================================
# INTERFAZ STREAMLIT - PRODUCTOS SKINCARE - SISTEMA CON COLUMNAS
# ============================================================

import streamlit as st
import xgboost as xgb
import numpy as np
import joblib
import os

# ============================================================
# 1. CARGA DE MODELO Y CODIFICADORES
# ============================================================

if not os.path.exists("models1"):
    st.error("No se encontró la carpeta 'models'. Verifica la ruta.")
else:
    le_brand1 = joblib.load("models1/brand_encoder_V4.pkl")
    le_cat1 = joblib.load("models1/category_encoder_v4.pkl")
    features1 = joblib.load("models1/features_seleccionadas_V4.pkl")

    model1 = xgb.Booster()
    model1.load_model("models1/xgb_model_V4.json")

# ============================================================
# 2. FUNCIÓN DE PREDICCIÓN
# ============================================================

def predict_model1(product_name, brand, category, ingredients, highlights,
                   loves_count=0, reviews=0,
                   limited_edition=False, new=False, online_only=False,
                   out_of_stock=False, sephora_exclusive=False):

    brand_encoded = le_brand1.transform([brand])[0]
    category_encoded = le_cat1.transform([category])[0]

    num_ingredients = len(ingredients.split(",")) if ingredients else 0
    num_highlights = len(highlights.split(",")) if highlights else 0
    len_ingredients = len(ingredients) if ingredients else 0
    len_highlights = len(highlights) if highlights else 0
    ing_high_interaction = num_ingredients * num_highlights
    popularity = loves_count + reviews
    is_limited = int(limited_edition)
    is_new = int(new)
    is_online_only = int(online_only)
    is_exclusive = int(sephora_exclusive)
    is_out_of_stock = int(out_of_stock)

    X_input = np.zeros((1, len(features1)))
    for i, f in enumerate(features1):
        if f == "brand_encoded":
            X_input[0, i] = brand_encoded
        elif f == "category_encoded":
            X_input[0, i] = category_encoded
        elif f == "num_ingredients":
            X_input[0, i] = num_ingredients
        elif f == "num_highlights":
            X_input[0, i] = num_highlights
        elif f == "len_ingredients":
            X_input[0, i] = len_ingredients
        elif f == "len_highlights":
            X_input[0, i] = len_highlights
        elif f == "ing_high_interaction":
            X_input[0, i] = ing_high_interaction
        elif f == "popularity":
            X_input[0, i] = popularity
        elif f == "is_limited":
            X_input[0, i] = is_limited
        elif f == "is_new":
            X_input[0, i] = is_new
        elif f == "is_online_only":
            X_input[0, i] = is_online_only
        elif f == "is_exclusive":
            X_input[0, i] = is_exclusive
        elif f == "is_out_of_stock":
            X_input[0, i] = is_out_of_stock
        else:
            X_input[0, i] = 0

    dmatrix_input = xgb.DMatrix(X_input, feature_names=features1)
    pred = model1.predict(dmatrix_input)[0]
    return round(pred, 2), popularity

# ============================================================
# 3. INTERFAZ STREAMLIT CON COLUMNAS
# ============================================================

st.title("Sephora Product Prices Predictor")
st.markdown("Predice precio aproximado de productos de skincare usando XGBoost.")

# Crear dos columnas: izquierda para inputs, derecha para predicción
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Entradas del Modelo")
    
    # Selector de ejemplos predefinidos
    ejemplo = st.selectbox(
        "Selecciona un ejemplo o ingresa manualmente:",
        ["Manual", "BLEU DE CHANEL", "Skin Foundation SPF15", "Woodphoria Eau de Parfum"]
    )
    
    # Diccionario con ejemplos predefinidos
    ejemplos = {
        "BLEU DE CHANEL": {
            "name": "BLEU DE CHANEL Parfum Twist & Spray",
            "brand": "CHANEL",
            "category": "Fragrance",
            "ingredients": "Alcohol, Parfum (Fragrance), Aqua (Water), Limonene, Linalool, Citronellol, Alpha-Isomethyl Ionone, Coumarin, Citral, Geraniol, Benzyl Benzoate, Farnesol, Benzyl Alcohol, Butyl Methoxydibenzoylmethane, Ci 60730 (Ext. Violet 2), Ci 15985 (Yellow 6), Ci 19140 (Yellow 5), Il17-1",
            "highlights": ""
        },
        "Skin Foundation SPF15": {
            "name": "Skin Foundation SPF15",
            "brand": "Bobbi Brown",
            "category": "Makeup",
            "ingredients": "Octinoxate 7.50%, Titanium Dioxide 1.00%",
            "highlights": "Light Coverage, Natural Finish, Liquid Formula, Hydrating, Oil Free, Best for Oily, Combo, Normal Skin"
        },
        "Woodphoria Eau de Parfum": {
            "name": "Woodphoria Eau de Parfum",
            "brand": "Boy Smells",
            "category": "Fragrance",
            "ingredients": "Alcohol Denat., Parfum (Fragrance), Aqua (Water/Eau), Benzyl Salicylate, Cinnamal, Cinnamyl Alcohol, Limonene.",
            "highlights": "Vegan, Unisex/ Genderless Scent, Without Phthalates, Without Parabens, Cruelty-Free"
        }
    }
    
    # Obtener valores según selección
    if ejemplo != "Manual":
        default_values = ejemplos[ejemplo]
        product_name = st.text_input("Nombre del Producto", value=default_values["name"])
        brand = st.text_input("Marca", value=default_values["brand"])
        category = st.text_input("Categoría", value=default_values["category"])
        ingredients = st.text_input("Ingredientes (coma separada)", value=default_values["ingredients"])
        highlights = st.text_input("Highlights (coma separada)", value=default_values["highlights"])
    else:
        product_name = st.text_input("Nombre del Producto", "Ej: Fragrance Discovery Set")
        brand = st.text_input("Marca", "Ej: Acqua di Parma")
        category = st.text_input("Categoría", "Ej: Bath & Body")
        ingredients = st.text_input("Ingredientes (coma separada)", "Ej: Coumarin, Eugenol, Citral")
        highlights = st.text_input("Highlights (coma separada)", "Ej: Unisex/ Genderless Scent, Fresh Scent")
    loves_count = st.number_input("Loves Count (opcional)", value=0, step=1)
    reviews = st.number_input("Reviews (opcional)", value=0, step=1)
    limited_edition = st.checkbox("Limited Edition")
    new = st.checkbox("New")
    online_only = st.checkbox("Online Only")
    out_of_stock = st.checkbox("Out of Stock")
    sephora_exclusive = st.checkbox("Sephora Exclusive")
    submit = st.button("💰 Predecir Precio")

with col2:
    if submit:
        resultado, popularity = predict_model1(
            product_name, brand, category, ingredients, highlights,
            loves_count, reviews,
            limited_edition, new, online_only,
            out_of_stock, sephora_exclusive
        )

        # Colores según precio
        if resultado <= 50:
            color = "green"
            label = "Bajo"
        elif resultado <= 120:
            color = "orange"
            label = "Medio"
        else:
            color = "red"
            label = "Alto"

        # Tarjeta visual con predicción y barra de popularidad
        st.markdown(
            f"""
            <div style='border:2px solid {color}; border-radius:10px; padding:15px; margin:10px 0;'>
                <h3 style='margin:0'>{product_name}</h3>
                <p style='margin:2px 0'><b>Marca:</b> {brand}</p>
                <p style='margin:2px 0'><b>Categoría:</b> {category}</p>
                <p style='margin:2px 0'><b>Ingredientes:</b> {ingredients}</p>
                <p style='margin:2px 0'><b>Highlights:</b> {highlights}</p>
                <h2 style='color:{color}; text-align:right;'>${resultado:.2f} ({label})</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
