# Recomendacion-de-productos-skincare

- Objetivo del trabajo:
Desarrollar un sistema de recomendación inteligente de productos de skincare que, mediante técnicas de minería de datos y aprendizaje automático, identifique patrones de comportamiento y preferencias de los usuarios para ofrecer sugerencias personalizadas según su tipo de piel, características individuales y valores éticos. El sistema busca reducir la sobrecarga de información en plataformas con amplios catálogos, mejorar la experiencia de compra y apoyar la toma de decisiones tanto de los consumidores como de las marcas, a partir del análisis estructurado y textual de datos históricos de productos y reseñas.
-  Nombre de los alumnos participantes:
  
U202014068        Andrea Fabiana Garcia Napuri

U20181B618        Nikole Scarlet Garcia Chavez

U202122837      	Fernando Samuel Paredes Espinoza

-  Breve descripción del dataset:
  
El presente proyecto utiliza el Sephora Products and Skincare Reviews Dataset, disponible en la plataforma Kaggle. Este conjunto de datos combina información estructurada y textual, por lo que se clasifica como semiestructurado, al integrar variables numéricas, categóricas y texto libre.

De acuerdo con su descripción en Kaggle, el dataset fue recolectado mediante un scraper en Python en marzo de 2023 y contiene información sobre más de 8,000 productos del catálogo en línea de Sephora, incluyendo nombres de productos y marcas, precios, ingredientes, valoraciones y características destacadas. Además, reúne aproximadamente 1 millón de reseñas de usuarios sobre más de 2,000 productos de la categoría Skincare, con datos sobre apariencia, calificaciones y recomendaciones realizadas por otros consumidores (Nadyinky, 2023).
El dataset puede consultarse en el siguiente enlace: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews 

El contenido del Dataset consta de seis archivos Product data content y cinco de Reviews data content, los cuales se describen a continuación.

  Product Data Content
  
Este archivo contiene información estructurada de productos, con variables como:

product_name, brand_name, price_usd, rating, reviews, ingredients, highlights y categories, además de variables booleanas (limited_edition, online_only, sephora_exclusive, entre otras). Las columnas ingredients y highlights, por su contenido textual, serán empleadas para la construcción de perfiles de producto mediante técnicas de TF-IDF y similitud del coseno, lo que permitirá identificar relaciones entre productos según sus componentes o beneficios.

  Reviews Data (0–250, 250–500, etc)
  
Esta serie de archivos, que en conjunto suman cerca de 1 millón de reseñas, recopila información sobre la experiencia de los usuarios, incluyendo variables como:

rating, is_recommended, review_text, review_title, skin_type, skin_tone, eye_color, hair_color, helpfulness y submission_time. Estas reseñas, por su naturaleza no estructurada, permitirán aplicar técnicas de Minería de Textos (NLP) para extraer opiniones, temas y patrones de preferencia asociados con la satisfacción y recomendación de los productos.
