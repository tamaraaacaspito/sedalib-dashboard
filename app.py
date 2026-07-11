import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SEDALIB — Detección de Fugas", page_icon="💧", layout="wide")

# --- Cargar artefactos ---
@st.cache_resource
def cargar_artefactos():
    return {
        "modelos": joblib.load("modelos_cluster.pkl"),
        "escalador": joblib.load("escalador.pkl"),
        "columnas_modelo": joblib.load("columnas_modelo.pkl"),
        "cols_num": joblib.load("cols_num.pkl"),
        "zonas": joblib.load("zonas_lista.pkl"),
        "materiales": joblib.load("materiales_lista.pkl"),
        "importancias": joblib.load("importancias.pkl"),
        "ranking_zonas": joblib.load("ranking_zonas.pkl"),
        "tiempos": joblib.load("tiempos.pkl"),
        "cm": joblib.load("matriz_confusion.pkl"),
        "metricas": joblib.load("metricas.pkl"),
        "X_test": joblib.load("X_test.pkl"),
        "y_test": joblib.load("y_test.pkl"),
    }

art = cargar_artefactos()
modelos_locales = art["modelos"]
escalador = art["escalador"]
columnas_modelo = art["columnas_modelo"]
cols_num = art["cols_num"]
zonas_lista = art["zonas"]
materiales_lista = art["materiales"]
importancias = art["importancias"]
ranking_zonas = art["ranking_zonas"]
t_sec, t_par, t_cluster = art["tiempos"]["t_sec"], art["tiempos"]["t_par"], art["tiempos"]["t_cluster"]
NUCLEOS, N_NODOS = art["tiempos"]["NUCLEOS"], art["tiempos"]["N_NODOS"]
cm = art["cm"]
metricas = art["metricas"]
X_test, y_test = art["X_test"], art["y_test"]

def predecir_cluster(modelos, X):
    probas = np.mean([m.predict_proba(X)[:, 1] for m in modelos.values()], axis=0)
    return (probas >= 0.5).astype(int), probas

def f1_manual(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

st.title("💧 Dashboard — Detección de Fugas SEDALIB S.A.")
st.caption("Proyecto Final — Cómputo Distribuido y Paralelo — UPAO")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Insights", "⚡ Rendimiento", "🛡️ Tolerancia a fallos", "🖥️ Monitoreo", "🚨 Simulador"]
)

# --- Tab 1: Insights ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        importancias.iloc[::-1].plot.barh(ax=ax, color='#2b7bba')
        ax.set_title('Top 10 variables — importancia'); ax.set_xlabel('importancia')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        (ranking_zonas * 100).plot.barh(ax=ax, color='#d62728')
        ax.set_xlabel('tasa de fuga predicha (%)'); ax.set_title('Ranking de zonas')
        st.pyplot(fig)

# --- Tab 2: Rendimiento ---
with tab2:
    fig, ax = plt.subplots(figsize=(7, 4))
    config = ['Secuencial\n(1 núcleo)', f'Paralelo\n({NUCLEOS} núcleos)', f'Distribuido\n({N_NODOS} nodos)']
    valores = [t_sec, t_par, t_cluster]
    barras = ax.bar(config, valores, color=['#999999', '#2b7bba', '#1D9E75'])
    for barra, t in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width()/2, t, f'{t:.2f}s', ha='center', va='bottom')
    ax.set_ylabel('tiempo (s)'); ax.set_title('Secuencial vs. paralelo vs. distribuido')
    st.pyplot(fig)
    st.markdown(f"**Speedup paralelo:** {t_sec/t_par:.2f}x &nbsp;|&nbsp; **Speedup distribuido:** {t_sec/t_cluster:.2f}x")

# --- Tab 3: Tolerancia a fallos ---
with tab3:
    st.write("Simula la caída de un nodo y observa cómo el clúster sigue prediciendo con los nodos restantes.")
    nodo_a_caer = st.selectbox("Nodo a simular caída", list(range(N_NODOS)), index=2)
    if st.button("Simular caída y recuperación"):
        modelos_activos = {k: v for k, v in modelos_locales.items() if k != nodo_a_caer}
        pred_f, _ = predecir_cluster(modelos_activos, X_test)
        f1_con_falla = f1_manual(y_test.values, pred_f)
        pred_r, _ = predecir_cluster(modelos_locales, X_test)
        f1_recuperado = f1_manual(y_test.values, pred_r)

        estado_cols = st.columns(N_NODOS)
        for i in range(N_NODOS):
            with estado_cols[i]:
                st.metric(f"Nodo {i}", "🔴 caído" if i == nodo_a_caer else "🟢 activo")

        st.markdown(f"**F1 con nodo {nodo_a_caer} caído** ({len(modelos_activos)} nodos activos): `{f1_con_falla:.4f}`")
        st.markdown(f"**F1 tras recuperar el checkpoint del nodo {nodo_a_caer}:** `{f1_recuperado:.4f}`")

# --- Tab 4: Monitoreo ---
with tab4:
    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Fuga']); ax.set_yticklabels(['Normal', 'Fuga'])
        ax.set_xlabel('Predicción'); ax.set_ylabel('Valor real'); ax.set_title('Matriz de confusión')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                         color='white' if cm[i,j] > cm.max()/2 else 'black')
        st.pyplot(fig)
    with col2:
        st.metric("Accuracy", f"{metricas['acc']:.4f}")
        st.metric("Precision", f"{metricas['prec']:.4f}")
        st.metric("Recall", f"{metricas['rec']:.4f}")
        st.metric("F1-score", f"{metricas['f1']:.4f}")
        st.metric("AUC-ROC", f"{metricas['auc']:.4f}")

# --- Tab 5: Simulador ---
with tab5:
    st.write("Introduce una lectura de sensor para clasificarla con el modelo distribuido.")
    col1, col2 = st.columns(2)
    with col1:
        zona = st.selectbox("Zona", zonas_lista)
        material = st.selectbox("Material de tubería", materiales_lista)
        edad = st.slider("Antigüedad tubería (años)", 0, 60, 20)
        diametro = st.slider("Diámetro (mm)", 50, 300, 110)
        hora_dia = st.slider("Hora del día", 0, 23, 3)
    with col2:
        presion = st.slider("Presión (m.c.a.)", 0.0, 50.0, 25.0)
        caudal = st.slider("Caudal (L/s)", 0.0, 60.0, 20.0)
        caudal_noc = st.slider("Caudal mín. nocturno (L/s)", 0.0, 25.0, 8.0)
        var_presion = st.slider("Variación de presión", -10.0, 10.0, 0.0)
        consumo = st.slider("Consumo facturado (m³/h)", 0.0, 150.0, 80.0)
        reportes = st.slider("Reportes previos (12 meses)", 0, 10, 0)

    if st.button("Clasificar lectura", type="primary"):
        fila = pd.DataFrame([{
            'edad_tuberia_anios': edad, 'diametro_mm': diametro, 'presion_mca': presion,
            'caudal_lps': caudal, 'caudal_min_nocturno_lps': caudal_noc,
            'variacion_presion_mca': var_presion, 'consumo_facturado_m3h': consumo,
            'reportes_previos': reportes, 'hora': hora_dia, 'zona': zona, 'material_tuberia': material
        }])
        fila_d = pd.get_dummies(fila, columns=['zona', 'material_tuberia']).reindex(columns=columnas_modelo, fill_value=0)
        fila_d[cols_num] = escalador.transform(fila_d[cols_num])
        pred, proba = predecir_cluster(modelos_locales, fila_d)

        if pred[0] == 1:
            st.error(f"🔴 FUGA DETECTADA — probabilidad: {proba[0]*100:.1f}%")
        else:
            st.success(f"🟢 NORMAL — probabilidad de fuga: {proba[0]*100:.1f}%")