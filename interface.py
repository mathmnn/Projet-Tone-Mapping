import streamlit as st
import cv2
import numpy as np
import time
import os
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

# ==========================================
# 1. FONCTIONS MATHÉMATIQUES (Ton algorithme)
# ==========================================

def wls_filter_color(img, lamb, alpha, eps=1e-4): 
    nb_l, nb_c, nb_ch = img.shape
    nb_p = nb_l * nb_c 

    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dx_main, dx_plus1 = -np.ones(nb_p), np.ones(nb_p)
    dx_main[nb_c-1::nb_c], dx_plus1[nb_c-1::nb_c] = 0, 0
    Dx = diags([dx_main, dx_plus1], [0, 1], shape=(nb_p, nb_p)) 

    dy_main, dy_plus_nbc = -np.ones(nb_p), np.ones(nb_p)
    dy_main[-nb_c:] = 0 
    Dy = diags([dy_main, dy_plus_nbc], [0, nb_c], shape=(nb_p, nb_p))

    L = np.log(img_gris + 1e-4)
    L_flat = L.flatten()

    grad_x = Dx @ L_flat
    grad_y = Dy @ L_flat

    ax = 1 / (np.abs(grad_x)**alpha + eps)
    ay = 1 / (np.abs(grad_y)**alpha + eps)

    Ax = diags([ax], [0], shape=(nb_p, nb_p))
    Ay = diags([ay], [0], shape=(nb_p, nb_p))

    Lg = Dx.T @ Ax @ Dx + Dy.T @ Ay @ Dy

    I = eye(nb_p) 
    A_sys = I + lamb * Lg
    A_sys = A_sys.tocsr()

    img_lisse = np.zeros_like(img)
    for c in range(nb_ch):
        g_canal = img[:, :, c].flatten()
        u_flat = spsolve(A_sys, g_canal)
        img_lisse[:, :, c] = u_flat.reshape((nb_l, nb_c))

    return img_lisse

def recombine_multiscale(image_orig, base1, base2, base3, lf, lm, lc):
    res = base3 + lc*(base2 - base3) + lm*(base1 - base2) + lf*(image_orig - base1)
    return np.clip(res, 0.0, 1.0)

# ==========================================
# 2. INTERFACE GRAPHIQUE (Streamlit)
# ==========================================

st.set_page_config(page_title="HDR Multi-Scale Tone Mapping", layout="wide")
st.title("Studio de Tone Mapping Multi-Échelle (Filtre WLS)")

# --- BARRE LATÉRALE (Contrôles principaux) ---
st.sidebar.header("1. Configuration")

# AJOUT : Choix de la source de l'image
image_source = st.sidebar.radio(
    "Source de l'image", 
    ["Image de démo (castle.jpg)", "Uploader une image personnelle"]
)

img_bgr = None

# Logique de chargement selon le choix
if image_source == "Image de démo (castle.jpg)":
    if os.path.exists('castle.jpg'):
        img_bgr = cv2.imread('castle.jpg')
    else:
        st.sidebar.error("Le fichier 'castle.jpg' est introuvable. Vérifie qu'il est bien sur GitHub !")
else:
    uploaded_file = st.sidebar.file_uploader("Charge une image test (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

# Si une image est chargée (soit par défaut, soit uploadée), on affiche la suite
if img_bgr is not None:
    
    # Sécurité taille (on redimensionne si c'est trop grand pour éviter que spsolve ne plante)
    h, w = img_bgr.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        st.sidebar.warning(f"Image redimensionnée à {img_bgr.shape[1]}x{img_bgr.shape[0]} pour la fluidité.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0

    # Paramètre Alpha
    alpha = st.sidebar.number_input("Paramètre Alpha (Sensibilité contours)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

    # Bouton de calcul lourd
    if st.sidebar.button("Calculer les couches WLS", type="primary"):
        start_time = time.time()
        
        # Barre de progression
        progress_bar = st.progress(0, text="Initialisation...")
        
        progress_bar.progress(10, text="Calcul de la couche 1 (λ = 0.125)...")
        d1 = wls_filter_color(img_float, 0.125, alpha)
        
        progress_bar.progress(40, text="Calcul de la couche 2 (λ = 0.5)...")
        d2 = wls_filter_color(img_float, 0.5, alpha)
        
        progress_bar.progress(70, text="Calcul de la couche 3 (λ = 2.0)...")
        d3 = wls_filter_color(img_float, 2.0, alpha)
        
        progress_bar.progress(100, text="Calcul terminé ! ✔️")
        
        # Sauvegarde dans la mémoire de l'interface (Session State)
        st.session_state['d1'] = d1
        st.session_state['d2'] = d2
        st.session_state['d3'] = d3
        st.session_state['img_float'] = img_float
        st.session_state['calc_time'] = time.time() - start_time

# --- ZONE PRINCIPALE (Modifications instantanées) ---
if 'd1' in st.session_state:
    st.success(f" Couches calculées en {st.session_state['calc_time']:.2f} secondes. Tu peux maintenant manipuler les détails en temps réel !")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        lf = st.slider("🔍 Détails Fins (lf)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    with col2:
        lm = st.slider("🪄 Détails Moyens (lm)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    with col3:
        lc = st.slider("🧱 Détails Grossiers (lc)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    # Recombinaison instantanée (très rapide, pas besoin de recharger le WLS)
    img_finale = recombine_multiscale(st.session_state['img_float'], 
                                      st.session_state['d1'], 
                                      st.session_state['d2'], 
                                      st.session_state['d3'], 
                                      lf, lm, lc)

    # Affichage comparatif
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(st.session_state['img_float'], caption="Image Originale", use_container_width=True)
    with col_img2:
        st.image(img_finale, caption="Résultat Multi-Échelle", use_container_width=True)
else:
    st.info("👈 Sélectionne une image et clique sur 'Calculer les couches WLS' dans le menu de gauche pour commencer.")
