from enum import Enum
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# Titlul aplicației
st.title("🐾 Serviciu de Generare Imagini cu Pisici 🐾")
st.markdown(
    """
    **Bine ai venit la cel mai adorabil generator de imagini cu pisici!**  
    Folosește opțiunile de mai jos pentru a crea și vizualiza imagini cu pisici în funcție de preferințele tale.
    """
)

# Setări în sidebar
st.sidebar.header("Setări generare 🖼️")


class CatGeneratorsModels(Enum):
    DCGan = "DC Gan"
    StyleGan2 = "Style Gan2"
    RealImages = "Real Images"


model_map_size = {
    CatGeneratorsModels.DCGan.value: 64,
    CatGeneratorsModels.StyleGan2.value: 256,
    CatGeneratorsModels.RealImages.value: 256,
}

# Selectează modelul de generare
selected_model = st.sidebar.radio(
    "Alege modelul de generare a pisicilor:",
    [CatGeneratorsModels.DCGan.value, CatGeneratorsModels.StyleGan2.value, CatGeneratorsModels.RealImages.value],
    index=0,
)

# Folosește session_state pentru a păstra starea imaginii generate
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None


# Funcție pentru a genera imaginea
def generate_cat_image():
    if selected_model == CatGeneratorsModels.DCGan.value:
        from dcgan import generate_single_image, generator
        st.session_state.generated_image = Image.open(
            generate_single_image(generator, 100)
        )
        st.session_state.current_model = selected_model
    elif selected_model == CatGeneratorsModels.StyleGan2.value:
        from stylegan import generate_single_image, generator
        st.session_state.generated_image = Image.open(
            generate_single_image(generator)
        )
        st.session_state.current_model = selected_model
    elif selected_model == CatGeneratorsModels.RealImages.value:
        response = requests.get("https://cataas.com/cat")
        if response.status_code == 200:
            st.session_state.generated_image = Image.open(BytesIO(response.content))
            st.session_state.current_model = selected_model
        else:
            st.error("Oops! Nu s-a putut genera imaginea. Încearcă din nou.")


# Secțiunea principală pentru generare
st.subheader(f"🔍 Generare imagine de pisică [{selected_model}]")
if st.button("Generează imagine"):
    generate_cat_image()

# Afișăm imaginea generată, dacă există
if st.session_state.generated_image and st.session_state.current_model == selected_model:
    if selected_model != CatGeneratorsModels.RealImages.value:
        caption = f"Pisica generată folosind [{selected_model}] 😻"
    else:
        caption = "Pisica descarcata de pe https://cataas.com/cat 😻"

    st.image(
        st.session_state.generated_image,
        caption=caption,
        width=model_map_size[selected_model],
    )

# Galerie dinamică cu imagini statice
st.subheader("🎨 Galerie de imagini cu pisici")
st.markdown("Explorează câteva pisici generate recent:")
cols = st.columns(3)

# Menținem imaginile din galerie constante pentru sesiune
if "gallery_images" not in st.session_state:
    st.session_state.gallery_images = []

if not st.session_state.gallery_images:
    for _ in range(3):
        response = requests.get("https://cataas.com/cat?random")
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.session_state.gallery_images.append(image)

# Afișăm imaginile din galerie
for col, img in zip(cols, st.session_state.gallery_images):
    col.image(img, use_container_width=True)

# Footer aplicație
st.markdown(
    """
    ---
    🐈 **Aplicație creată cu drag pentru iubitorii de pisici.**  
    Contribuie la proiect sau trimite feedback! 😻
    """
)
