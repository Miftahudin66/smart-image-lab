import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Smart Image Lab", page_icon="🧠", layout="wide")
st.title("🧠 Smart Image Lab - Aplikasi Pengolahan Citra Digital")

uploaded_file = st.file_uploader("📂 Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar sebagai array numpy (RGB)
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Sidebar
    st.sidebar.header("🎛️ Pilihan Filter & Pengaturan")
    option = st.sidebar.selectbox(
        "Pilih efek/filter:",
        [
            "Asli",
            "Grayscale",
            "Gaussian Blur",
            "Sharpen",
            "Edge Detection (Canny)",
            "Emboss",
            "Sepia",
            "Pencil Sketch",
            "Brightness & Contrast",
            "Auto Enhance",
            "Mirror Effect",
        ],
    )

    # Salin gambar asli
    processed = image_np.copy()

    # ========================== FILTER-FILTER ==============================
    if option == "Grayscale":
        processed = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    elif option == "Gaussian Blur":
        processed = cv2.GaussianBlur(image_np, (15, 15), 0)

    elif option == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        processed = cv2.filter2D(image_np, -1, kernel)

    elif option == "Edge Detection (Canny)":
        edges = cv2.Canny(image_np, 100, 200)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    elif option == "Emboss":
        kernel = np.array([[ -2, -1, 0],
                           [ -1, 1, 1],
                           [ 0, 1, 2]])
        processed = cv2.filter2D(image_np, -1, kernel) + 128

    elif option == "Sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        processed = cv2.transform(image_np, sepia_filter)
        processed = np.clip(processed, 0, 255)

    elif option == "Pencil Sketch":
        try:
            # Pastikan gambar punya 3 channel (BGR)
            if len(image_np.shape) == 2:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Gunakan pencilSketch dari OpenCV
            dst_gray, dst_color = cv2.pencilSketch(
                image_bgr,
                sigma_s=60,
                sigma_r=0.07,
                shade_factor=0.05
            )
            processed = cv2.cvtColor(dst_gray, cv2.COLOR_GRAY2RGB)

        except Exception as e:
            st.warning("⚠️ Fitur Pencil Sketch gagal dijalankan, menggunakan versi alternatif.")
            # Versi alternatif pencil sketch manual
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            processed = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    elif option == "Brightness & Contrast":
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        img_enhance = ImageEnhance.Brightness(image)
        bright_img = img_enhance.enhance(brightness)
        contrast_enhancer = ImageEnhance.Contrast(bright_img)
        processed = np.array(contrast_enhancer.enhance(contrast))

    elif option == "Auto Enhance":
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        processed = cv2.merge((l, a, b))
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)

    elif option == "Mirror Effect":
        flip_type = st.sidebar.radio("Pilih arah mirror:", ("Horizontal", "Vertical"))
        if flip_type == "Horizontal":
            processed = cv2.flip(image_np, 1)
        else:
            processed = cv2.flip(image_np, 0)

    # ========================== TAMPILKAN HASIL ==============================
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("✨ Hasil Olahan")
        st.image(processed, use_container_width=True)

    # ========================== DOWNLOAD HASIL ==============================
    result = Image.fromarray(np.uint8(processed))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button(
        label="💾 Download Hasil",
        data=buf.getvalue(),
        file_name="hasil_olah_citra.png",
        mime="image/png"
    )

else:
    st.info("Silakan upload gambar terlebih dahulu untuk memulai pengolahan citra.")