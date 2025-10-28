import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

st.set_page_config(page_title="Smart Image Lab", layout="wide")

st.title("🎨 Smart Image Lab - Enhanced Edition")
st.write("Eksperimen dengan berbagai filter dan efek cerdas untuk mengubah tampilan gambar Anda!")

uploaded_file = st.file_uploader("📤 Unggah gambar (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Baca gambar menggunakan PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", use_column_width=True)

    # Konversi ke OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.subheader("🧠 Pilih Filter & Efek")
    option = st.selectbox(
        "Pilih jenis efek atau filter:",
        [
            "Grayscale",
            "Blur",
            "Canny Edge Detection",
            "Brightness Control",
            "Contrast Adjustment",
            "Pencil Sketch",
            "Cartoonize",
            "Emboss Effect",
            "Sharpen Image",
            "Sepia Tone",
        ]
    )

    # Default hasil
    result_img = None

    try:
        if option == "Grayscale":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        elif option == "Blur":
            k = st.slider("Tingkat blur", 3, 25, 9, step=2)
            result_img = cv2.GaussianBlur(img_cv, (k, k), 0)

        elif option == "Canny Edge Detection":
            t1 = st.slider("Threshold 1", 50, 200, 100)
            t2 = st.slider("Threshold 2", 50, 200, 150)
            edges = cv2.Canny(img_cv, t1, t2)
            result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        elif option == "Brightness Control":
            factor = st.slider("Kecerahan", 0.1, 3.0, 1.2)
            enhancer = ImageEnhance.Brightness(image)
            result_img = np.array(enhancer.enhance(factor))

        elif option == "Contrast Adjustment":
            factor = st.slider("Kontras", 0.1, 3.0, 1.5)
            enhancer = ImageEnhance.Contrast(image)
            result_img = np.array(enhancer.enhance(factor))

        elif option == "Pencil Sketch":
            gray, sketch = cv2.pencilSketch(img_cv, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            result_img = sketch

        elif option == "Cartoonize":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
            )
            color = cv2.bilateralFilter(img_cv, 9, 250, 250)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            result_img = cartoon

        elif option == "Emboss Effect":
            kernel = np.array([[ -2, -1, 0 ],
                               [ -1,  1, 1 ],
                               [  0,  1, 2 ]])
            embossed = cv2.filter2D(img_cv, -1, kernel) + 128
            result_img = embossed

        elif option == "Sharpen Image":
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            sharpened = cv2.filter2D(img_cv, -1, kernel)
            result_img = sharpened

        elif option == "Sepia Tone":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia = cv2.transform(img_cv, sepia_filter)
            sepia = np.clip(sepia, 0, 255)
            result_img = sepia.astype(np.uint8)

        # Tampilkan hasil akhir
        if result_img is not None:
            st.subheader("📸 Hasil Gambar")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Tombol unduh
            result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            st.download_button(
                label="💾 Unduh Gambar Hasil",
                data=result_pil.tobytes(),
                file_name="hasil_filter.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk mulai menggunakan aplikasi.")
