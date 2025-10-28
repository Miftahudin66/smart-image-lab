import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

# ===============================
# Konfigurasi Aplikasi
# ===============================
st.set_page_config(page_title="Smart Image Lab", page_icon="🧠", layout="centered")

st.title("🧠 Smart Image Lab")
st.write("Aplikasi Pengolahan Citra Digital - Filtering, Enhancement, dan Restorasi Gambar")

st.markdown("---")

# ===============================
# Upload Gambar
# ===============================
uploaded_file = st.file_uploader("📸 Upload Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# ===============================
# Fungsi Filter dan Enhancement
# ===============================

def apply_filter(image, filter_type):
    """Fungsi utama untuk menerapkan berbagai filter pada gambar"""
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    if filter_type == "Grayscale":
        result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    elif filter_type == "Gaussian Blur":
        result = cv2.GaussianBlur(img_cv, (15, 15), 0)

    elif filter_type == "Median Blur":
        result = cv2.medianBlur(img_cv, 9)

    elif filter_type == "Edge Detection (Canny)":
        edges = cv2.Canny(img_cv, 100, 200)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    elif filter_type == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        result = cv2.filter2D(img_cv, -1, kernel)

    elif filter_type == "Emboss":
        kernel = np.array([[ -2, -1, 0],
                           [ -1, 1, 1],
                           [ 0, 1, 2]])
        result = cv2.filter2D(img_cv, -1, kernel)

    elif filter_type == "Pencil Sketch":
        gray, sketch = cv2.pencilSketch(img_cv, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
        result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    elif filter_type == "Stylization (Watercolor Effect)":
        result = cv2.stylization(img_cv, sigma_s=60, sigma_r=0.6)

    elif filter_type == "HDR Effect":
        result = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.15)

    else:
        result = img_cv

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Fungsi untuk peningkatan kualitas (enhancement) citra"""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)

    return image


def restore_image(image):
    """Fungsi sederhana untuk restorasi (noise removal)"""
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    restored = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    return restored


# ===============================
# Bagian Utama Aplikasi
# ===============================
if uploaded_file is not None:
    try:
        # Baca dan tampilkan gambar
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)

        st.markdown("### 🔧 Pilih Mode Operasi")
        mode = st.radio("Pilih operasi yang ingin dilakukan:", 
                        ["Filtering", "Enhancement", "Restorasi"])

        # -------------------- FILTER --------------------
        if mode == "Filtering":
            filter_option = st.selectbox("Pilih Jenis Filter:", 
                ["Grayscale", "Gaussian Blur", "Median Blur", "Edge Detection (Canny)",
                 "Sharpen", "Emboss", "Pencil Sketch", "Stylization (Watercolor Effect)", "HDR Effect"])

            if st.button("🔍 Terapkan Filter"):
                filtered_img = apply_filter(image, filter_option)
                st.image(filtered_img, caption=f"Hasil Filter: {filter_option}", use_container_width=True)

                # Tombol download hasil filter
                img_download = Image.fromarray(filtered_img)
                buf = io.BytesIO()
                img_download.save(buf, format="PNG")
                st.download_button("💾 Download Hasil", data=buf.getvalue(),
                                   file_name=f"filtered_{filter_option}.png", mime="image/png")

        # -------------------- ENHANCEMENT --------------------
        elif mode == "Enhancement":
            st.markdown("### ✨ Pengaturan Enhancement")
            brightness = st.slider("Kecerahan", 0.5, 2.0, 1.0)
            contrast = st.slider("Kontras", 0.5, 2.0, 1.0)
            sharpness = st.slider("Ketajaman", 0.5, 3.0, 1.0)

            if st.button("⚙️ Tingkatkan Gambar"):
                enhanced_img = enhance_image(image, brightness, contrast, sharpness)
                st.image(enhanced_img, caption="Hasil Enhancement", use_container_width=True)

                buf = io.BytesIO()
                enhanced_img.save(buf, format="PNG")
                st.download_button("💾 Download Hasil Enhancement", data=buf.getvalue(),
                                   file_name="enhanced_image.png", mime="image/png")

        # -------------------- RESTORASI --------------------
        elif mode == "Restorasi":
            if st.button("🧼 Lakukan Restorasi"):
                restored_img = restore_image(image)
                st.image(restored_img, caption="Hasil Restorasi", use_container_width=True)

                buf = io.BytesIO()
                Image.fromarray(restored_img).save(buf, format="PNG")
                st.download_button("💾 Download Hasil Restorasi", data=buf.getvalue(),
                                   file_name="restored_image.png", mime="image/png")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

else:
    st.info("👆 Silakan upload gambar terlebih dahulu untuk memulai.")
