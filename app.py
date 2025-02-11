import streamlit as st
from nst import stylizeImage
from PIL import Image
import numpy as np

def main():
    st.title("Image Cartoonification using Neural Style Transfer")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    
        if st.button("Stylize"):
            data = stylizeImage(content_image, style_choice)
            img = data.clone().clamp(0, 255).numpy()
            img = img.transpose(1, 2, 0).astype("uint8")
            img = Image.fromarray(img)
            st.image(img, caption='Stylized Image.', use_column_width=True)

if __name__ == "__main__":
    main()