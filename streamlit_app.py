import streamlit as st
from fastai.vision.all import *
import gdown

st.title('Mongolian Food Classifier')

st.subheader('Upload your image here')

image_file = st.file_uploader("Upload an image", type=["png", "jpeg", "jpg"])

model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1ZxdUZ-QtlhrcngA4UVhqJpZ5u2J1VLBr'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

if image_file is not None:
    img = PILImage.create(image_file)
    st.image(img)
    pred, pred_idx, probs = learn_inf.predict(img)

    st.markdown(f"""### Predicted food: {pred.capitalize()}""")
    st.markdown(f"""### Probability: {round(max(probs.tolist()), 3) * 100}%""")