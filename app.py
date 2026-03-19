import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50

# -------------------------
# LOAD MODEL (same architecture)
# -------------------------

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1)(x)

model = Model(inputs=base_model.input, outputs=output)

# LOAD YOUR TRAINED WEIGHTS
model.load_weights("age_model.h5")

# -------------------------
# AGE RANGE FUNCTION
# -------------------------

def age_to_range(age):
    if age <= 12:
        return "Child"
    elif age <= 19:
        return "Teen"
    elif age <= 30:
        return "Young Adult"
    elif age <= 45:
        return "Adult"
    elif age <= 60:
        return "Middle Age"
    else:
        return "Senior"

# -------------------------
# STREAMLIT UI
# -------------------------

st.title("🎯 Age Estimator App")
st.write("Upload a face image and predict age")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # preprocess
    img_resized = cv2.resize(img, (224,224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # prediction
    pred = model.predict(img_array)[0][0]
    pred = np.clip(pred, 0, 100)

    st.subheader(f"Predicted Age: {int(pred)}")
    st.subheader(f"Age Range: {age_to_range(pred)}")