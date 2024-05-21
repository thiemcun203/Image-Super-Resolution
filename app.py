import time
import streamlit as st
import subprocess
import numpy as np
from PIL import Image
from io import BytesIO
from models.HAT.hat import *
from models.RCAN.rcan import *
from models.SRGAN.srgan import *
from models.VDSR.vdsr import *
from models.Interpolation.nearest_neighbor import NearestNeighbor_for_deployment
from models.Interpolation.bilinear import Bilinear_for_deployment
from models.Interpolation.bicubic import Bicubic_for_deployment


# Initialize session state for enhanced images
if 'nearest_enhanced_image' not in st.session_state:
    st.session_state['nearest_enhanced_image'] = None
if 'bilinear_enhanced_image' not in st.session_state:
    st.session_state['bilinear_enhanced_image'] = None
if 'bicubic_enhanced_image' not in st.session_state:
    st.session_state['bicubic_enhanced_image'] = None
if 'hat_enhanced_image' not in st.session_state:
    st.session_state['hat_enhanced_image'] = None
if 'rcan_enhanced_image' not in st.session_state:
    st.session_state['rcan_enhanced_image'] = None
if 'srgan_enhanced_image' not in st.session_state:
    st.session_state['srgan_enhanced_image'] = None
if 'srflow_enhanced_image' not in st.session_state:
    st.session_state['srflow_enhanced_image'] = None
if 'vdsr_enhanced_image' not in st.session_state:
    st.session_state['vdsr_enhanced_image'] = None

# Initialize session state for button clicks
if 'nearest_clicked' not in st.session_state:
    st.session_state['nearest_clicked'] = False
if 'bilinear_clicked' not in st.session_state:
    st.session_state['bilinear_clicked'] = False
if 'bicubic_clicked' not in st.session_state:
    st.session_state['bicubic_clicked'] = False
if 'hat_clicked' not in st.session_state:
    st.session_state['hat_clicked'] = False
if 'rcan_clicked' not in st.session_state:
    st.session_state['rcan_clicked'] = False
if 'srgan_clicked' not in st.session_state:
    st.session_state['srgan_clicked'] = False 
if 'srflow_clicked' not in st.session_state:
    st.session_state['srflow_clicked'] = False
if 'vdsr_clicked' not in st.session_state:
    st.session_state['vdsr_clicked'] = False

st.markdown("<h1 style='text-align: center'>Image Super Resolution</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the input source", ["Upload image", "Take a photo"])

# Depending on the choice, show the uploader widget or webcam capture
if app_mode == "Upload image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"], on_change=lambda: reset_states())
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
elif app_mode == "Take a photo":
    camera_input = st.camera_input("Take a picture", on_change=lambda: reset_states())
    if camera_input is not None:
        image = Image.open(camera_input).convert("RGB")
        
def reset_states():
    st.session_state['hat_enhanced_image'] = None
    st.session_state['rcan_enhanced_image'] = None
    st.session_state['srgan_enhanced_image'] = None
    st.session_state['srflow_enhanced_image'] = None
    st.session_state['bicubic_enhanced_image'] = None
    st.session_state['bilinear_enhanced_image'] = None
    st.session_state['nearest_enhanced_image'] = None
    st.session_state['vdsr_enhanced_image'] = None
    st.session_state['hat_clicked'] = False
    st.session_state['rcan_clicked'] = False
    st.session_state['srgan_clicked'] = False
    st.session_state['srflow_clicked'] = False
    st.session_state['bicubic_clicked'] = False
    st.session_state['bilinear_clicked'] = False
    st.session_state['nearest_clicked'] = False
    st.session_state['vdsr_clicked'] = False
    
def get_image_download_link(img, filename):
    """Generates a link allowing the PIL image to be downloaded"""
    # Convert the PIL image to Bytes
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return st.download_button(
        label="Download Image",
        data=buffered.getvalue(),
        file_name=filename,
        mime="image/png"
    )
    
if 'image' in locals():
    # st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    # ------------------------ Nearest Neighbor ------------------------ #
    if st.button('Enhance with Nearest Neighbor'):
        with st.spinner('Processing using Nearest Neighbor...'):
            enhanced_image = NearestNeighbor_for_deployment(image)
            st.session_state['nearest_enhanced_image'] = enhanced_image
            st.session_state['nearest_clicked'] = True
            st.success('Done!')
    if st.session_state['nearest_enhanced_image'] is not None:
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(image, use_column_width=True)
        col2.header("Enhanced")
        col2.image(st.session_state['nearest_enhanced_image'], use_column_width=True)
        with col2:
            get_image_download_link(st.session_state['nearest_enhanced_image'], 'nearest_enhanced.jpg')

    # ------------------------ Bilinear ------------------------ #
    if st.button('Enhance with Bilinear'):
        with st.spinner('Processing using Bilinear...'):
            enhanced_image = Bilinear_for_deployment(image)
            st.session_state['bilinear_enhanced_image'] = enhanced_image
            st.session_state['bilinear_clicked'] = True
            st.success('Done!')
    if st.session_state['bilinear_enhanced_image'] is not None:
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(image, use_column_width=True)
        col2.header("Enhanced")
        col2.image(st.session_state['bilinear_enhanced_image'], use_column_width=True)
        with col2:
            get_image_download_link(st.session_state['bilinear_enhanced_image'], 'bilinear_enhanced.jpg')
    
    # ------------------------ Bicubic ------------------------ #
    if st.button('Enhance with Bicubic'):
        with st.spinner('Processing using Bicubic...'):
            enhanced_image = Bicubic_for_deployment(image)
            st.session_state['bicubic_enhanced_image'] = enhanced_image
            st.session_state['bicubic_clicked'] = True
            st.success('Done!')
    if st.session_state['bicubic_enhanced_image'] is not None:
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(image, use_column_width=True)
        col2.header("Enhanced")
        col2.image(st.session_state['bicubic_enhanced_image'], use_column_width=True)
        with col2:
            get_image_download_link(st.session_state['bicubic_enhanced_image'], 'bicubic_enhanced.jpg')
            
    # --------------------------SRGAN-------------------------- #
    if st.button('Enhance with SRGAN'):
        with st.spinner('Processing using SRGAN...'):
            with st.spinner('Wait for it... the model is processing the image'):
                srgan_model = GeneratorResnet()
                device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
                checkpoint = torch.load('models/SRGAN/srgan_checkpoint.pth', map_location=device)
                srgan_model.load_state_dict(checkpoint['generator'])
                enhanced_image = srgan_model.inference(image)
                st.session_state['srgan_enhanced_image'] = enhanced_image
            st.session_state['srgan_clicked'] = True 
            st.success('Done!')
    if st.session_state['srgan_enhanced_image'] is not None:
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(image, use_column_width=True)
        col2.header("Enhanced")
        col2.image(st.session_state['srgan_enhanced_image'], use_column_width=True)
        with col2:
            get_image_download_link(st.session_state['srgan_enhanced_image'], 'srgan_enhanced.jpg')
            
    # ------------------------ VDSR ------------------------ #
    if st.button('Enhance with VDSR'):
        with st.spinner('Processing using VDSR...'):
            # Load the VDSR model
            device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
            vdsr_model = VDSR()
            checkpoint = torch.load('models/VDSR/vdsr_checkpoint.pth', map_location=device)
            vdsr_model.load_state_dict(checkpoint['model'])
            enhanced_image = vdsr_model.inference(image)
            st.session_state['vdsr_enhanced_image'] = enhanced_image
            st.session_state['vdsr_clicked'] = True
            st.success('Done!')
    if st.session_state['vdsr_enhanced_image'] is not None:
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(image, use_column_width=True)
        col2.header("Enhanced")
        col2.image(st.session_state['vdsr_enhanced_image'], use_column_width=True)
        with col2:
            get_image_download_link(st.session_state['vdsr_enhanced_image'], 'vdsr_enhanced.jpg')
            
    # ------------------------ RCAN ------------------------ #
    if st.button('Enhance with RCAN'):
        with st.spinner('Processing using RCAN...'):
            with st.spinner('Wait for it... the model is processing the image'):
                rcan_model = RCAN()
                device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
                rcan_model.load_state_dict(torch.load('models/RCAN/rcan_checkpoint.pth', map_location=device))
                enhanced_image = rcan_model.inference(image)
                st.session_state['rcan_enhanced_image'] = enhanced_image
            st.session_state['rcan_clicked'] = True 
            st.success('Done!')
    if st.session_state['rcan_enhanced_image'] is not None:
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(image, use_column_width=True)
        col2.header("Enhanced")
        col2.image(st.session_state['rcan_enhanced_image'], use_column_width=True)
        with col2:
            get_image_download_link(st.session_state['rcan_enhanced_image'], 'rcan_enhanced.jpg')

    # ------------------------ HAT ------------------------ #
    if st.button('Enhance with HAT'):
        with st.spinner('Processing using HAT...'):                             
            with st.spinner('Wait for it... the model is processing the image'):                                                  
                enhanced_image = HAT_for_deployment(image)
                st.session_state['hat_enhanced_image'] = enhanced_image
            st.session_state['hat_clicked'] = True
            st.success('Done!')
    if st.session_state['hat_enhanced_image'] is not None:
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(image, use_column_width=True)
        col2.header("Enhanced")
        col2.image(st.session_state['hat_enhanced_image'], use_column_width=True)
        with col2:
            get_image_download_link(st.session_state['hat_enhanced_image'], 'hat_enhanced.jpg')
    

    



