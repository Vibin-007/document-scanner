import streamlit as st
import cv2
import pytesseract
import numpy as np
import os
import re
from openai import OpenAI

# REQUIRED: Download language packs (tam.traineddata, tel.traineddata, etc.) 
# from https://github.com/tesseract-ocr/tessdata
# Place them in C:\Program Files\Tesseract-OCR\tessdata

st.set_page_config(page_title="Document Scanner", layout="wide")

def inject_custom_css():
    st.markdown("""
    <style>
        /* Unique premium UI overrides */
        .stApp {
            background: radial-gradient(circle at top, #130e24 0%, #000000 70%);
            color: #ffffff;
        }

        /* Standard Header */
        h1 {
            font-size: 3.5rem !important;
            font-weight: 900 !important;
            color: #ffffff !important;
            text-align: center;
            margin-bottom: 40px !important;
            letter-spacing: -1.5px;
            padding-top: 20px;
        }

        /* Modern Primary Button */
        .stButton > button[kind="primary"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 50px !important;
            padding: 16px 32px !important;
            font-weight: 800 !important;
            font-size: 16px !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            backdrop-filter: blur(15px) !important;
            -webkit-backdrop-filter: blur(15px) !important;
        }

        /* Secondary & Download Buttons */
        .stButton > button[kind="secondary"], .stDownloadButton > button {
            background-color: rgba(255, 255, 255, 0.03) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 50px !important;
            backdrop-filter: blur(10px);
        }

        /* Premium inputs & dropdowns (Glass Style) */
        .stSelectbox div[data-baseweb="select"] > div,
        .stTextInput input,
        .stTextArea textarea {
            background-color: #000000 !important;
            color: #ffffff !important;
            
            backdrop-filter: blur(15px) !important;
            -webkit-backdrop-filter: blur(15px) !important;
        }

        /* Override Streamlit Accent Colors in Checkboxes & Radios */
        div[data-testid="stCheckbox"] div[data-checked="true"] {
            background-color: #ffffff !important;
            border-color: #ffffff !important;
        }
        div[data-testid="stCheckbox"] div[data-checked="true"] svg {
            fill: #000000 !important;
            color: #000000 !important;
        }
        div[role="radio"][aria-checked="true"] > div:first-child {
            background-color: #ffffff !important;
            border-color: #ffffff !important;
        }
        div[role="radio"][aria-checked="true"] > div:first-child > div {
            background-color: #000000 !important; 
        }

        /* Tabs Styling (Floating Pills) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
            border-bottom: none !important;
            padding-bottom: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 30px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 10px 24px;
            margin-right: 5px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            border-color: #ffffff !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #000000 !important;
            font-weight: 800 !important;
        }

        /* File Uploader - Glass Effect (aggressive selectors) */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] > div,
        [data-testid="stFileUploader"] > div > div {
            background: transparent !important;
        }

        /* Target the actual dropzone regardless of wrapper */
        [data-testid="stFileUploadDropzone"],
        section[data-testid="stFileUploadDropzone"],
        div[data-testid="stFileUploadDropzone"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 24px !important;
            min-height: 320px !important;
            padding: 50px 30px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 16px !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1) !important;
        }

        /* Fallback — Streamlit sometimes renders as a plain <div> with a specific class structure */
        .uploadedFile, [class*="fileUpload"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 24px !important;
            backdrop-filter: blur(20px) !important;
        }

        [data-testid="stFileUploadDropzone"] button {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 50px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            padding: 12px 28px !important;
            margin-top: 10px !important;
            backdrop-filter: blur(10px) !important;
        }

        [data-testid="stFileUploadDropzone"] small,
        [data-testid="stFileUploadDropzone"] span,
        [data-testid="stFileUploadDropzone"] p {
            color: rgba(255, 255, 255, 0.6) !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0a0f 0%, #000000 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        /* Images Container Styling */
        [data-testid="stImage"] {
            border-radius: 16px !important;
            overflow: hidden !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        [data-testid="stImage"] > img {
            border-radius: 16px !important;
        }

        /* Success Messages */
        .stAlert[data-baseweb="notification"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            backdrop-filter: blur(10px);
        }
        
        .stAlert[data-baseweb="notification"] p, 
        .stAlert[data-baseweb="notification"] span {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Subheaders / Headers */
        h2, h3 {
            font-weight: 800 !important;
            letter-spacing: -0.5px !important;
            color: #ffffff !important;
        }

        /* Divider */
        hr {
            border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
            margin: 2.5em 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

client = OpenAI(
    api_key="YOUR_API_KEY", 
    base_url="https://api.cometapi.com/v1"
)

def preprocess_image(image, enhance_contrast=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    try:
        osd = pytesseract.image_to_osd(gray, config='--psm 0')
        angle = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
        if angle == 90:
            gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            gray = cv2.rotate(gray, cv2.ROTATE_180)
        elif angle == 270:
            gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except:
        pass 

    if enhance_contrast:
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
        )
    else:
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    return processed

def extract_entities(text):
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    return list(set(emails)), list(set(phones))

def analyze_document(text, mode, target_lang="English"):
    if mode == "General Summary":
        sys_prompt = "You are a helpful assistant. Provide a concise summary and 3 key bullet points of the text."
    elif mode == "Invoice / Receipt":
        sys_prompt = "You are an accounting assistant. Extract the following from the text: 1. Vendor/Store Name, 2. Date, 3. Total Amount. Format as a clean list. If not found, say 'Not Found'."
    elif mode == "Translate":
        sys_prompt = f"You are a professional translator. Translate the following text into {target_lang}. Preserve the original formatting as much as possible."

    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Document Text:\n\n{text}"}
        ]
    )
    return response.choices[0].message.content

st.title("Document Scanner")

with st.sidebar:
    st.header("OCR Engine Settings")
    input_method = st.radio("Input Method:", ("Upload Image", "Webcam Scanner"))
    
    language_map = {
        "English": "eng",
        "Tamil": "tam",
        "Telugu": "tel",
        "Malayalam": "mal",
        "Hindi": "hin",
        "Kannada": "kan"
    }
    
    selected_lang = st.selectbox("Document Language:", list(language_map.keys()))
    ocr_lang = language_map[selected_lang]
    
    ocr_mode = st.selectbox("OCR Layout (PSM):", [6, 1], index=0)
    enhance = False
    
    st.divider()
    st.header("AI Analysis Settings")
    ai_mode = st.selectbox("Processing Mode:", ["General Summary", "Invoice / Receipt", "Translate"])
    
    target_language = "English"
    if ai_mode == "Translate":
        target_language = st.selectbox("Translate to:", list(language_map.keys()), index=0)

image_file = None

if input_method == "Webcam Scanner":
    image_file = st.camera_input("Capture Document")
else:
    image_file = st.file_uploader("Upload Document Image", type=["jpg", "jpeg", "png"])

    # Inject glass style AFTER uploader renders so it overrides Streamlit defaults
    st.markdown("""
    <style>
        /* Glass Uploader — injected late to win specificity war */
        div[data-testid="stFileUploadDropzone"] {
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.25) !important;
            border-radius: 24px !important;
            min-height: 300px !important;
            padding: 50px 30px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 16px !important;
            backdrop-filter: blur(24px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5),
                        inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        }
        div[data-testid="stFileUploadDropzone"] > div {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 12px !important;
            width: 100% !important;
        }
        div[data-testid="stFileUploadDropzone"] button {
            background: rgba(255, 255, 255, 0.12) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 50px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            padding: 12px 28px !important;
            backdrop-filter: blur(10px) !important;
        }
        div[data-testid="stFileUploadDropzone"] span,
        div[data-testid="stFileUploadDropzone"] small,
        div[data-testid="stFileUploadDropzone"] p {
            color: rgba(255, 255, 255, 0.7) !important;
        }
    </style>
    """, unsafe_allow_html=True)

if image_file is not None:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("Process Document", use_container_width=True, type="primary"):
        with st.spinner("Processing image and extracting text..."):
            processed_img = preprocess_image(img, enhance_contrast=enhance)
            
            with col2:
                st.subheader("OCR View (Processed)")
                st.image(processed_img, use_container_width=True, channels="GRAY")
            
            custom_config = f'-l {ocr_lang} --oem 3 --psm {ocr_mode}'
            extracted_text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
            
        if extracted_text:
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            st.markdown(f"""
            <style>
                @keyframes slideDown {{
                    from {{ opacity: 0; transform: translateY(-30px) scale(0.95); }}
                    to   {{ opacity: 1; transform: translateY(0) scale(1); }}
                }}
                @keyframes autoFadeOut {{
                    0%   {{ opacity: 1; transform: translateY(0); }}
                    80%  {{ opacity: 1; transform: translateY(0); }}
                    100% {{ opacity: 0; transform: translateY(-20px); pointer-events: none; }}
                }}
                .extract-popup {{
                    position: fixed;
                    top: 30px;
                    left: 50%;
                    transform: translateX(-50%);
                    z-index: 9999;
                    background: rgba(15, 15, 15, 0.8);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 20px;
                    padding: 24px 40px;
                    min-width: 320px;
                    text-align: center;
                    backdrop-filter: blur(30px) saturate(180%);
                    -webkit-backdrop-filter: blur(30px) saturate(180%);
                    box-shadow: 0 20px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08);
                    animation: slideDown 0.4s cubic-bezier(0.34,1.56,0.64,1) forwards,
                               autoFadeOut 5s 0.4s ease-in-out forwards;
                    color: #ffffff;
                    font-family: sans-serif;
                }}
                .extract-popup .icon {{
                    font-size: 36px;
                    display: block;
                    margin-bottom: 8px;
                }}
                .extract-popup h3 {{
                    margin: 0 0 4px 0;
                    font-size: 18px;
                    font-weight: 800;
                    color: #ffffff !important;
                }}
                .extract-popup p {{
                    margin: 0;
                    font-size: 13px;
                    color: rgba(255,255,255,0.5);
                }}
            </style>
            <div class="extract-popup" id="extractPopup">
                <span class="icon">✦</span>
                <h3>Text Extracted!</h3>
                <p>{word_count} words &nbsp;·&nbsp; {char_count} characters</p>
            </div>
            <script>
                setTimeout(() => {{
                    const el = document.getElementById('extractPopup');
                    if (el) el.remove();
                }}, 5500);
            </script>
            """, unsafe_allow_html=True)
            
            emails, phones = extract_entities(extracted_text)
            
            tab1, tab2, tab3 = st.tabs(["AI Output", "Found Entities", "Raw Text"])
            
            with tab3:
                line_count = len(extracted_text.splitlines())
                st.markdown(f"""
                <style>
                    .stats-bar {{
                        display: flex;
                        gap: 12px;
                        margin-bottom: 16px;
                        flex-wrap: wrap;
                    }}
                    .stat-card {{
                        flex: 1;
                        min-width: 100px;
                        background: rgba(255,255,255,0.05);
                        border: 1px solid rgba(255,255,255,0.12);
                        border-radius: 14px;
                        padding: 14px 20px;
                        text-align: center;
                        backdrop-filter: blur(10px);
                    }}
                    .stat-card .val {{
                        font-size: 28px;
                        font-weight: 900;
                        color: #ffffff;
                        letter-spacing: -1px;
                        line-height: 1;
                    }}
                    .stat-card .lbl {{
                        font-size: 11px;
                        color: rgba(255,255,255,0.45);
                        text-transform: uppercase;
                        letter-spacing: 1.2px;
                        margin-top: 4px;
                    }}
                </style>
                <div class="stats-bar">
                    <div class="stat-card">
                        <div class="val">{word_count}</div>
                        <div class="lbl">Words</div>
                    </div>
                    <div class="stat-card">
                        <div class="val">{char_count}</div>
                        <div class="lbl">Characters</div>
                    </div>
                    <div class="stat-card">
                        <div class="val">{line_count}</div>
                        <div class="lbl">Lines</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.text_area("Raw Extracted Text", extracted_text, height=300, label_visibility="collapsed")
                st.download_button("Download Text", extracted_text, file_name="raw_text.txt")
                
            with tab2:
                st.write("**Emails Found:**")
                st.write(emails if emails else "None detected.")
                st.write("**Phone Numbers Found:**")
                st.write(phones if phones else "None detected.")

            with tab1:
                with st.spinner("Analyzing document..."):
                    try:
                        analysis_result = analyze_document(extracted_text, ai_mode, target_language)
                        st.markdown(analysis_result)
                        st.download_button("Download AI Report", analysis_result, file_name="ai_report.txt")
                    except Exception as e:
                        st.error(f"API Error: {e}")
        else:
            st.error("No text found.")
