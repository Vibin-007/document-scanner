# Document Scanner & AI Analyzer ✦

A premium, glassmorphism-themed Streamlit application that combines **OpenCV computer vision**, **Tesseract OCR**, and **GPT-4o (via CometAPI)** to scan documents, extract structured text, and perform AI-driven summaries, translation, or receipt parsing.

---

## Features

### 1. Advanced Document OCR
* **Multi-Language Support:** Pre-configured for English, Hindi, Tamil, Telugu, Malayalam, and Kannada.
* **Auto-Orientation Detection:** Automatically detects document skew/rotation using Tesseract's Orientation and Script Detection (OSD) and auto-corrects the rotation.
* **Image Preprocessing:** Cleans and binarizes uploaded images using adaptive thresholding or Otsu's thresholding for maximum OCR accuracy.

### 2. Smart Entity Extraction
* Built-in regular expression parser to detect:
  * Email addresses
  * Phone numbers

### 3. GPT-4o Document Understanding
Three AI processing modes powered by CometAPI:
* **General Summary:** Generates a concise summary along with three key bullet points.
* **Invoice / Receipt Parser:** Extracts vendor name, transaction date, and total amount in a clean list format.
* **Translator:** Translates the extracted document text into any of the supported target languages while preserving format.

### 4. Interactive UX/UI
* Premium, dark-themed glassmorphic user interface.
* Support for both web-camera capture (`st.camera_input`) and image file uploads (`PNG`, `JPG`, `JPEG`).
* Clean metrics dashboard displaying word count, character count, and line count.
* One-click download buttons for raw extracted text and generated AI reports.

---

## Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your system:
* **Python 3.8+**
* **Tesseract OCR Engine**
  * Windows installation path default: `C:\Program Files\Tesseract-OCR\tesseract.exe`
  * Download the language training data files (`.traineddata`) for Hindi, Tamil, Telugu, Malayalam, and Kannada.

### 2. Setup Language Packs
Ensure you have placed the necessary language training data files in Tesseract's `tessdata` folder:
* **Default Directory:** `C:\Program Files\Tesseract-OCR\tessdata\`
* **Local Workspace Fallback:** You can also run the helper script `download_tessdata.py` to download them to a local `./tessdata` folder:
  ```bash
  python download_tessdata.py
  ```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install streamlit opencv-python pytesseract numpy openai
```

### 4. Configure API Key
Make sure your CometAPI key is configured in `app.py`. The app currently uses:
* **Base URL:** `https://api.cometapi.com/v1`
* **Model:** `gpt-4o`

---

## Running the Application

To start the Streamlit web application, run the following command in your terminal:

```bash
streamlit run app.py
```

This will spin up a local development server, typically accessible at `http://localhost:8501`.

---

## File Directory Structure

* `app.py` - The main Streamlit web application with UI, camera integration, OCR processing, and OpenAI API integration.
* `download_tessdata.py` - A utility script to download specific language packs from the official Tesseract repository.
* `code.py` - A command-line script implementing the document capture (via OpenCV webcam window), image preprocessing, and summarization.
* `tessdata/` - (Optional local directory) Stores the local language `.traineddata` files.
