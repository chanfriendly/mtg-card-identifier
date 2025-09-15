import streamlit as st
import easyocr
import cv2
import numpy as np
import requests
import json
from PIL import Image
import io
import time
import re
from typing import Optional, Tuple, Dict, Any

# Configure Streamlit page
st.set_page_config(
    page_title="MTG Card Scanner",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .card-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .processing-step {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_ocr_reader():
    """Initialize EasyOCR reader once and cache it."""
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        return reader
    except Exception as e:
        st.error(f"Failed to initialize OCR reader: {str(e)}")
        return None

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the uploaded image for better OCR results.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple of (original_image, processed_image)
    """
    original = image.copy()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Crop to focus on card name area (top 25% of image)
    height = blurred.shape[0]
    name_region = blurred[:int(height * 0.25), :]
    
    # Additional sharpening for text clarity
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(name_region, -1, kernel)
    
    return original, sharpened

def extract_text_from_image(reader, processed_image: np.ndarray) -> Tuple[str, float]:
    """
    Extract text from processed image using OCR.
    
    Args:
        reader: EasyOCR reader instance
        processed_image: Preprocessed image
        
    Returns:
        Tuple of (extracted_text, confidence_score)
    """
    try:
        results = reader.readtext(processed_image)
        
        if not results:
            return "", 0.0
        
        # Extract text and calculate average confidence
        texts = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Filter low-confidence detections
                texts.append(text)
                confidences.append(confidence)
        
        if not texts:
            return "", 0.0
        
        # Join all text and calculate average confidence
        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)
        
        return full_text, avg_confidence
        
    except Exception as e:
        st.error(f"OCR extraction failed: {str(e)}")
        return "", 0.0

def clean_card_name(raw_text: str) -> str:
    """
    Clean and normalize extracted card name text.
    
    Args:
        raw_text: Raw OCR text
        
    Returns:
        Cleaned card name
    """
    if not raw_text:
        return ""
    
    # Remove special characters and normalize spacing
    cleaned = re.sub(r'[^\w\s\-\',]', '', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Take only the first line (usually the card name)
    lines = cleaned.split('\n')
    if lines:
        cleaned = lines[0].strip()
    
    return cleaned

def search_scryfall_card(card_name: str) -> Optional[Dict[str, Any]]:
    """
    Search for MTG card using Scryfall API.
    
    Args:
        card_name: Name of the card to search
        
    Returns:
        Card data dictionary or None if not found
    """
    if not card_name:
        return None
    
    try:
        # Use fuzzy search endpoint
        url = f"https://api.scryfall.com/cards/named"
        params = {"fuzzy": card_name}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Try exact search as fallback
            params = {"exact": card_name}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse API response: {str(e)}")
        return None

def display_card_info(card_data: Dict[str, Any]) -> None:
    """
    Display formatted card information.
    
    Args:
        card_data: Card data from Scryfall API
    """
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display card image
        if 'image_uris' in card_data and 'normal' in card_data['image_uris']:
            st.image(card_data['image_uris']['normal'], width=250)
        elif 'card_faces' in card_data and card_data['card_faces']:
            # Handle double-faced cards
            if 'image_uris' in card_data['card_faces'][0]:
                st.image(card_data['card_faces'][0]['image_uris']['normal'], width=250)
    
    with col2:
        st.markdown('<div class="card-info">', unsafe_allow_html=True)
        
        # Card name and mana cost
        name = card_data.get('name', 'Unknown')
        mana_cost = card_data.get('mana_cost', '')
        st.markdown(f"**{name}** {mana_cost}")
        
        # Type line
        type_line = card_data.get('type_line', '')
        if type_line:
            st.markdown(f"*{type_line}*")
        
        # Oracle text (truncated)
        oracle_text = card_data.get('oracle_text', '')
        if oracle_text:
            if len(oracle_text) > 200:
                oracle_text = oracle_text[:200] + "..."
            st.markdown(f"**Text:** {oracle_text}")
        
        # Set and rarity
        set_name = card_data.get('set_name', '')
        rarity = card_data.get('rarity', '').title()
        if set_name and rarity:
            st.markdown(f"**Set:** {set_name} ({rarity})")
        
        # Market price
        if 'prices' in card_data and card_data['prices'].get('usd'):
            price = card_data['prices']['usd']
            st.markdown(f"**Price:** ${price} USD")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🃏 Magic: The Gathering Card Scanner</h1>', unsafe_allow_html=True)
    st.markdown("Upload a photo of your MTG card and I'll identify it for you!")
    
    # Initialize OCR reader
    with st.spinner("Initializing OCR engine..."):
        reader = initialize_ocr_reader()
    
    if reader is None:
        st.error("Failed to initialize OCR. Please check your installation.")
        st.stop()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a card image",
        type=['jpg', 'jpeg', 'png', 'heic'],
        help="Upload a clear photo of your MTG card. Make sure the card name is visible."
    )
    
    if uploaded_file is not None:
        # Load and display original image
        try:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            # Process image
            with st.spinner("Processing image..."):
                start_time = time.time()
                original, processed = preprocess_image(image_array)
                processing_time = time.time() - start_time
            
            with col2:
                st.subheader("Processed Image")
                st.image(processed, use_column_width=True, channels="GRAY")
            
            # Display processing info
            st.markdown('<div class="processing-step">', unsafe_allow_html=True)
            st.write(f"✅ Image processed in {processing_time:.2f} seconds")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Extract text using OCR
            with st.spinner("Extracting text..."):
                start_time = time.time()
                raw_text, confidence = extract_text_from_image(reader, processed)
                ocr_time = time.time() - start_time
            
            # Display OCR results
            st.markdown('<div class="processing-step">', unsafe_allow_html=True)
            st.write(f"✅ OCR completed in {ocr_time:.2f} seconds")
            st.write(f"**Raw text detected:** {raw_text}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean the extracted text
            cleaned_name = clean_card_name(raw_text)
            
            # Manual override option if confidence is low
            if confidence < 0.7 or not cleaned_name:
                st.warning("OCR confidence is low. Please verify or enter the card name manually.")
                manual_name = st.text_input("Card Name (manual entry):", value=cleaned_name)
                if manual_name:
                    cleaned_name = manual_name
            
            if cleaned_name:
                st.markdown('<div class="processing-step">', unsafe_allow_html=True)
                st.write(f"**Cleaned card name:** {cleaned_name}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Search for card
                with st.spinner("Searching Scryfall database..."):
                    start_time = time.time()
                    card_data = search_scryfall_card(cleaned_name)
                    search_time = time.time() - start_time
                
                st.markdown('<div class="processing-step">', unsafe_allow_html=True)
                st.write(f"✅ Database search completed in {search_time:.2f} seconds")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if card_data:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.write("🎉 **Card found!**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display card information
                    display_card_info(card_data)
                    
                else:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.write("❌ **Card not found**")
                    st.write(f"No card found matching '{cleaned_name}'. Try:")
                    st.write("- Taking a clearer photo")
                    st.write("- Entering the card name manually")
                    st.write("- Checking for spelling variations")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.write("❌ **No text detected**")
                st.write("Could not extract card name. Try:")
                st.write("- Taking a clearer photo")
                st.write("- Ensuring good lighting")
                st.write("- Making sure the card name is visible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Scan another card button
            if st.button("🔄 Scan Another Card", type="primary"):
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📖 How to Use")
        st.write("""
        1. **Take a clear photo** of your MTG card
        2. **Upload the image** using the file uploader
        3. **Wait for processing** - the app will:
           - Enhance the image for better text recognition
           - Extract the card name using OCR
           - Search the Scryfall database
        4. **View results** with card details and current pricing
        5. **Manual override** available if OCR fails
        """)
        
        st.header("💡 Tips for Best Results")
        st.write("""
        - Ensure good lighting
        - Keep the card flat and straight
        - Make sure the card name is clearly visible
        - Avoid shadows and glare
        - Use high resolution images when possible
        """)
        
        st.header("⚙️ Technical Info")
        st.write("""
        - **OCR Engine:** EasyOCR
        - **Image Processing:** OpenCV
        - **Card Database:** Scryfall API
        - **Supported Formats:** JPG, PNG, HEIC
        """)

if __name__ == "__main__":
    main()