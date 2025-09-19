import streamlit as st
import easyocr
import cv2
import numpy as np
import requests
import json
import pandas as pd
from PIL import Image
import io
import time
import re
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

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
    .bulk-progress {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
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

def detect_card_orientation(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct card orientation by finding the orientation where the top
    region is significantly more text-heavy than the bottom region.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Correctly oriented image
    """
    orientations = [
        (0, image),
        (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(image, cv2.ROTATE_180)),
        (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    ]

    best_orientation_img = image
    max_score = -float('inf')

    for angle, oriented_img in orientations:
        height, width = oriented_img.shape

        # Skip orientations that are clearly not portrait-like for a card
        if width > height:
            continue

        # Region for Card Name (top 15%)
        top_roi = oriented_img[0:int(height * 0.15), :]
        # Region for Set/Artist Info (bottom 15%)
        bottom_roi = oriented_img[int(height * 0.85):, :]

        # Use Canny edge detection as a proxy for text "busyness"
        # A slight blur reduces noise before edge detection
        top_blur = cv2.GaussianBlur(top_roi, (5, 5), 0)
        bottom_blur = cv2.GaussianBlur(bottom_roi, (5, 5), 0)
        
        top_edges = cv2.Canny(top_blur, 50, 150)
        bottom_edges = cv2.Canny(bottom_blur, 50, 150)

        top_score = np.sum(top_edges > 0)
        bottom_score = np.sum(bottom_edges > 0)

        # The heuristic: The top (name/mana cost) should be more edge-dense
        # than the bottom (set info/artist). This helps distinguish 0 from 180 degrees.
        current_score = top_score - bottom_score

        if current_score > max_score:
            max_score = current_score
            best_orientation_img = oriented_img
    
    return best_orientation_img

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the uploaded image for better OCR results.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple of (original_image, name_region, set_info_region)
    """
    original = image.copy()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Auto-detect and correct orientation
    oriented = detect_card_orientation(gray)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(oriented)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    height = filtered.shape[0]
    
    # Extract card name region (top 25%)
    name_region = filtered[:int(height * 0.25), :]
    
    # Extract set info region (bottom 15% - where collector info typically appears)
    set_region = filtered[int(height * 0.85):, :]
    
    # Sharpen both regions for better OCR
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    name_sharpened = cv2.filter2D(name_region, -1, kernel)
    set_sharpened = cv2.filter2D(set_region, -1, kernel)
    
    return original, name_sharpened, set_sharpened

def extract_card_information(reader, name_region: np.ndarray, set_region: np.ndarray) -> Tuple[str, str, float]:
    """
    Extract card name and set information from processed images.
    
    Args:
        reader: EasyOCR reader instance
        name_region: Preprocessed card name region
        set_region: Preprocessed set information region
        
    Returns:
        Tuple of (card_name, set_info, confidence_score)
    """
    try:
        # Extract card name from top region
        name_results = reader.readtext(name_region)
        card_name = ""
        name_confidence = 0.0
        
        if name_results:
            # Get the text with highest confidence from name region
            best_name = max(name_results, key=lambda x: x[2])
            card_name = best_name[1].strip()
            name_confidence = best_name[2]
        
        # Extract set information from bottom region
        set_results = reader.readtext(set_region)
        set_info = ""
        set_confidence = 0.0
        
        if set_results:
            # Look for collector number pattern (like "123/456" or "123★")
            for bbox, text, confidence in set_results:
                text_clean = text.strip()
                # Match patterns like "123/456", "123★", "M15", etc.
                if re.search(r'\d+[/★]\d*|[A-Z]{2,4}|\d{4}', text_clean):
                    if confidence > set_confidence:
                        set_info = text_clean
                        set_confidence = confidence
        
        # Combine confidences (weighted toward name since it's more critical)
        overall_confidence = (name_confidence * 0.7) + (set_confidence * 0.3)
        
        return card_name, set_info, overall_confidence
        
    except Exception as e:
        st.error(f"OCR extraction failed: {str(e)}")
        return "", "", 0.0

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
    
    # Remove obvious OCR artifacts that aren't card names
    # Common patterns like "9 8 2 Q" that are clearly not card names
    if re.match(r'^[\d\s\w]{1,6}$', raw_text) and any(c.isdigit() for c in raw_text):
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

def search_scryfall_card(card_name: str, set_info: str = "") -> Optional[Dict[str, Any]]:
    """
    Search for MTG card using Scryfall API with enhanced set-aware search.
    
    Args:
        card_name: Name of the card to search
        set_info: Additional set information (collector number, set code, etc.)
        
    Returns:
        Card data dictionary or None if not found
    """
    if not card_name:
        return None
    
    try:
        # Add small delay to respect Scryfall rate limits (100ms between requests)
        time.sleep(0.1)
        # Strategy 1: If we have set info, try to use it for more precise search
        if set_info:
            # Try searching with set constraint first
            # Look for set codes (like "M15", "KTK") or collector numbers
            set_match = re.search(r'[A-Z]{2,4}', set_info)
            collector_match = re.search(r'(\d+)[/★]', set_info)
            
            if set_match or collector_match:
                # Try search with set constraint
                search_url = "https://api.scryfall.com/cards/search"
                query_parts = [f'name:"{card_name}"']
                
                if set_match:
                    set_code = set_match.group()
                    query_parts.append(f'set:{set_code}')
                
                if collector_match:
                    collector_num = collector_match.group(1)
                    query_parts.append(f'cn:{collector_num}')
                
                search_query = " ".join(query_parts)
                
                response = requests.get(search_url, params={"q": search_query}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data'):
                        return data['data'][0]  # Return first match
        
        # Strategy 2: Standard fuzzy search fallback
        url = "https://api.scryfall.com/cards/named"
        params = {"fuzzy": card_name}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Try exact search as final fallback
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

def process_single_card(reader, image_file, filename: str) -> Dict[str, Any]:
    """
    Process a single card image and return structured data.
    
    Args:
        reader: EasyOCR reader instance
        image_file: Uploaded image file
        filename: Name of the image file
        
    Returns:
        Dictionary with card processing results
    """
    try:
        # Load and process image
        image = Image.open(image_file)
        image_array = np.array(image)
        
        # Process image regions
        original, name_region, set_region = preprocess_image(image_array)
        
        # Extract information
        card_name, set_info, confidence = extract_card_information(reader, name_region, set_region)
        cleaned_name = clean_card_name(card_name)
        
        # Search for card data
        card_data = None
        if cleaned_name:
            card_data = search_scryfall_card(cleaned_name, set_info)
        
        # Structure the results
        result = {
            'filename': filename,
            'ocr_card_name': card_name,
            'ocr_set_info': set_info,
            'ocr_confidence': round(confidence * 100, 2),  # Convert to percentage
            'cleaned_name': cleaned_name,
            'found_in_database': card_data is not None,
            'card_name': card_data.get('name', '') if card_data else '',
            'mana_cost': card_data.get('mana_cost', '') if card_data else '',
            'type_line': card_data.get('type_line', '') if card_data else '',
            'set_name': card_data.get('set_name', '') if card_data else '',
            'set_code': card_data.get('set', '') if card_data else '',
            'collector_number': card_data.get('collector_number', '') if card_data else '',
            'rarity': card_data.get('rarity', '') if card_data else '',
            'price_usd': card_data.get('prices', {}).get('usd', '') if card_data else '',
            'oracle_text': card_data.get('oracle_text', '')[:200] + '...' if card_data and card_data.get('oracle_text', '') else '',
            'image_url': card_data.get('image_uris', {}).get('normal', '') if card_data else '',
            'scryfall_id': card_data.get('id', '') if card_data else '',
            'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
        
    except Exception as e:
        # Return error result
        return {
            'filename': filename,
            'ocr_card_name': '',
            'ocr_set_info': '',
            'ocr_confidence': 0.0,
            'cleaned_name': '',
            'found_in_database': False,
            'card_name': '',
            'mana_cost': '',
            'type_line': '',
            'set_name': '',
            'set_code': '',
            'collector_number': '',
            'rarity': '',
            'price_usd': '',
            'oracle_text': '',
            'image_url': '',
            'scryfall_id': '',
            'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'error': str(e)
        }

def create_excel_export(results: List[Dict[str, Any]]) -> bytes:
    """
    Create Excel file from processing results.
    
    Args:
        results: List of card processing results
        
    Returns:
        Excel file as bytes
    """
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='MTG Cards', index=False)
        
        # Get the worksheet to format it
        workbook = writer.book
        worksheet = writer.sheets['MTG Cards']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output.read()

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
    st.markdown("Upload photos of your MTG cards and I'll identify them for you!")
    
    # Initialize OCR reader
    with st.spinner("Initializing OCR engine..."):
        reader = initialize_ocr_reader()
    
    if reader is None:
        st.error("Failed to initialize OCR. Please check your installation.")
        st.stop()
    
    # Mode selection
    mode = st.radio(
        "Choose scanning mode:",
        ["Single Card", "Bulk Processing"],
        horizontal=True
    )
    
    if mode == "Single Card":
        # Original single card functionality
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
                    original, name_region, set_region = preprocess_image(image_array)
                    processing_time = time.time() - start_time
                
                # Display processed regions
                with col2:
                    st.subheader("Processed Regions")
                    st.write("**Card Name Area:**")
                    st.image(name_region, use_column_width=True, channels="GRAY")
                    st.write("**Set Info Area:**")
                    st.image(set_region, use_column_width=True, channels="GRAY")
                
                # Display processing info
                st.markdown('<div class="processing-step">', unsafe_allow_html=True)
                st.write(f"✅ Image processed in {processing_time:.2f} seconds")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Extract text using OCR
                with st.spinner("Extracting text..."):
                    start_time = time.time()
                    card_name, set_info, confidence = extract_card_information(reader, name_region, set_region)
                    ocr_time = time.time() - start_time
                
                # Display OCR results
                st.markdown('<div class="processing-step">', unsafe_allow_html=True)
                st.write(f"✅ OCR completed in {ocr_time:.2f} seconds")
                st.write(f"**Card name detected:** '{card_name}'")
                if set_info:
                    st.write(f"**Set info detected:** '{set_info}'")
                st.write(f"**Overall confidence:** {confidence:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean the extracted text
                cleaned_name = clean_card_name(card_name)
                
                # Manual override option if confidence is low
                manual_input_needed = confidence < 0.6 or not cleaned_name
                
                if manual_input_needed:
                    st.warning("⚠️ OCR results need verification. Please check or enter information manually.")
                    col_name, col_set = st.columns(2)
                    with col_name:
                        manual_name = st.text_input("Card Name:", value=cleaned_name)
                    with col_set:
                        manual_set = st.text_input("Set Info (optional):", value=set_info, help="Set code, collector number, or year")
                    
                    if manual_name:
                        cleaned_name = manual_name
                    if manual_set:
                        set_info = manual_set
                
                if cleaned_name:
                    st.markdown('<div class="processing-step">', unsafe_allow_html=True)
                    st.write(f"**Final card name:** '{cleaned_name}'")
                    if set_info:
                        st.write(f"**Set information:** '{set_info}'")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Search for card with enhanced set-aware search
                    with st.spinner("Searching Scryfall database..."):
                        start_time = time.time()
                        card_data = search_scryfall_card(cleaned_name, set_info)
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
    
    else:  # Bulk Processing mode
        st.subheader("Bulk Card Processing")
        st.write("Upload multiple card images to process them all at once and export to Excel.")
        
        uploaded_files = st.file_uploader(
            "Choose card images",
            type=['jpg', 'jpeg', 'png', 'heic'],
            accept_multiple_files=True,
            help="Upload multiple clear photos of your MTG cards. The app will process them sequentially."
        )
        
        if uploaded_files:
            st.write(f"📁 **{len(uploaded_files)} files uploaded**")
            
            # Show file list
            with st.expander("View uploaded files"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name} ({file.size:,} bytes)")
            
            # Process button
            if st.button("🚀 Process All Cards", type="primary"):
                results = []
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.write(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Process the card
                    result = process_single_card(reader, uploaded_file, uploaded_file.name)
                    results.append(result)
                    
                    # Show intermediate results
                    if result['found_in_database']:
                        st.success(f"✅ {uploaded_file.name}: Found '{result['card_name']}'")
                    else:
                        st.warning(f"⚠️ {uploaded_file.name}: Not found (OCR: '{result['ocr_card_name']}')")  
                
                total_time = time.time() - start_time
                
                # Final results summary
                st.markdown('<div class="bulk-progress">', unsafe_allow_html=True)
                found_count = sum(1 for r in results if r['found_in_database'])
                st.write(f"**Processing Complete!**")
                st.write(f"⏱️ Total time: {total_time:.1f} seconds")
                st.write(f"📊 Cards found: {found_count}/{len(results)}")
                st.write(f"🎯 Success rate: {found_count/len(results)*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create and display results DataFrame
                df = pd.DataFrame(results)
                
                # Select key columns for display
                display_columns = ['filename', 'card_name', 'set_name', 'rarity', 'price_usd', 'ocr_confidence', 'found_in_database']
                if all(col in df.columns for col in display_columns):
                    display_df = df[display_columns].copy()
                    display_df.columns = ['File', 'Card Name', 'Set', 'Rarity', 'Price (USD)', 'OCR Confidence (%)', 'Found']
                    
                    st.subheader("📋 Results Summary")
                    st.dataframe(display_df, use_container_width=True)
                
                # Export to Excel
                excel_data = create_excel_export(results)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mtg_cards_scan_{timestamp}.xlsx"
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_confidence = df['ocr_confidence'].mean() if 'ocr_confidence' in df.columns else 0
                    st.metric("Average OCR Confidence", f"{avg_confidence:.1f}%")
                with col2:
                    # Calculate total value from valid price entries
                    valid_prices = df[df['price_usd'] != '']['price_usd']
                    if len(valid_prices) > 0:
                        # Convert to float, handling any non-numeric values
                        numeric_prices = []
                        for price in valid_prices:
                            try:
                                numeric_prices.append(float(price))
                            except (ValueError, TypeError):
                                continue
                        total_value = sum(numeric_prices)
                    else:
                        total_value = 0.0
                    st.metric("Total Collection Value", f"${total_value:.2f}")
                with col3:
                    unique_sets = df[df['set_name'] != '']['set_name'].nunique() if 'set_name' in df.columns else 0
                    st.metric("Unique Sets", unique_sets)
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📖 How to Use")
        
        if mode == "Single Card":
            st.write("""
            **Single Card Mode:**
            1. Upload one card image
            2. Wait for automatic processing
            3. Review and correct OCR results if needed
            4. View card details and pricing
            """)
        else:
            st.write("""
            **Bulk Processing Mode:**
            1. Upload multiple card images at once
            2. Click "Process All Cards"
            3. Monitor progress as each card is processed
            4. Download Excel report with all results
            5. Review statistics and collection value
            """)
        
        st.header("💡 Tips for Best Results")
        st.write("""
        - Ensure good lighting
        - Keep the card flat and straight
        - Make sure both card name and bottom text are visible
        - Avoid shadows and glare
        - Use high resolution images when possible
        - Cards can be uploaded in any orientation
        """)
        
        st.header("📊 Excel Export Includes")
        st.write("""
        - Original filename
        - OCR extracted text and confidence
        - Card name, set, and rarity
        - Mana cost and type line
        - Collector number and set code
        - Current market price (USD)
        - Scryfall ID and image URL
        - Processing timestamp
        """)
        
        st.header("⚙️ Technical Info")
        st.write("""
        - **OCR Engine:** EasyOCR
        - **Image Processing:** OpenCV with auto-rotation
        - **Card Database:** Scryfall API with set-aware search
        - **Supported Formats:** JPG, PNG, HEIC
        - **Features:** Dual-region scanning, bulk processing, Excel export
        - **Rate Limiting:** 100ms delay between API calls
        """)

if __name__ == "__main__":
    main()