"""
OCR Layer - Extract text from scanned PDFs and images
Handles image preprocessing and text extraction
"""

import os
import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from typing import Optional, List


# Converts scanned PDF pages to images and extracts text using OCR
def extract_text_from_pdf(pdf_path: str, dpi: int = 300) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    pages = convert_from_path(pdf_path, dpi)
    extracted_text = ""
    
    for page_num, page_image in enumerate(pages, start=1):
        preprocessed = preprocess_image(np.array(page_image))
        pil_image = Image.fromarray(preprocessed)
        text = pytesseract.image_to_string(pil_image, lang='eng')
        extracted_text += f"--- Page {page_num} ---\n{text}\n"
    
    return clean_ocr_output(extracted_text)


# Extracts text from image files using Tesseract OCR
def extract_text_from_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    preprocessed = preprocess_image(img)
    pil_image = Image.fromarray(preprocessed)
    text = pytesseract.image_to_string(pil_image, lang='eng')
    
    return clean_ocr_output(text)


# Applies grayscale, noise reduction, thresholding, and deskewing to improve OCR accuracy
def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return binary
    
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    
    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        binary, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return deskewed


# Removes extra whitespace, special characters, and normalizes line breaks
def clean_ocr_output(text: str) -> str:
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.strip()
    
    return text


# Test functions
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TESTING OCR LAYER")
    print("=" * 60)
    
    # Test PDF OCR
    print("\n🧪 Testing PDF OCR...")
    try:
        pdf_path = "../mockdata/samplepdf/sample_invoice.pdf"
        pdf_text = extract_text_from_pdf(pdf_path)
        print(f"   ✅ Extracted {len(pdf_text)} characters")
        print(f"   Preview: {pdf_text[:150]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test Image OCR
    print("\n🧪 Testing Image OCR...")
    try:
        # FIXED: Use actual filename from your folder
        img_path = "../mockdata/samplescanned/scanned_contract.png"
        img_text = extract_text_from_image(img_path)
        print(f"   ✅ Extracted {len(img_text)} characters")
        print(f"   Preview: {img_text[:150]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ OCR LAYER TESTING COMPLETE")
    print("=" * 60)
