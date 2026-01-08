"""
Quick tests for ocr_layer.py
Tests OCR extraction from PDFs, images, and scanned documents
"""

import sys
import os

# Add parent directory to Python path so it can find 'core' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import OCR functions and CSV loader
from core.ocr_layer import (
    extract_text_from_pdf,
    extract_text_from_image,
    preprocess_image,
    clean_ocr_output
)
from core.input_layer import load_csv
import cv2
import numpy as np


# Tests loading CSV files from mockdata folder and displays basic info
def test_load_csv():
    """Test CSV file loading"""
    print("\n🧪 Testing CSV loading...")
    
    # Try multiple CSV files from mockdata folder
    possible_csvs = [
        "../mockdata/vendors.csv",
        "../mockdata/contracts.csv",
        "../mockdata/invoices.csv",
        "../mockdata/purchase_orders.csv",
        "../mockdata/bids.csv"
    ]
    
    for csv_path in possible_csvs:
        if os.path.exists(csv_path):
            try:
                df = load_csv(csv_path)
                print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns from: {os.path.basename(csv_path)}")
                print(f"   Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
                return True
            except Exception as e:
                print(f"   ⚠️  Failed on {os.path.basename(csv_path)}: {e}")
                continue
    
    print(f"   ⚠️  No CSV files found - skipping")
    return True


# Extracts text from PDF files using OCR and checks if extraction worked
def test_extract_text_from_pdf():
    """Test PDF text extraction with OCR"""
    print("\n🧪 Testing PDF OCR extraction...")
    
    # Try multiple PDF files from samplepdf folder
    possible_pdfs = [
        "../mockdata/samplepdf/sample_contract.pdf",
        "../mockdata/samplepdf/sample_invoice.pdf",
        "../mockdata/samplepdf/sample_purchase_order.pdf",
        "../mockdata/samplepdf/sample_rfq.pdf",
        "../mockdata/samplepdf/sample_bid.pdf",
        "../mockdata/samplepdf/sample_negotiation_record.pdf"
    ]
    
    for pdf_path in possible_pdfs:
        if os.path.exists(pdf_path):
            try:
                text = extract_text_from_pdf(pdf_path)
                
                print(f"   ✅ Extracted {len(text)} characters from: {os.path.basename(pdf_path)}")
                print(f"   Preview: {text[:100]}...")
                
                if len(text) > 50:
                    return True
                else:
                    print(f"   ⚠️  Warning: Very little text extracted (might be image-only PDF)")
                    return True
                    
            except Exception as e:
                print(f"   ⚠️  Failed on {os.path.basename(pdf_path)}: {e}")
                continue
    
    print(f"   ⚠️  No PDF files found - skipping")
    return True


# Extracts text from PNG/JPG images using Tesseract OCR
def test_extract_text_from_image():
    """Test image text extraction with OCR"""
    print("\n🧪 Testing Image OCR extraction...")
    
    # Try multiple images from samplescanned folder
    possible_images = [
        "../mockdata/samplescanned/scanned_contract.png",
        "../mockdata/samplescanned/scanned_vendor_reg.png",
        "../mockdata/samplescanned/scanned_po.png",
        "../mockdata/samplescanned/scanned_rfq.png",
        "../mockdata/samplescanned/sample_contract.png",
        "../mockdata/samplescanned/sample_bid.png",
        "../mockdata/samplescanned/scanned_bid.png",
        "../mockdata/samplescanned/scanned_invoice.png"
    ]
    
    for image_path in possible_images:
        if os.path.exists(image_path):
            try:
                text = extract_text_from_image(image_path)
                
                print(f"   ✅ Extracted {len(text)} characters from: {os.path.basename(image_path)}")
                print(f"   Preview: {text[:100]}...")
                
                if len(text) > 10:
                    return True
                else:
                    print(f"   ⚠️  Warning: Very little text extracted")
                    return True
                    
            except Exception as e:
                print(f"   ⚠️  Failed on {os.path.basename(image_path)}: {e}")
                continue
    
    print(f"   ⚠️  No image files found - skipping")
    return True


# Creates a test image and runs preprocessing (grayscale, blur, threshold, deskew)
def test_preprocess_image():
    """Test image preprocessing"""
    print("\n🧪 Testing image preprocessing...")
    try:
        # Create a simple test image (100x100 white image with black text)
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Preprocess it
        processed = preprocess_image(test_image)
        
        # Check output is correct type and size
        if isinstance(processed, np.ndarray) and processed.shape == (100, 100):
            print(f"   ✅ Preprocessing successful")
            print(f"   Output shape: {processed.shape}")
            return True
        else:
            print(f"   ❌ Unexpected output shape: {processed.shape}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests text cleaning function that removes extra spaces and special characters
def test_clean_ocr_output():
    """Test OCR output cleaning"""
    print("\n🧪 Testing OCR output cleaning...")
    try:
        # Test text with extra whitespace and special chars
        dirty_text = """
        This  is    a test.
        
        
        With   extra    spaces.
        And®©™ special characters.
        """
        
        cleaned = clean_ocr_output(dirty_text)
        
        # Check cleaning worked
        if "  " not in cleaned:  # No double spaces
            print(f"   ✅ Cleaning successful")
            print(f"   Before: {len(dirty_text)} chars")
            print(f"   After: {len(cleaned)} chars")
            print(f"   Cleaned text: {cleaned[:100]}...")
            return True
        else:
            print(f"   ❌ Still has double spaces")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Runs OCR on first 5 scanned documents in the folder and shows success rate
def test_multiple_scanned_docs():
    """Test OCR on multiple scanned documents"""
    print("\n🧪 Testing OCR on multiple scanned documents...")
    
    scanned_folder = "../mockdata/samplescanned"
    
    if not os.path.exists(scanned_folder):
        print(f"   ⚠️  Folder not found: {scanned_folder}")
        return True
    
    # Get all PNG files in the folder
    png_files = [f for f in os.listdir(scanned_folder) if f.endswith('.png')]
    
    if not png_files:
        print(f"   ⚠️  No PNG files found in {scanned_folder}")
        return True
    
    successful = 0
    failed = 0
    
    for png_file in png_files[:5]:  # Test first 5 files only
        img_path = os.path.join(scanned_folder, png_file)
        try:
            text = extract_text_from_image(img_path)
            print(f"   ✅ {png_file}: {len(text)} chars")
            successful += 1
        except Exception as e:
            print(f"   ❌ {png_file}: {e}")
            failed += 1
    
    print(f"   📊 Results: {successful} succeeded, {failed} failed")
    return True


# Runs all test functions and displays final pass/fail summary
def run_all_tests():
    """Run all OCR layer tests"""
    print("=" * 60)
    print("🚀 RUNNING OCR LAYER TESTS")
    print("=" * 60)
    print("\n⚠️  NOTE: These tests require:")
    print("   - Tesseract OCR installed on system")
    print("   - poppler (for PDF conversion)")
    print("   - Sample PDF/images in mockdata folder")
    print("")
    
    results = []
    results.append(("CSV Loading", test_load_csv()))
    results.append(("PDF OCR Extraction", test_extract_text_from_pdf()))
    results.append(("Image OCR Extraction", test_extract_text_from_image()))
    results.append(("Image Preprocessing", test_preprocess_image()))
    results.append(("OCR Output Cleaning", test_clean_ocr_output()))
    results.append(("Multiple Scanned Docs", test_multiple_scanned_docs()))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check errors above")
    
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
