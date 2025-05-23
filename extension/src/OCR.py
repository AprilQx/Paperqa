"""
OCR the pdf file with Mistral OCR, save to JSON, and prepare for PaperQA2 integration.
"""

import os
import re
import json
import time
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging

# Mistral AI imports
from mistralai import Mistral, DocumentURLChunk


# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MistralOCR")

# Path configurations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PAPERS_DIR = os.path.join(PROJECT_ROOT, 'data', 'cosmopaperqa_paper')
OCR_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'ocr_output')

# Ensure output directory exists
os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
api_key = os.environ.get('MISTRAL_API_KEY')
class MistralOCRProcessor:
    """Process PDFs with Mistral OCR API and prepare for PaperQA2."""
    
    def __init__(self):
        """
        Initialize the OCR processor.
        
        Args:
            api_key: Mistral API key
        """
        self.client = Mistral(api_key=api_key)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF file with Mistral OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text by page
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.is_file():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Upload PDF file to Mistral's OCR service
        try:
            logger.info(f"Uploading PDF: {pdf_file.name}")
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )
            
            # Get URL for the uploaded file
            signed_url = self.client.files.get_signed_url(
                file_id=uploaded_file.id, 
                expiry=60
            )
            
            # Process PDF with OCR
            logger.info("Processing PDF with OCR...")
            ocr_response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=False  # We don't need images for text extraction
            )
            
            return self._extract_structured_content(ocr_response, pdf_file.stem)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _extract_structured_content(self, ocr_response, pdf_name: str) -> Dict[str, Any]:
        """
        Extract structured content from OCR response.
        
        Args:
            ocr_response: Response from OCR processing
            pdf_name: Name of the PDF file
            
        Returns:
            Dictionary with structured content
        """
        structured_content = {
            "filename": pdf_name,
            "num_pages": len(ocr_response.pages),
            "pages": [],
            "sections": [],
            "full_text": ""
        }
        
        # Process each page
        full_text = []
        current_section = None
        section_content = []
        
        for i, page in enumerate(ocr_response.pages):
            page_num = i + 1
            page_text = page.markdown
            full_text.append(page_text)
            
            # Store page content
            structured_content["pages"].append({
                "page_num": page_num,
                "text": page_text
            })
            
            # Try to identify sections (headers usually start with numbers)
            lines = page_text.split("\n")
            for line in lines:
                # Check for section headers (e.g., "1. Introduction", "2.1 Method")
                section_match = re.match(r'^(\d+\.(?:\d+)?)\s+([A-Z][a-zA-Z\s]+)$', line)
                if section_match:
                    # If we were building a previous section, save it
                    if current_section:
                        structured_content["sections"].append({
                            "section_id": current_section,
                            "content": "\n".join(section_content)
                        })
                    
                    # Start new section
                    current_section = section_match.group(1)
                    section_title = section_match.group(2)
                    section_content = [f"{current_section} {section_title}"]
                else:
                    # Add to current section if one exists
                    if current_section:
                        section_content.append(line)
        
        # Add the last section if it exists
        if current_section and section_content:
            structured_content["sections"].append({
                "section_id": current_section,
                "content": "\n".join(section_content)
            })
        
        # Combine all text
        structured_content["full_text"] = "\n".join(full_text)
        
        return structured_content

    def save_to_json(self, data: Dict[str, Any], output_path: str) -> None:
        """
        Save the structured content to a JSON file.
        
        Args:
            data: Structured content dictionary
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON output to {output_path}")
        except Exception as e:
            logger.error(f"Error saving JSON output: {str(e)}")
            raise

def process_all_pdfs() -> List[str]:
    """
    Process all PDFs in a directory.
    
    Args:
        api_key: Mistral API key
        pdf_dir: Directory containing PDFs
        output_dir: Directory to save JSON outputs
        
    Returns:
        List of paths to the generated JSON files
    """
    processor = MistralOCRProcessor()
    output_files = []
    
    # Find all PDF files in the directory
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith('.pdf')]
    
    # URL mapping
    url_map = {}  # filename -> url
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PAPERS_DIR, pdf_file)
        output_filename = f"{os.path.splitext(pdf_file)[0]}_ocr.json"
        output_path = os.path.join(OCR_OUTPUT_DIR, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            logger.info(f"Skipping already processed file: {pdf_file}")
            output_files.append(output_path)
            continue
        
        try:
            # Process the PDF
            logger.info(f"Processing PDF: {pdf_file}")
            structured_content = processor.process_pdf(pdf_path)
            
            # Save the output
            processor.save_to_json(structured_content, output_path)
            output_files.append(output_path)
            
            # Get the URL for the processed PDF
            url = processor.client.files.get_signed_url(file_id=structured_content["filename"]).url
            url_map[pdf_file] = url
            
            # Sleep to avoid rate limiting
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
    
    # Save the mapping to a JSON file
    try:
        url_json_path = os.path.join(OCR_OUTPUT_DIR, 'ocr_pdf_urls.json')
        with open(url_json_path, 'w', encoding='utf-8') as f:
            json.dump(url_map, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved URL mapping to {url_json_path}")
    except Exception as e:
        logger.error(f"Error saving URL mapping: {str(e)}")
    
    return output_files

def create_paperqa2_compatible_files(ocr_files: List[str]) -> None:
    """
    Convert OCR JSON files to text files compatible with PaperQA2.
    
    Args:
        ocr_files: List of OCR JSON file paths
        output_dir: Directory to save text files
    """
    os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
    
    for ocr_file in ocr_files:
        try:
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            # Create text file with the same name
            filename = os.path.basename(ocr_file)
            base_name = filename.replace("_ocr.json", "")
            txt_path = os.path.join(OCR_OUTPUT_DIR, f"{base_name}.txt")
            
            # Write the full text
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(ocr_data["full_text"])
                
            logger.info(f"Created PaperQA2 compatible file: {txt_path}")
            
        except Exception as e:
            logger.error(f"Error creating text file for {ocr_file}: {str(e)}")

if __name__ == "__main__":

    
    # Process all PDFs
    ocr_files = process_all_pdfs()
    
    # Create PaperQA2 compatible text files
    create_paperqa2_compatible_files(ocr_files)