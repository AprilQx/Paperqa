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

# Path configurations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PAPERS_DIR = os.path.join(PROJECT_ROOT, 'pdfs')

api_key = os.environ.get('MISTRAL_API_KEY')

class MistralOCRProcessor:
    """Process PDFs with Mistral OCR API and prepare for PaperQA2."""
    
    def __init__(self):
        """Initialize the OCR processor."""
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        self.client = Mistral(api_key=api_key)
    
    def process_single_pdf(self, pdf_path: str, save_markdown: bool = True, save_json: bool = True, output_dir=None) -> Dict[str, Any]:
        """
        Process a single PDF file with Mistral OCR.
        
        Args:
            pdf_path: Path to the PDF file
            save_markdown: Whether to save markdown files
            save_json: Whether to save JSON files
            
        Returns:
            Dictionary with extracted text by page
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Upload PDF file to Mistral's OCR service
        try:
            print(f"Uploading PDF: {pdf_file.name}")
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
            print("Processing PDF with OCR...")
            ocr_response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=False
            )
            
            # Extract structured content
            structured_content = self._extract_structured_content(ocr_response, pdf_file.stem)
            
            # Save outputs
            base_name = pdf_file.stem
            
            if save_json:
                json_path = os.path.join(output_dir, f"{base_name}_ocr.json")
                self.save_to_json(structured_content, json_path)
            
            if save_markdown:
                markdown_path = os.path.join(output_dir, f"{base_name}.md")
                self.save_to_markdown(structured_content, markdown_path)
            
            # Also create PaperQA2 compatible text file
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            self.save_to_text(structured_content, txt_path)
            
            return structured_content
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise
    
    def _extract_structured_content(self, ocr_response, pdf_name: str) -> Dict[str, Any]:
        """Extract structured content from OCR response."""
        structured_content = {
            "filename": pdf_name,
            "num_pages": len(ocr_response.pages),
            "pages": [],
            "sections": [],
            "full_text": "",
            "full_markdown": ""
        }
        
        # Process each page
        full_text = []
        full_markdown = []
        current_section = None
        section_content = []
        
        for i, page in enumerate(ocr_response.pages):
            page_num = i + 1
            page_markdown = page.markdown
            page_text = page.text if hasattr(page, 'text') else page_markdown
            
            full_text.append(page_text)
            full_markdown.append(page_markdown)
            
            # Store page content
            structured_content["pages"].append({
                "page_num": page_num,
                "text": page_text,
                "markdown": page_markdown
            })
            
            # Try to identify sections
            lines = page_markdown.split("\n")
            for line in lines:
                # Check for section headers
                section_match = re.match(r'^#{1,3}\s*(\d+\.(?:\d+)?)\s+([A-Z][a-zA-Z\s]+)$', line) or \
                               re.match(r'^(\d+\.(?:\d+)?)\s+([A-Z][a-zA-Z\s]+)$', line)
                
                if section_match:
                    # Save previous section
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
                    if current_section:
                        section_content.append(line)
        
        # Add the last section
        if current_section and section_content:
            structured_content["sections"].append({
                "section_id": current_section,
                "content": "\n".join(section_content)
            })
        
        # Combine all content
        structured_content["full_text"] = "\n\n".join(full_text)
        structured_content["full_markdown"] = "\n\n".join(full_markdown)
        
        return structured_content

    def save_to_json(self, data: Dict[str, Any], output_path: str) -> None:
        """Save the structured content to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved JSON output to {output_path}")
        except Exception as e:
            print(f"Error saving JSON output: {str(e)}")
            raise

    def save_to_markdown(self, data: Dict[str, Any], output_path: str) -> None:
        """Save the full markdown content to a .md file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Add title
                f.write(f"# {data['filename']}\n\n")
                f.write(f"**Pages:** {data['num_pages']}\n\n")
                f.write("---\n\n")
                f.write(data["full_markdown"])
            print(f"Saved Markdown output to {output_path}")
        except Exception as e:
            print(f"Error saving Markdown output: {str(e)}")
            raise

    def save_to_text(self, data: Dict[str, Any], output_path: str) -> None:
        """Save as plain text for PaperQA2 compatibility."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(data["full_text"])
            print(f"Saved text output to {output_path}")
        except Exception as e:
            print(f"Error saving text output: {str(e)}")
            raise
