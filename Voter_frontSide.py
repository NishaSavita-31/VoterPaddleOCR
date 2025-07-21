import os
import re
import json
import logging
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from pathlib import Path
import tempfile
import paddle
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class VoterData(BaseModel):
   document_number: Optional[str]=Field(None, pattern=r'\b[A-Z]{3}\d{7}\b')
   name:Optional[str] =Field(None,min_length=2,max_length=100)
  

   @field_validator('document_number', mode='before')
   @classmethod
   def format_voter_number(cls, value: Any, info: ValidationInfo) -> str:
        if not isinstance(value, str):
            return value  # Return original value if not a string

        # Ensure the input is in the format: letters + numbers (e.g., "ABC1234567")
        if len(value) < 10 or not value[:3].isalpha() or not value[3:10].isdigit():
            raise ValueError("Voter number must be in the format 'XXX1234567' (3 letters followed by 7 numbers).")

        # Extract the letters and digits from the voter number
        letters = value[:3]
        digits = value[3:10]

        # Format the digits (group into chunks of 4 and 3)
        formatted_digits = f"{digits[:4]} {digits[4:]}"

        # Return the formatted voter number with letters and formatted digits
        return f"{letters} {formatted_digits}"

class PaddleImagePreprocessor:

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        return image

    @staticmethod
    def auto_rotate(image: np.ndarray) -> np.ndarray:
        try:
            # Convert to grayscale for rotation detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

            if lines is not None:
                # Calculate the most common angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta) - 90
                    angles.append(angle)

                # Get median angle
                median_angle = np.median(angles)

                # Rotate image if needed (threshold: 1 degree)
                if abs(median_angle) > 1:
                    center = (image.shape[1]//2, image.shape[0]//2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated
        except Exception as e:
            logger.warning("Auto-rotation failed", error=str(e))

        return image

    @staticmethod
    def enhance_quality(image: np.ndarray) -> np.ndarray:
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)

        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB) # Changed to BGR2LAB
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR) # Changed to LAB2BGR

        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    @staticmethod
    def preprocess(image_path: str) -> np.ndarray:
        """Complete preprocessing pipeline"""
        image = PaddleImagePreprocessor.load_image(image_path)
        image = PaddleImagePreprocessor.auto_rotate(image)
        image = PaddleImagePreprocessor.enhance_quality(image)
        return image


class PaddleAadhaarOCR:

    def __init__(self, use_gpu: bool = False, langs: List[str] = ['en', 'hi']):
        """Initialize PaddleOCR reader"""
        self.use_gpu = use_gpu
        self.langs = langs

        # Initialize PaddleOCR with language support
        # PaddleOCR supports: en (English), hi (Hindi)
        lang_codes = []
        if 'en' in langs:
            lang_codes.append('en')
        if 'hi' in langs:
            lang_codes.append('hi')

        # Use first language as primary
        primary_lang = lang_codes[0] if lang_codes else 'en'

        self.ocr = PaddleOCR(
            lang=primary_lang   # Primary language
        )

        self.preprocessor = PaddleImagePreprocessor()
        logger.info("PaddleAadhaarOCR initialized", langs=langs, use_gpu=use_gpu, primary_lang=primary_lang)

    def extract_text(self, image_input) -> Tuple[str, List[Dict]]:
        try:
            # Handle both file paths and numpy arrays
            if isinstance(image_input, str):
                # File path - PaddleOCR can handle this directly
                image_path = image_input
            else:
                # Numpy array - need to save temporarily or process directly
                image_path = image_input  # For now, assume it's always a path

            # Run OCR - use predict method for newer PaddleOCR
            results = self.ocr.predict(image_path)

            # Parse results - PaddleOCR v3 format
            full_text_parts = []
            structured_results = []

            # PaddleOCR returns a list with dict containing rec_texts and rec_scores
            if isinstance(results, list) and len(results) > 0:
                result_dict = results[0]

                # Extract recognized texts and scores
                rec_texts = result_dict.get('rec_texts', [])
                rec_scores = result_dict.get('rec_scores', [])
                rec_polys = result_dict.get('rec_polys', [])

                for i, text in enumerate(rec_texts):
                    if text and text.strip():  # Skip empty texts
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        bbox = rec_polys[i].tolist() if i < len(rec_polys) else []

                        full_text_parts.append(text)
                        structured_results.append({
                            "text": text,
                            "confidence": confidence,
                            "bbox": bbox
                        })

            full_text = " ".join(full_text_parts)

            return full_text, structured_results
        except Exception as e:
            logger.error("Text extraction failed", error=str(e))
            raise

    def extract_document_number(self,text:str) -> Optional[str]:
        cleaned_text = text.replace('@','0').replace('O','0')

        patterns = [
           r'\b[A-Z]{3}\d{7}\b'
        ]
        def match_voter_card_number(text):
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    print(f"Matched: {match.group()}")
                    return match.group()
            return None

        result = match_voter_card_number(cleaned_text)
        return result # Return the extracted document number

    def extract_name(self, text: str) -> Optional[str]:
         # Pattern to match 'NAME' or similar keywords followed by the actual name
        name_pattern = r'(?i)\b(NAME|Name\'S)\s+([A-Za-z\s]+)'

        # Match using the pattern
        def match_name(text):
            match = re.search(name_pattern, text)
            if match:
                print(f"Matched Name: {match.group(2)}")  # group(2) is the actual name
                return match.group(2).strip()
            return None

        # Call the function to get the result
        result = match_name(text)
        return result


    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Main processing function"""
        logger.info("Processing image with PaddleOCR", path=image_path)

        try:
            # Extract text directly from image path (PaddleOCR can handle file paths)
            text, structured_results = self.extract_text(image_path)

            # Extract fields
            voter_data = {

                "document_number": self. extract_document_number(text),
                "name": self.extract_name(text)
            }

            # Validate using Pydantic
            try:
                validated_data = VoterData(**voter_data)
                result = {
                    "success": True,
                    "data": validated_data.model_dump(exclude_none=True),
                    "raw_text": text,
                    "confidence_scores": [r["confidence"] for r in structured_results],
                    "ocr_engine": "PaddleOCR"
                }
            except Exception as e:
                # Return partial data if validation fails
                result = {
                    "success": False,
                    "data": {k: v for k, v in voter_data.items() if v}, # Changed aadhaar_data to voter_data
                    "raw_text": text,
                    "validation_errors": str(e),
                    "confidence_scores": [r["confidence"] for r in structured_results],
                    "ocr_engine": "PaddleOCR"
                }

            logger.info("Processing completed", success=result["success"], engine="PaddleOCR")
            return result

        except Exception as e:
            logger.error("Processing failed", error=str(e), engine="PaddleOCR")
            return {
                "success": False,
                "error": str(e),
                "data": {},
                "ocr_engine": "PaddleOCR"
            }


def main():
    """Main function for testing"""
    # Initialize OCR
    ocr = PaddleAadhaarOCR(use_gpu=False, langs=['en', 'hi'])

    # Process sample image
    result = ocr.process_image('/content/VoterfFront.png')

    # Pretty print result
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()