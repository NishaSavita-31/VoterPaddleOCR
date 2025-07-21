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
from pydantic import BaseModel, Field, field_validator
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
    address: Optional[str] = Field(None)
    # father_name: Optional[str] = Field(None)

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
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

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


class PaddleVoterOCR:

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
        logger.info("PaddleVoterOCR initialized", langs=langs, use_gpu=use_gpu, primary_lang=primary_lang)

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

    def extract_address(self, text: str) -> Optional[str]:
        """Extract address and pincode from text"""
        address_patterns = [
            r'(?:Address|पता)\s*[:\-]?\s*([\s\S]+?\d{6})',     # Address block ending with PIN
            r'(?:W/O|D/O|S/O|C/O)\s+[\s\S]+?\d{6}',             # Relational format ending with PIN
            r'आधा\s*र[^\s]*\s*(.+\d{6})'                        # Hindi Aadhaar + address ending with PIN
        ]

        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                address = match.group(0).strip()
                address = re.sub(r'\s+', ' ', address)
                address = address.replace('\n', ' ')

                # Extract 6-digit PIN code
                pincode_match = re.search(r'\b\d{6}\b', address)
                pincode = pincode_match.group(0) if pincode_match else None

                return {
                    "address": address,
                    "pincode": pincode
                }

        # Note: Address is typically on the back of the card
        # This is the front side, so address might not be present
        return None


    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Main processing function"""
        logger.info("Processing image with PaddleOCR", path=image_path)

        try:
            # Extract text directly from image path (PaddleOCR can handle file paths)
            text, structured_results = self.extract_text(image_path)

            # Extract fields
            voter_data = {
                "address": self.extract_address(text),
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
                    "data": {k: v for k, v in voter_data.items() if v},
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
    ocr = PaddleVoterOCR(use_gpu=False, langs=['en', 'hi'])

    # Process sample image
    result = ocr.process_image('/content/VoterBack.png')
    # Pretty print result
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()