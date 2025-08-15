import pytesseract
from PIL import Image

# If Tesseract isn't in PATH, uncomment and set manually:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load sample image (replace with actual test image path)
img = Image.new('RGB', (200, 100), color='white')
import PIL.ImageDraw as ImageDraw
draw = ImageDraw.Draw(img)
draw.text((10, 40), "Hello OCR!", fill='black')

# Run OCR
text = pytesseract.image_to_string(img)
print("Extracted text:", text.strip())
