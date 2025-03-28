
import easyocr

reader = easyocr.Reader(['en'])

result = reader.readtext('images.jpeg')

for detection in result:
    print(f"Detected text: {detection[1]} with confidence: {detection[2]}")
