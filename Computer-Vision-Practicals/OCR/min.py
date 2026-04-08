from mineru import MinerU

# Initialize the client. No token is needed for "flash" mode.
client = MinerU()

# Path to your local PDF or a remote URL
pdf_path = r"C:\Users\HP\Desktop\Bacancy\AI_ML\Deep_Learning\Computer_Vision\computer-vision-practical-new\OCR\nutrition_ocr_package\test_images\20240319_202800_jpg.rf.1e90260ba7ad10a5bbbec1b25da3ee29_rot90.jpg"

# Extract content to Markdown
result = client.flash_extract(pdf_path,language="en")

# Print the extracted Markdown text
print("result",result.markdown)

# If you need extracted images, you can access them via:
# print(result.images)