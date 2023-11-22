import pyqrcode
from PIL import Image
from pyzbar.pyzbar import decode

# 读取文本文件内容
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# 将文本转换为QR码并保存为图片，然后打印图片
def generate_qr_code_and_print(text):
    qr = pyqrcode.create(text)

    qr.png("qr_code.png", scale=10)

    qr_image = Image.open("qr_code.png")
    qr_image.show()

# 识别QR码图片并返回内容
def decode_qr_code(qr_code_file_path):
    qr_image = Image.open(qr_code_file_path)
    qr_code_data = decode(qr_image)
    if qr_code_data:
        return qr_code_data[0].data.decode('utf-8')
    else:
        return None

# 读取文本文件
text_file_path = "path/to/your/text_file.txt"
text_content = read_text_file(text_file_path)

# 生成QR码并保存为图片，并打印图片
generate_qr_code_and_print(text_content)

# 接下来的解码部分与之前的代码相同，识别QR码并打印内容
qr_code_file_path = "qr_code.png"
decoded_content = decode_qr_code(qr_code_file_path)
if decoded_content:
    print("QR Code Content:", decoded_content)
else:
    print("QR Code not detected or could not be decoded.")
