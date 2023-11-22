import qrcode
from PIL import Image
from pyzbar.pyzbar import decode

# 读取文本文件内容
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# 将文本转换为QR码并保存为图片
def generate_qr_code(text, qr_code_file_path):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)

    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image.save(qr_code_file_path)

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

# 生成QR码并保存为图片
qr_code_file_path = "path/to/save/qr_code.png"
generate_qr_code(text_content, qr_code_file_path)

# 识别QR码并打印内容
decoded_content = decode_qr_code(qr_code_file_path)
if decoded_content:
    print("QR Code Content:", decoded_content)
else:
    print("QR Code not detected or could not be decoded.")
