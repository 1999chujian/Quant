import qrcode

# 定义要转换为二维码的 Python 文件路径
python_file_path = "/path/to/your_python_file.py"

# 创建二维码对象
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

# 将 Python 文件路径添加到二维码中
qr.add_data(python_file_path)

# 根据数据生成二维码
qr.make(fit=True)

# 创建并保存二维码图像
qr_image = qr.make_image(fill_color="black", back_color="white")
qr_image.save("/path/to/save_qr_code.png")



import base64
import qrcode

# 读取 Python 文件
file_path = 'path/to/your_file.py'
with open(file_path, 'rb') as file:
    file_content = file.read()

# 将文件内容编码为 Base64
encoded_content = base64.b64encode(file_content).decode('utf-8')

# 创建二维码对象
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

# 添加 Base64 编码的内容到二维码
qr.add_data(encoded_content)
qr.make(fit=True)

# 生成二维码图像
qr_image = qr.make_image(fill_color="black", back_color="white")

# 保存二维码图像
qr_image.save('path/to/qrcode.png')
