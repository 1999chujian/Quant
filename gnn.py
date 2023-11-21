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
