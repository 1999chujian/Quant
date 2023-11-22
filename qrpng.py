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

##
from PIL import Image
from pyzbar.pyzbar import decode

# 定义要解码的QR码图片路径
qr_code_file_path = "C:\Users\13721\Documents\qrcode.png"

# 读取QR码图片并解码
qr_image = Image.open(qr_code_file_path)
qr_code_data = decode(qr_image)

# 检查是否成功解码并打印内容
if qr_code_data:
    decoded_content = qr_code_data[0].data.decode('utf-8')
    print("QR Code Content:", decoded_content)
else:
    print("QR Code not detected or could not be decoded.")

###
def convert_to_lowercase(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    lowercase_content = content.lower()

    with open(output_file, 'w') as file:
        file.write(lowercase_content)

# 定义输入和输出文件路径
input_file_path = "/path/to/input_file.txt"
output_file_path = "/path/to/output_file.txt"

# 调用函数进行转换
convert_to_lowercase(input_file_path, output_file_path)

###
def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        content1 = f1.read()
        content2 = f2.read()

    # 逐个字符比较内容
    for char_num, (char1, char2) in enumerate(zip(content1, content2)):
        if char1 != char2:
            print(f"位置 {char_num+1}:")
            print(f"文件1: {char1}")
            print(f"文件2: {char2}")
            print()

# 定义要比较的两个文件路径
file1_path = "/path/to/file1.txt"
file2_path = "/path/to/file2.txt"

# 调用函数进行比较
compare_files(file1_path, file2_path)

ssh -rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQClMkMVVcNlgkRl7at3fNR+d7I8pHBI0MLo+A9e81emqbsT84TC3GkGB04MtJSOGq9PF830nDTkk043/VhrPHDJKUW5I/E/6KRyXWztm8d25X6YIMwP1HrQz/wEJnnuYBI+PpZTwxgMcOMg3FKj8C/H2xYndQGqfFmGB8c/GkyCmhlS/oVgbkqHTCInzAHZU1HB+F9DNRuGeMO0PDIJHyygX6C+x4SYNBNmkxmVDRy0GhmRY76RqMVjTodcZz2GG9usMKJwt7CpImmhYHvztkCZnmeEmOGLeP8OuWKXzUKnWoJjaPTd0iUPsK9OuuAXYzT5mXro5xqyhTim8f9ikH6l16NnGN7vw0blT5OS/nAeImndCR3IMdrVbfUI2SBFal2nJ3iWGoU5XushrXZoCauYvxuux2+DIjqIY/+l/yc0zbVjNrSHB8Zi8DNvk0UTaXpuTsSFg7hUyw78WfPNdwe+BeCIK78NTSWFJ1DQPoUSnBWAUwuE3Tz/XvrljVWYLwX5gVARbXlYm7g5rse3c9wdb/WuRLGdbtnk638b5Ky3yvUCOCuY3chq9gN9nxDV4pKWZFy6zcd1lQlwTkQOugda+waBWidvoAsRpl03MpGJue50y9VbomXnj9Po3oanb0oJwkPWHAQhVPsuWmVjnRnakzrurW5w18jcKS+lC7w== chujian_hit@163.com
