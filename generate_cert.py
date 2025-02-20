from OpenSSL import crypto
import os

def generate_self_signed_cert():
    # 生成密钥
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)

    # 创建证书
    cert = crypto.X509()
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # 有效期一年
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, 'sha256')

    # 保存证书和私钥
    cert_path = os.path.join(os.path.dirname(__file__), 'cert.pem')
    key_path = os.path.join(os.path.dirname(__file__), 'key.pem')

    with open(cert_path, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open(key_path, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

    print(f"证书已生成：\n证书文件：{cert_path}\n密钥文件：{key_path}")

if __name__ == "__main__":
    generate_self_signed_cert()
