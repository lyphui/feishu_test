from flask import Flask, request
from flask_sslify import SSLify
import os
import OpenSSL

app = Flask(__name__)
# 强制启用HTTPS
sslify = SSLify(app)

# 生成自签名SSL证书（首次运行自动生成，后续复用）
def generate_ssl_cert():
    cert_file = "authorize/feishu_key/localhost.crt"
    key_file = "authorize/feishu_key/localhost.key"
    # 如果证书已存在，直接返回
    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file
    
    # 生成私钥
    key = OpenSSL.crypto.PKey()
    key.generate_key(OpenSSL.crypto.TYPE_RSA, 2048)
    # 生成证书请求
    req = OpenSSL.crypto.X509Req()
    req.get_subject().CN = "localhost"
    req.set_pubkey(key)
    req.sign(key, "sha256")
    # 生成自签名证书
    cert = OpenSSL.crypto.X509()
    cert.set_subject(req.get_subject())
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # 有效期1年
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(req.get_pubkey())
    cert.sign(key, "sha256")
    
    # 保存证书和私钥到本地
    with open(cert_file, "wb") as f:
        f.write(OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_PEM, cert))
    with open(key_file, "wb") as f:
        f.write(OpenSSL.crypto.dump_privatekey(OpenSSL.crypto.FILETYPE_PEM, key))
    
    return cert_file, key_file

# 回调接口：接收飞书的code
@app.route('/callback')
def callback():
    # 从URL参数中获取code和state
    code = request.args.get('code')
    state = request.args.get('state')
    
    if not code:
        return f"授权失败！错误信息：{request.args}", 400
    
    # 打印code（你可以复制这个code去换token）
    print("\n===== 授权码CODE已获取 =====")
    print(f"code: {code}")
    print(f"state: {state}")
    print("===========================\n")
    
    # 页面展示结果，方便查看
    return f"""
    <h1>授权成功！</h1>
    <p>你的code：<b>{code}</b></p>
    <p>复制这个code，即可去换取user_access_token</p>
    """

if __name__ == '__main__':
    # 生成SSL证书
    cert, key = generate_ssl_cert()
    print(f"SSL证书已生成：{cert}, {key}")
    # 启动HTTPS服务，监听8080端口
    app.run(
        host='0.0.0.0',  # 允许外部访问（飞书回调需要）
        port=8080,
        ssl_context=(cert, key),  # 启用HTTPS
        debug=True  # 调试模式，方便看日志
    )


## appid:  cli_a914760ada389cda


'''
https://open.feishu.cn/open-apis/authen/v1/authorize?app_id=cli_a914760ada389cda&redirect_uri=https%3A%2F%2Flocalhost%3A8080%2Fcallback&scope=im:message:readonly im:chat:readonly drive:file:readonly&response_type=code&state=123456

https://open.feishu.cn/open-apis/authen/v1/index?
  app_id=YOUR_APP_ID
  &redirect_uri=YOUR_REDIRECT_URI
  &scope=im:message:readonly im:chat:readonly   ← 这里必须包含新权限
  &state=xxx
'''