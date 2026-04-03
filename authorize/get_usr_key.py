"""
飞书授权脚本
运行后自动打开浏览器 → 点击授权 → 从地址栏复制code粘贴 → 自动换取token保存
"""
import requests
import webbrowser
from urllib.parse import quote
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ======================== 配置项 ========================
APP_ID       = os.getenv("FEISHU_APP_ID")
APP_SECRET   = os.getenv("FEISHU_APP_SECRET")
REDIRECT_URI = "https://localhost:8080/callback"
TOKEN_FILE   = os.getenv("TOKEN_FILE")

# 所有需要的权限
SCOPES = [
    "im:message:readonly",
    "im:chat:readonly",
    "drive:file:readonly",
    "wiki:wiki:readonly",
    "docx:document:readonly",
    "bitable:app:readonly",
    "docs:document.content:read",
    "sheets:spreadsheet:readonly",
]
# =======================================================

AUTH_URL = (
    f"https://accounts.feishu.cn/open-apis/authen/v1/authorize"
    f"?app_id={APP_ID}"
    f"&redirect_uri={quote(REDIRECT_URI, safe='')}"
    f"&response_type=code"
    f"&scope={quote(' '.join(SCOPES), safe='')}"
    f"&state=auto_refresh"
)


def exchange_token(code):
    """用 code 换取 user_access_token"""
    url = "https://open.feishu.cn/open-apis/authen/v1/access_token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "app_id": APP_ID,
        "app_secret": APP_SECRET,
        "redirect_uri": REDIRECT_URI,
    }
    resp = requests.post(url, json=data)
    result = resp.json()

    if result.get("code") == 0:
        token_data = result["data"]
        access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 7200)
        refresh_token = token_data.get("refresh_token", "")

        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(access_token)

        refresh_file = TOKEN_FILE.replace(".txt", "_refresh.txt")
        if refresh_token:
            with open(refresh_file, "w", encoding="utf-8") as f:
                f.write(refresh_token)

        print(f"✅ token 获取成功！")
        print(f"   access_token : {access_token[:15]}...")
        print(f"   有效期       : {expires_in} 秒（约 {expires_in//3600} 小时）")
        print(f"   已保存到     : {TOKEN_FILE}")
        return access_token
    else:
        print(f"❌ 换取 token 失败：{result.get('msg')}（code={result.get('code')}）")
        return None


def refresh_with_refresh_token():
    """用 refresh_token 静默刷新（token过期时使用，无需重新授权）"""
    refresh_file = TOKEN_FILE.replace(".txt", "_refresh.txt")
    try:
        with open(refresh_file, "r", encoding="utf-8") as f:
            refresh_token = f.read().strip()
    except FileNotFoundError:
        print("❌ 未找到 refresh_token，请先完整授权一次")
        return None

    url = "https://open.feishu.cn/open-apis/authen/v1/refresh_access_token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "app_id": APP_ID,
        "app_secret": APP_SECRET,
    }
    resp = requests.post(url, json=data)
    result = resp.json()

    if result.get("code") == 0:
        token_data = result["data"]
        new_token = token_data["access_token"]
        new_refresh = token_data.get("refresh_token", refresh_token)

        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(new_token)
        with open(TOKEN_FILE.replace(".txt", "_refresh.txt"), "w", encoding="utf-8") as f:
            f.write(new_refresh)

        print(f"✅ token 静默刷新成功：{new_token[:15]}...")
        return new_token
    else:
        print(f"❌ 静默刷新失败：{result.get('msg')}，需要重新完整授权")
        return None


def full_auth_flow():
    """完整授权：打开浏览器 → 手动复制code → 自动换token"""
    print("\n🌐 正在打开浏览器授权页面...")
    print(f"   如未自动打开，请手动复制以下链接到浏览器：\n")
    print(f"   {AUTH_URL}\n")
    webbrowser.open(AUTH_URL)

    print("授权完成后，浏览器跳转到类似：")
    print("   https://localhost:8080/callback?code=xxxxxxxx&state=auto_refresh")
    print()
    print("⚠️  浏览器会显示'无法访问'，这是正常的，不用管。")
    print("   从地址栏找到 code= 后面的值（到 & 符号之前），复制粘贴到下方：\n")

    code = input("请粘贴 code：").strip()
    if not code:
        print("❌ code 不能为空")
        return None

    print("\n✅ 正在换取 token...")
    return exchange_token(code)


def main():
    print("=" * 50)
    print("  飞书 user_access_token 获取工具")
    print("=" * 50)
    print("\n请选择操作：")
    print("  1. 完整授权（首次使用 / 新增权限后）")
    print("  2. 静默刷新（token过期时，无需重新授权）")
    print()

    choice = input("请输入 1 或 2：").strip()

    if choice == "1":
        full_auth_flow()
    elif choice == "2":
        result = refresh_with_refresh_token()
        if not result:
            print("\n自动刷新失败，切换到完整授权...")
            full_auth_flow()
    else:
        print("无效输入，执行完整授权...")
        full_auth_flow()


if __name__ == "__main__":
    main()
