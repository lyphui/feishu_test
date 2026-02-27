import requests
import json
from datetime import datetime
import os
import time

# -------------------------- 配置项（必须改这2处） --------------------------
TOKEN_FILE_PATH = "/Users/daisy/PycharmProjects/feishu_test/authorize/feishu_key/feishu_token.txt"  # token保存的txt文件路径
TARGET_CHAT_ID = "oc_23477216df0db3f480e8c4585c09a54a"  # 替换成你的群ID（必填！从飞书群设置里复制）


# ---------------------------------------------------------------------------

# 1. 从txt文件读取user_access_token
def read_token_from_file(file_path):
    """读取txt文件中的token（文件内仅需保存token字符串，无其他内容）"""
    if not os.path.exists(file_path):
        print(f"❌ 未找到token文件：{file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            token = f.read().strip()  # 去除空格/换行
        if not token:
            print("❌ token文件为空，请检查文件内容")
            return None
        print(f"✅ 成功读取token：{token[:10]}...（已隐藏后半段）")
        return token
    except Exception as e:
        print(f"❌ 读取token失败：{e}")
        return None


# 2. 获取指定群的消息并筛选文档类消息（彻底跳过群列表）
def get_chat_messages_with_docs(token):
    """直接获取指定群的消息，筛选出文档/文件类消息"""
    if not TARGET_CHAT_ID:
        print("❌ 请先配置 TARGET_CHAT_ID（填写你的群ID）")
        return []

    headers = {"Authorization": f"Bearer {token}"}
    all_doc_messages = []
    page_token = ""
    doc_count = 0
    chat_name = f"目标群（ID：{TARGET_CHAT_ID}）"

    print(f"\n🔍 正在获取「{chat_name}」的文档消息...")

    # 分页获取群消息（避免接口限流，每页50条）
    while True:
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        params = {
            "container_id_type": "chat",
            "container_id": TARGET_CHAT_ID,
            "page_size": 50,
            "page_token": page_token
        }

        try:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()  # 捕获HTTP错误
            result = resp.json()

            if result["code"] != 0:
                print(f"❌ 获取群消息失败：{result['msg']}（错误码：{result['code']}）")
                break

            messages = result["data"].get("items", [])
            if not messages:
                break

            # 筛选文档/文件类消息（doc/sheet/slide/file）
            for msg in messages:
                msg_type = msg.get("msg_type", "")
                if msg_type in ["doc", "sheet", "slide", "file"]:
                    # 兼容不同消息类型的字段，避免KeyError
                    msg_content = msg.get(msg_type, {})
                    doc_info = {
                        "所属群": chat_name,
                        "群ID": TARGET_CHAT_ID,
                        "消息ID": msg.get("message_id", "未知"),
                        "发送人ID": msg.get("sender", {}).get("id", {}).get("user_id", "未知"),
                        "发送时间": datetime.fromtimestamp(int(msg.get("create_time", 0))).strftime(
                            "%Y-%m-%d %H:%M:%S"),
                        "消息类型": msg_type,
                        "文档名称": msg_content.get("name", "未知"),
                        "文档ID": msg_content.get("file_token", "未知"),
                        "文档链接": msg_content.get("url", "未知")
                    }
                    all_doc_messages.append(doc_info)
                    doc_count += 1

            # 获取下一页token，无则终止
            page_token = result["data"].get("page_token", "")
            if not page_token:
                break

            # 防接口限流，每次请求间隔0.3秒
            time.sleep(0.3)

        except requests.exceptions.RequestException as e:
            print(f"❌ 网络请求失败：{e}")
            break
        except KeyError as e:
            print(f"❌ 消息字段缺失：{e}，跳过该消息")
            continue

    print(f"✅ 「{chat_name}」共获取到 {doc_count} 条文档消息")
    return all_doc_messages


# 3. 主函数：整合逻辑
def main():
    # 1. 读取token
    token = read_token_from_file(TOKEN_FILE_PATH)
    if not token:
        return

    # 2. 获取群消息中的文档
    doc_messages = get_chat_messages_with_docs(token)

    # 3. 输出结果
    print("\n==================== 群消息中的文档汇总 ====================")
    if doc_messages:
        # 终端格式化打印
        for i, doc in enumerate(doc_messages, 1):
            print(f"\n{i}. 文档名称：{doc['文档名称']}")
            print(f"   发送时间：{doc['发送时间']}")
            print(f"   文档类型：{doc['文档类型']}")
            print(f"   文档链接：{doc['文档链接']}")

        # 保存到JSON文件（方便后续处理）
        with open("chat_documents_list.json", "w", encoding="utf-8") as f:
            json.dump(doc_messages, f, ensure_ascii=False, indent=2)
        print(f"\n📄 文档消息已保存到 chat_documents_list.json（共{len(doc_messages)}条）")
    else:
        print("❌ 未获取到任何群文档消息（检查：1.群ID是否正确 2.机器人能力是否开启 3.token是否有效 4.权限是否开通）")


if __name__ == "__main__":
    main()

    b'{"code":230027,"msg":"Lack of necessary permissions, ext=need scope: im:message.group_msg:get_as_user","error":{"log_id":"20260226223708A046B2912E7A2158CAE2","troubleshooter":"\xe6\x8e\x92\xe6\x9f\xa5\xe5\xbb\xba\xe8\xae\xae\xe6\x9f\xa5\xe7\x9c\x8b(Troubleshooting suggestions): https://open.feishu.cn/search?from=openapi&log_id=20260226223708A046B2912E7A2158CAE2&code=230027&method_id=6936075528891187228"}}'