import requests
import json
import time
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class PerplexityAPI:
    def __init__(self, api_key, group_id):
        """
        初始化Perplexity API客户端

        Args:
            api_key (str): API密钥，格式为 pplx-xxxxxxx
            group_id (str): API Group ID
        """
        self.api_key = api_key
        self.group_id = group_id
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "x-api-group": group_id  # 使用API Group ID
        }

    def chat(self, model, messages, max_tokens=None, temperature=0.2, top_p=0.9):
        """
        调用Perplexity API进行对话

        Args:
            model (str): 模型名称，可选：
                - sonar-reasoning-pro (每次调用最多搜索3次，最大输出8000 tokens) [[3]]
                - sonar-reasoning (每次调用只搜索1次，最大输出4000 tokens) [[3]]
                - sonar-deep-research (每次调用会搜索很多次，适合深度研究) [[3]]
                - sonar-medium-pro (God mode API，推理能力强) [[5]]
            messages (list): 对话消息列表，格式为 [{"role": "user", "content": "问题"}]
            max_tokens (int, optional): 最大输出token数
            temperature (float): 温度参数，控制随机性
            top_p (float): 采样概率阈值

        Returns:
            dict: API响应
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            # "search_recency_filter" : "day",
            # "search_after_date_filter": "12/11/2025",
            # "search_before_date_filter": "12/11/2025",
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"错误响应: {e.response.text}")
            return None

    def sonar_deep_research(self, query, timeout=300):
        """
        调用Sonar Deep Research模型进行深度研究
        该模型能够进行详尽的搜索，分析数百个来源，生成专家级见解和详细报告 [[8]]

        Args:
            query (str): 研究问题
            timeout (int): 等待响应的超时时间（秒）

        Returns:
            dict: 研究结果
        """
        messages = [{"role": "user", "content": query}]

        # 深度研究可能需要更长时间
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "sonar-deep-research",
                    "messages": messages,
                    # "temperature": 0.1,
                    # "top_p": 0.95,
            # "search_recency_filter" : "day",
            # "search_after_date_filter": "12/11/2025",
            # "search_before_date_filter": "12/11/2025",
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"深度研究API请求失败: {e}")
            return None

    def sonar_reasoning_pro(self, query):
        """
        调用Sonar Reasoning Pro模型进行高级推理
        该模型专为长链思维设计，内置网络搜索能力 [[4]]

        Args:
            query (str): 需要推理的问题

        Returns:
            dict: 推理结果
        """
        messages = [{"role": "user", "content": query}]
        return self.chat(
            model="sonar-reasoning-pro",
            messages=messages,
            max_tokens=8000,
            temperature=0.3,
        )

    def god_mode(self, query):
        """
        调用God Mode API (sonar-medium-pro)
        这是Perplexity的高级推理模型 [[5]]

        Args:
            query (str): 问题

        Returns:
            dict: 响应结果
        """
        messages = [{"role": "user", "content": query}]
        return self.chat(
            model="sonar-medium-pro",
            messages=messages,
            max_tokens=4096,
            temperature=0.7
        )

# 使用示例
if __name__ == "__main__":
    API_KEY  = os.getenv("PPLX_API_KEY")
    GROUP_ID = os.getenv("PPLX_GROUP_ID")

    # 初始化API客户端
    pplx = PerplexityAPI(API_KEY, GROUP_ID)

    # # ===== 示例1: 使用Sonar Reasoning Pro =====
    # print("=== Sonar Reasoning Pro 示例 ===")
    #
    # with open("output/sonar_reasoning_pro.txt", "r", encoding="utf-8") as f:
    #     previous_contents = f.read()
    topic = "science"
    # reasoning_result = pplx.sonar_reasoning_pro(
    #      f"List domestic and foreign news about {topic} in detail, and each piece of news should be marked with the release-time information."
    # )
    #
    # if reasoning_result:
    #     print("模型响应:")
    #     print(json.dumps(reasoning_result, indent=2, ensure_ascii=False))
    #     # 提取回答内容
    #     if 'choices' in reasoning_result and reasoning_result['choices']:
    #         content = reasoning_result['choices'][0]['message']['content']
    #         print("\n回答内容:")
    #         print(content)
    #         ## 把content写入txt中，放入文件夹 "output\sonar_reasoning_pro.txt"
    #         with open("output/sonar_reasoning_pro.txt", "a", encoding="utf-8") as f:
    #             f.write(content.split('</think>')[-1].strip())
    #             f.writelines("\n" + "=" * 50 + "\n\n\n")
    #
    #
    # print("\n" + "=" * 50 + "\n")

    # ===== 示例2: 使用Sonar Deep Research =====
    print("=== Sonar Deep Research 示例 ===")

    # with open("output/deep_research_result.txt", "r", encoding="utf-8") as f:
    #     previous_contents = f.read()

    deep_research_result = pplx.sonar_deep_research(
         # f"请搜索2025年12月8日关于计算机科学的最新进展，生成一份3000字的深度新闻分析，包括具体的时间（越具体越好，最好把新闻时间精确到日期）、技术细节和未来趋势。\n以下是我已知的信息，请不要重复搜索与生成： {previous_contents}.\n\n\n请生成3000字的详细报告，避免重复已知的信息"
        # f"List domestic and foreign news about {topic}, and each piece of news should be marked with the release-time information. "
        f"Please generate a technical report on the applications of artificial intelligence in the medical field, with a maximum length of 2,000 words."
    )

    if deep_research_result:
        print("深度研究结果:")
        print(json.dumps(deep_research_result, indent=2, ensure_ascii=False))

        if 'choices' in deep_research_result and deep_research_result['choices']:
            content = deep_research_result['choices'][0]['message']['content']
            print("\n回答内容:")
            print(content)
            ## 把content写入txt中，放入文件夹 "output\sonar_reasoning_pro.txt"
            with open("output/deep_research_result1.txt", "a", encoding="utf-8") as f:
                f.write(content.split('</think>')[-1].strip())
                f.writelines('\n====================\n')

    print("\n" + "=" * 50 + "\n")

    # # ===== 示例3: 使用God Mode API =====
    # print("=== God Mode API (sonar-medium-pro) 示例 ===")
    # god_mode_result = pplx.god_mode(
    #     "作为AI专家，详细解释transformer架构的工作原理及其在现代AI系统中的重要性"
    # )
    #
    # if god_mode_result:
    #     print("God Mode响应:")
    #     print(json.dumps(god_mode_result, indent=2, ensure_ascii=False))