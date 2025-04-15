import time
from typing import List, Dict
import pandas as pd
from zhipuai import ZhipuAI

# 初始化 client（请填写你的 API key）
client = ZhipuAI(api_key="3c87b4c80506418a8882bef064ec2199.BaNPFlZcQHfJOFyg")

# 核心测速函数（不依赖 stream_options）
def measure_glm_speed(client, model: str, messages: List[Dict[str, str]]) -> Dict:
    start_time = time.time()
    first_token_time = None
    content_all = ""

    # 发起流式请求（不包含 stream_options）
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning_piece = getattr(delta, 'reasoning_content', "")
        content_piece = getattr(delta, 'content', "")

        if first_token_time is None and (reasoning_piece or content_piece):
            first_token_time = time.time() - start_time

        print(reasoning_piece + content_piece, end="")
        content_all += reasoning_piece + content_piece

    end_time = time.time()

    # 粗略估算 token 数（1 token ≈ 1.5 个字符）
    estimated_tokens = int(len(content_all) / 1.5)
    generation_time = end_time - (start_time + first_token_time)
    avg_token_speed = estimated_tokens / generation_time if generation_time > 0 else 0

    return {
        "首 token 延迟（秒）": round(first_token_time, 3),
        "生成耗时（秒）": round(generation_time, 3),
        "估算 tokens 数": estimated_tokens,
        "总耗时（秒）": round(end_time - start_time, 3),
        "平均生成速度 (token/s)": round(avg_token_speed, 2)
    }

# 多轮测速函数
def batch_measure_speed(client, model: str, prompts: List[str]) -> pd.DataFrame:
    results = []

    for idx, prompt in enumerate(prompts):
        print(f"\n🧪 正在测量第 {idx + 1} 个问题：{prompt}\n")
        messages = [{"role": "user", "content": prompt}]
        result = measure_glm_speed(client, model, messages)
        result["问题编号"] = f"Q{idx + 1}"
        result["问题内容"] = prompt
        results.append(result)

    df = pd.DataFrame(results)
    cols = ["问题编号", "问题内容", "首 token 延迟（秒）", "生成耗时（秒）",
            "估算 tokens 数", "总耗时（秒）", "平均生成速度 (token/s)"]
    df = df[cols]
    return df

# 程序入口（主函数）
if __name__ == "__main__":
    prompts = [
        "袋子里有5个红球和3个蓝球，随机抽两个球，至少一个红球的概率是多少？",
        "请写一篇议论文，论述学习使用AI的重要性。",
        "9.11和9.9两个数字，谁更大？"
    ]

    df = batch_measure_speed(client, model="glm-z1-airx", prompts=prompts)
    print("\n📊 测试结果如下：\n")
    print(df.to_markdown(index=False))

    df.to_csv("glm_speed_results.csv", index=False)
    print("\n✅ 已保存结果为 glm_speed_results.csv")
