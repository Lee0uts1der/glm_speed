stream_options= dict(include_usage=True)
from zhipuai import ZhipuAI
import time

client = ZhipuAI(api_key="3c87b4c80506418a8882bef064ec2199.BaNPFlZcQHfJOFyg")

t0 = time.time()

response = client.chat.completions.create(
    model="glm-z1-airx",
    messages=[
        {"role": "user", "content": "写一篇议论文，论述学习AI是有必要的"}
    ],
    stream=True,
    stream_options={"include_usage": True}
)

t1 = None
t3 = None
total_tokens = 0
all_content = ""

for chunk in response:
    if not t1:
        t1 = time.time()
    print(chunk.choices[0].delta.content, end="")
    all_content += chunk.choices[0].delta.content

    # 使用详情出现在最后一个 chunk 中
    if hasattr(chunk, "usage") and chunk.usage:
        total_tokens = chunk.usage.completion_tokens

t3 = time.time()

# 输出统计信息
print("\n\n⏱️ 首 token 延迟（t1 - t0）: {:.2f} 秒".format(t1 - t0))
print("📝 文本生成耗时（t3 - t1）: {:.2f} 秒".format(t3 - t1))
print("🔢 总生成 token 数: {}".format(total_tokens))
if total_tokens > 0:
    print("🚀 平均生成速度: {:.2f} token/s".format(total_tokens / (t3 - t1)))
