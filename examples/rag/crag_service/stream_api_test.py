import requests
import json

url = "http://localhost:8000/v1/chat/completions"

querys = ["北京天气如何?", "失蜡铸造原理?"]

for query in querys:
    payload = {
        "model": "crag-agent",
        "messages": [{"role": "user", "content": query}],
        "stream": True
    }

    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                print(line)
                # if line.startswith('data: '):
                #     data = line[6:]
                #     if data != '[DONE]':
                #         chunk = json.loads(data)
                #         content = chunk['choices'][0]['delta'].get('content', '')
                #         if content:
                #             print(content, end='', flush=True)



