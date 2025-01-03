import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

SIZE = 5
OUTPUT_FOLDER = "texts"

load_dotenv()

IP = os.getenv("ELASTIC_IP")
USER = os.getenv("ELASTIC_USER")
PASSWORD = os.getenv("ELASTIC_PASSWORD")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

es = Elasticsearch(
    [IP],
    http_auth=(USER, PASSWORD),
    verify_certs=False,
)

if es.ping():
    print("Connected to Elasticsearch")
else:
    print("Could not connect to Elasticsearch")

query = {"_source": {"excludes": ["embeddings"]}}
data = es.search(index="texts", size=SIZE, body=query, scroll="2m")

scroll_id = data["_scroll_id"]


def extractText(textArr):
    out = ""
    for part in textArr:
        out += part["part"]
    return out


def processResult(res):
    hits = res["hits"]["hits"]
    for hit in hits:
        source = hit["_source"]

        id = hit["_id"]
        title = source["title"]
        author = source["author"]
        textArr = source["text"]
        print(f"{title}")
        textStr = extractText(textArr)
        with open(os.path.join(OUTPUT_FOLDER, id), "w") as file:
            file.write(f"{title}\n{author}\n{textStr}")


processResult(data)
docs = data["hits"]["hits"]
while docs:
    print(f"docs: {len(docs)}")
    data = es.scroll(scroll_id=scroll_id, scroll="2m")
    scroll_id = data["_scroll_id"]
    docs = data["hits"]["hits"]
    processResult(data)
