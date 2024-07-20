from flask import Flask, request
from langchain_community.llms import Ollama

app = Flask(__name__)

cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

@app.route("/ai", methods=["POST"])
def aiPost():
    print("POST /ai called")
    json_content = request.json
    query = json_content.get("query")
    # print(f"query: {query}")

    response_answer = cached_llm.invoke(query)
    return {"answer" : response_answer}

@app.route("/test", methods=["GET"])
def testGet():
    print("test called")

    return "test"

def start_app():
    app.run(host="0.0.0.0", port=5555, debug=True)


if __name__ == "__main__":
    start_app()