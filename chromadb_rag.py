import lazyllm
from lazyllm import pipeline, bind, Document, Retriever, OnlineEmbeddingModule, OnlineChatModule, WebModule, SentenceSplitter


prompt = """
You will play the role of an AI Q&A assistant and complete a dialogue task.
In this task, you need to provide your answer based on the given context and question.
"""
# embed_model = OnlineEmbeddingModule(source="qwen", api_key=api_key)

embed_model = OnlineEmbeddingModule(source="qwen",)
store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {
            'uri': './db/segment_store_1.db',
        },
    },
    'vector_store': {
        'type': 'chromadb',
        'kwargs': {
            'dir': './db/chromadb_1',
            'index_kwargs': {
                'hnsw': {
                    'space': 'cosine',
                    'ef_construction': 100,
                }
            }
        },
    },
}
doc = Document(
    dataset_path="./docs",
    embed=lazyllm.OnlineEmbeddingModule(source='qwen',),
    manager=False,
    store_conf=store_conf
)
doc.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

with pipeline() as ppl:
    ppl.retriever = Retriever(doc, group_name='sentences', similarity="cosine", topk=6, output_format='content')
    ppl.formatter = (lambda context, query: dict(context_str=str(context), query=query)) | bind(query=ppl.input)
    ppl.llm = OnlineChatModule(source='qwen', stream=True).prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))

w = WebModule(m=ppl, stream=True)
w.start().wait()
