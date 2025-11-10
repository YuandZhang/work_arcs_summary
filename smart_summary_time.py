import os
import glob
import shutil
import tempfile
from datetime import datetime
import gradio as gr
import lazyllm
from lazyllm import pipeline, bind, Document, Retriever, OnlineEmbeddingModule, OnlineChatModule, SentenceSplitter

class SmartWorkSummarySystem:
    def __init__(self, docs_path="./docs", db_dir="./db"):
        self.docs_path = docs_path
        self.db_dir = db_dir
        self.prompt = """
你是一个智能工作文件总结系统。
请根据用户指定的时间范围内修改的文件内容，提供准确、全面的工作总结。
【要求】
1. 对文件中的工作内容进行总结归纳；
2. 可按项目、任务或时间线组织；
3. 突出关键成果与数据；
4. 末尾可给出合理的改进或下一步计划；
5. 如检索结果为空，请返回“未找到可汇总的工作内容”。
【输出】
请以正式工作总结的格式输出，不超过 500 字。
"""
    
    def _get_files_in_time_range(self, start_time, end_time):
        """
        根据时间范围筛选文件
        :param start_time: 开始时间 (datetime对象)
        :param end_time: 结束时间 (datetime对象)
        :return: 符合时间范围的文件列表
        """
        # 获取所有PDF文件
        pdf_files = glob.glob(os.path.join(self.docs_path, "*.pdf"))
        filtered_files = []
        
        for file_path in pdf_files:
            # 获取文件最后修改时间
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            # 检查是否在指定时间范围内
            if start_time <= mod_time <= end_time:
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _create_temp_docs_folder(self, files_in_range):
        """
        创建临时文件夹并复制符合条件的文件
        :param files_in_range: 符合时间范围的文件列表
        :return: 临时文件夹路径
        """
        # 创建临时文件夹
        temp_dir = tempfile.mkdtemp(prefix="timefilter_docs_")
        
        # 复制文件到临时文件夹
        for file_path in files_in_range:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(temp_dir, filename)
            shutil.copy2(file_path, dest_path)
        
        return temp_dir
    
    def generate_summary(self, start_date_str, end_date_str):
        """
        根据时间范围生成工作总结
        :param start_date_str: 开始日期字符串 (格式: YYYY-MM-DD)
        :param end_date_str: 结束日期字符串 (格式: YYYY-MM-DD)
        :return: 工作总结
        """
        # 解析日期字符串
        start_time = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_time = datetime.strptime(end_date_str, "%Y-%m-%d")
        # 设置结束时间为当天的最后一刻
        end_time = end_time.replace(hour=23, minute=59, second=59)
        
        # 获取时间范围内的文件
        files_in_range = self._get_files_in_time_range(start_time, end_time)
        
        if not files_in_range:
            return "在指定时间范围内没有找到任何修改过的文件。"
        
        print(f"找到 {len(files_in_range)} 个文件在指定时间范围内:")
        for file in files_in_range:
            print(f"  - {os.path.basename(file)}")
        
        # 创建临时文件夹
        temp_docs_path = self._create_temp_docs_folder(files_in_range)
        
        try:
            # 创建临时数据库配置
            store_conf = {
                'segment_store': {
                    'type': 'map',
                    'kwargs': {
                        'uri': os.path.join(self.db_dir, 'segment_store_time.db'),
                    },
                },
                'vector_store': {
                    'type': 'chromadb',
                    'kwargs': {
                        'dir': os.path.join(self.db_dir, 'chromadb_time'),
                        'index_kwargs': {
                            'hnsw': {
                                'space': 'cosine',
                                'ef_construction': 100,
                            }
                        }
                    },
                },
            }
            
            # 创建文档对象，使用临时文件夹路径
            doc = Document(
                dataset_path=temp_docs_path,
                embed=lazyllm.OnlineEmbeddingModule(source='qwen'),
                manager=False,
                store_conf=store_conf
            )
            
            # 创建句子分割器
            doc.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
            
            # 构建pipeline
            with pipeline() as ppl:
                ppl.retriever = Retriever(doc, group_name='sentences', similarity="cosine", topk=6, output_format='content')
                ppl.formatter = (lambda context, query: dict(context_str=str(context), query=query)) | bind(query=ppl.input)
                ppl.llm = OnlineChatModule(source='qwen', stream=False).prompt(
                    lazyllm.ChatPrompter(self.prompt, extra_keys=["context_str"])
                )
            
            # 执行查询以生成总结
            query = f"请根据{start_date_str}至{end_date_str}期间修改的文件内容，生成一份工作总结报告。"
            result = ppl(query)
            
            return result
        finally:
            # 清理临时文件夹
            if os.path.exists(temp_docs_path):
                shutil.rmtree(temp_docs_path)

# 创建 Gradio 界面
def create_gradio_interface():
    # 初始化系统
    summary_system = SmartWorkSummarySystem()
    
    def generate_summary_wrapper(start_date, end_date):
        return summary_system.generate_summary(start_date, end_date)
    
    # 创建 Gradio 界面
    with gr.Blocks(title="智能工作总结系统") as demo:
        gr.Markdown("# 智能工作总结系统")
        gr.Markdown("选择时间范围以生成该时间段内修改文件的工作总结")
        
        with gr.Row():
            with gr.Column():
                start_date = gr.Textbox(label="开始日期 (YYYY-MM-DD)", value="2023-01-01")
                end_date = gr.Textbox(label="结束日期 (YYYY-MM-DD)", value="2023-12-31")
                btn = gr.Button("生成工作总结")
            
            with gr.Column():
                output = gr.Textbox(label="工作总结", lines=20, interactive=False)
        
        btn.click(fn=generate_summary_wrapper, inputs=[start_date, end_date], outputs=output)
    
    return demo

# 启动 Gradio 应用
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)