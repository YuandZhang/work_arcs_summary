# 基于 LazyLLM 的智能工作总结系统案例实践
## 目录
- 项目核心功能
- 整体代码架构
- 处理流程 RAG 技术解析
- ChromaDB 优化策略介绍
- 最终核心功能展示
## 项目核心功能
本项目旨在构建一个智能工作总结系统，能够根据用户指定的时间范围，自动筛选在该时间段内修改过的 PDF 文档，并基于这些文档的内容生成一份结构化的工作总结报告。系统通过结合 LazyLLM 框架和 ChromaDB 向量数据库，实现了从文档筛选、内容提取到智能生成的完整流程。

## 整体代码架构
项目的核心代码位于 smart_summary_time.py 文件中，主要包括以下几个部分：

1. SmartWorkSummarySystem 类 ：这是系统的核心类，负责管理文档路径、数据库配置以及整个工作总结的生成流程。
2. 文档处理模块 ：包括文件筛选、临时文件夹创建等功能，用于处理符合时间范围的文档。
3. 向量存储配置 ：使用 ChromaDB 作为向量存储，并配置了 HNSW 索引以优化检索性能。
4. RAG 流程构建 ：利用 LazyLLM 的 pipeline 构建从文档检索到大模型生成的完整流程。
5. Gradio 界面 ：提供一个简单的 Web 界面，让用户可以方便地输入时间范围并查看生成的总结。
## 处理流程 RAG 技术解析
系统的工作流程可以分为以下几个步骤：

1. 文档筛选 ：根据用户输入的时间范围，筛选出在该时间段内修改过的 PDF 文件。
   
   ```python
    def _get_files_in_time_range(self, 
    start_time, end_time):
        """
        根据时间范围筛选文件
        :param start_time: 开始时间 
        (datetime对象)
        :param end_time: 结束时间 (datetime
        对象)
        :return: 符合时间范围的文件列表
        """
        # 获取所有PDF文件
        pdf_files = glob.glob(os.path.join
        (self.docs_path, "*.pdf"))
        filtered_files = []
        
        for file_path in pdf_files:
            # 获取文件最后修改时间
            mod_time = datetime.
            fromtimestamp(os.path.getmtime
            (file_path))
            # 检查是否在指定时间范围内
            if start_time <= mod_time <= 
            end_time:
                filtered_files.append
                (file_path)
        
        return filtered_files
   ```
2. 内容提取 ：将筛选出的文件复制到临时文件夹，以便后续处理。
   
   ```python
    def _create_temp_docs_folder(self, 
    files_in_range):
        """
        创建临时文件夹并复制符合条件的文件
        :param files_in_range: 符合时间范围
        的文件列表
        :return: 临时文件夹路径
        """
        # 创建临时文件夹
        temp_dir = tempfile.mkdtemp
        (prefix="timefilter_docs_")
        
        # 复制文件到临时文件夹
        for file_path in files_in_range:
            filename = os.path.basename
            (file_path)
            dest_path = os.path.join
            (temp_dir, filename)
            shutil.copy2(file_path, 
            dest_path)
        
        return temp_dir
   ```
3. 向量存储 ：使用 LazyLLM 的 Document 类处理临时文件夹中的文档，并将其内容存储到 ChromaDB 向量数据库中。
   
   ```python
            # 创建文档对象，使用临时文件夹路径
            doc = Document(
                dataset_path=temp_docs_pat
                h,
                embed=lazyllm.
                OnlineEmbeddingModule
                (source='qwen'),
                manager=False,
                store_conf=store_conf
            )
   ```
4. 检索增强生成 (RAG) ：构建一个处理管道，首先通过 Retriever 从向量数据库中检索相关信息，然后将检索到的内容传递给大语言模型生成最终的工作总结。
   
   ```python
            # 构建pipeline
            with pipeline() as ppl:
                ppl.retriever = Retriever
                (doc, 
                group_name='sentences', 
                similarity="cosine", 
                topk=6, 
                output_format='content')
                ppl.formatter = (lambda 
                context, query: dict
                (context_str=str
                (context), 
                query=query)) | bind
                (query=ppl.input)
                ppl.llm = OnlineChatModule
                (source='qwen', 
                stream=False).prompt(
                    lazyllm.ChatPrompter
                    (self.prompt, 
                    extra_keys=
                    ["context_str"])
                )
   ```
## ChromaDB 优化策略介绍
为了提高文档检索的效率和准确性，我们对 ChromaDB 进行了以下优化：

1. 使用 HNSW 索引算法 ：HNSW (Hierarchical Navigable Small World) 是一种高效的近似最近邻搜索算法，能够在高维向量空间中快速找到相似向量。通过配置 index_kwargs ，我们启用了 HNSW 索引以加速检索过程。
   
   ```python
                'vector_store': {
                    'type': 'chromadb',
                    'kwargs': {
                        'dir': os.path.
                        join(self.db_dir, 
                        'chromadb_time'),
                        'index_kwargs': {
                            'hnsw': {
                                'space': 
                                'cosine',
                                'ef_constr
                                uction': 
                                100,
                            }
                        }
                    },
                },
   ```
2. 调整 ef_construction 参数 ： ef_construction 参数控制了索引构建时的精度和速度权衡。我们将其设置为 100，以在保证检索质量的同时，尽可能提高索引构建的效率。
## 最终核心功能展示
系统通过 Gradio 提供了一个简洁的 Web 界面，用户可以输入开始和结束日期，点击按钮即可生成该时间段内的工作总结。

```python
# 创建 Gradio 界面
def create_gradio_interface():
    # 初始化系统
    summary_system = 
    SmartWorkSummarySystem()
    
    def generate_summary_wrapper
    (start_date, end_date):
        return summary_system.
        generate_summary(start_date, 
        end_date)
    
    # 创建 Gradio 界面
    with gr.Blocks(title="智能工作总结系统
    ") as demo:
        gr.Markdown("# 智能工作总结系统")
        gr.Markdown("选择时间范围以生成该时间
        段内修改文件的工作总结")
        
        with gr.Row():
            with gr.Column():
                start_date = gr.Textbox
                (label="开始日期 
                (YYYY-MM-DD)", 
                value="2023-01-01")
                end_date = gr.Textbox
                (label="结束日期 
                (YYYY-MM-DD)", 
                value="2023-12-31")
                btn = gr.Button("生成工作总
                结")
            
            with gr.Column():
                output = gr.Textbox
                (label="工作总结", 
                lines=20, 
                interactive=False)
        
        btn.click
        (fn=generate_summary_wrapper, 
        inputs=[start_date, end_date], 
        outputs=output)
    
    return demo
```
通过以上设计和实现，系统能够高效地根据时间范围筛选文档，并利用 RAG 技术生成高质量的工作总结，为用户提供便捷的自动化工具。