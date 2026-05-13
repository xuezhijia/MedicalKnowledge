import os
import glob
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from models import *

# 获得嵌入模型客户端
llm, embeddings_model = get_ali_clients()

# 持久化路径配置
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
faiss_index_path = os.path.join(data_dir, "faiss_index")
metadata_path = os.path.join(data_dir, "metadata.json")

# 文档分割器配置（递归）
chunk_size = 1024  # 法律、条文、书籍的文档可以稍大一些，保持上下文完整
chunk_overlap = 100
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "。", "；", " ", ""]  # 针对中文优化
)


def load_doc_files(folder_path="original"):
    """加载文件夹内的所有docx文件（这里尽量都转docx，因为doc文档加载我试了会有些转化问题）"""
    documents = []

    if not os.path.exists(folder_path):
        print(f"警告: {folder_path} 文件夹不存在")
        return documents

    docx_files = glob.glob(os.path.join(folder_path, "*.docx"))

    if not docx_files:
        print(f"警告: 在 {folder_path} 中未找到任何docx文件")
        return documents

    print(f"找到 {len(docx_files)} 个文档文件\n")

    for file_path in docx_files:
        try:
            filename = os.path.basename(file_path)
            print(f"正在加载: {filename}")

            loader = Docx2txtLoader(file_path)
            docs = loader.load()

            # 添加元数据
            for doc in docs:
                doc.metadata['source'] = filename
                doc.metadata['type'] = 'regulation'

            documents.extend(docs)
            print(f"  ✓ 成功加载 ({len(docs)} 个段落)\n")

        except Exception as e:
            print(f"  ✗ 错误: {e}\n")

    print(f"总共加载了 {len(documents)} 个文档片段")
    return documents


def create_vector_index(documents):
    """创建向量索引"""
    print("\n" + "=" * 80)
    print("开始创建向量索引...")
    print("=" * 80)

    # 步骤1：分割文档
    print("\n[1/3] 分割文档...")
    split_docs = text_splitter.split_documents(documents)
    print(f"  ✓ 分割完成: {len(split_docs)} 个文档块")

    # 为每个文档块添加ID，必要，不然召回无法对应哦
    for i, doc in enumerate(split_docs):
        doc.metadata['doc_id'] = i

    # 步骤2：向量化
    print("\n[2/3] 向量化文档...")
    vectorstore = FAISS.from_documents(split_docs, embeddings_model)
    print(f"  ✓ 向量化完成")

    # 步骤3：构建元数据
    print("\n[3/3] 构建元数据...")
    metadata_list = []
    for i, doc in enumerate(split_docs):
        metadata_list.append({
            'doc_id': i,
            'source': doc.metadata.get('source', ''),
            'content': doc.page_content,
            'type': doc.metadata.get('type', '')
        })

    print(f"  ✓ 元数据构建完成: {len(metadata_list)} 条记录")

    return vectorstore, metadata_list


def save_index(vectorstore, metadata_list):
    """保存索引到磁盘"""
    print("\n保存索引...")

    vectorstore.save_local(faiss_index_path)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print(f"  ✓ FAISS索引: {faiss_index_path}")
    print(f"  ✓ 元数据: {metadata_path}")


def load_index():
    """加载索引"""
    print("加载索引...")

    vectorstore = FAISS.load_local(
        faiss_index_path,
        embeddings_model,
        allow_dangerous_deserialization=True
    )

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    print(f"  ✓ 已加载 {len(metadata_list)} 个文档块")

    return vectorstore, metadata_list


def check_index_exists():
    """检查索引是否存在"""
    return (os.path.exists(faiss_index_path) and
            os.path.exists(os.path.join(faiss_index_path, "index.faiss")) and
            os.path.exists(metadata_path))


def search_and_display(vectorstore, metadata_list, query, top_k=5):
    """搜索并展示结果"""
    print(f"\n正在搜索: {query}")
    print("-" * 80)

    # 向量检索
    similar_docs = vectorstore.similarity_search(query, k=top_k)

    if not similar_docs:
        print("未找到相关结果")
        return []

    print(f"找到 {len(similar_docs)} 个相关结果:\n")

    results = []
    for i, doc in enumerate(similar_docs, 1):
        doc_id = doc.metadata.get('doc_id')

        # 查找完整元数据
        for meta in metadata_list:
            if meta['doc_id'] == doc_id:
                result = {
                    'rank': i,
                    'source': meta['source'],
                    'content': meta['content'],
                    'preview': meta['content'][:300]
                }
                results.append(result)

                print(f"结果 {i}:")
                print(f"  来源: {meta['source']}")
                print(f"  预览: {result['preview']}...")
                print(f"  长度: {len(meta['content'])} 字符\n")
                break

    return results


def build_rag_prompt(query, results):
    """构建RAG提示词"""
    context = "\n\n".join([
        f"[文档{i + 1}] 来源: {r['source']}\n内容: {r['content']}"
        for i, r in enumerate(results)
    ])

    prompt = f"""基于以下参考文档，回答问题：{query}

参考文档：
{context}

请根据上述文档内容，给出准确、详细的回答。如果文档中没有相关信息，请明确说明。

回答："""

    # # 如果适用链或管道符形式，定义提示模版
    # prompt_template = ChatPromptTemplate.from_messages(
    #     [
    #         SystemMessagePromptTemplate.from_template(f"基于以下参考文档，回答问题{query}，请根据文档内容，给出准确、详细的回答。如果文档中没有相关信息，请明确说明"),
    #         ('human', f'参考文档：{context}')
    #     ]
    # )
    return prompt


def search_with_llm_enhancement(vectorstore, metadata_list, query, top_k=5):
    """搜索并使用大模型增强结果"""
    # 获取检索结果
    results = search_and_display(vectorstore, metadata_list, query, top_k)

    if not results:
        return "未找到相关文档"

    # 构建提示词
    prompt = build_rag_prompt(query, results)

    # 调用大模型（需要根据你的models.py中的实际接口调整）
    try:
        # 假设llm有invoke或call方法，根据实际情况调整
        response = llm.invoke(prompt)

        # 以链的形式调用
        # chain = prompt | llm | StrOutputParser()

        print("\n" + "=" * 80)
        print("大模型回答:")
        print("=" * 80)
        # 大模型返回的内容
        print(response.content)
        print("=" * 80)

        return {
            'query': query,
            'results': results,
            'answer': response
        }
    except Exception as e:
        print(f"调用大模型出错: {e}")
        return None


if __name__ == "__main__":
    print("=" * 80)
    print("医疗知识库检索系统")
    print("=" * 80)

    # 检查并加载/创建索引
    if check_index_exists():
        print("\n检测到现有索引，加载中...\n")
        vectorstore, metadata_list = load_index()
    else:
        print("\n未检测到索引，开始创建...\n")

        documents = load_doc_files("original")

        if not documents:
            print("错误: 没有加载到任何文档")
            exit(1)

        vectorstore, metadata_list = create_vector_index(documents)
        save_index(vectorstore, metadata_list)

        print("\n✓ 索引创建完成！\n")

    # 交互式查询
    print("=" * 80)
    print("开始查询 (输入 quit/exit/q 退出)")
    print("=" * 80)

    while True:
        user_query = input("\n请输入问题: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("感谢使用，再见！")
            break

        if not user_query:
            continue

        try:
            # 使用大模型增强的搜索
            search_with_llm_enhancement(vectorstore, metadata_list, user_query, top_k=5)
        except Exception as e:
            print(f"查询出错: {e}")
            import traceback

            traceback.print_exc()
