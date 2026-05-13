import os
import glob
import json
import hashlib
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from models import *

# 获得嵌入模型客户端
llm, embeddings_model = get_ali_clients()

# 持久化路径
data_dir = "data"
faiss_index_path = os.path.join(data_dir, "faiss_index")
metadata_path = os.path.join(data_dir, "metadata.json")
processed_log_path = os.path.join(data_dir, "processed_files.json")

# 文档分割器（与cutting.py保持一致）
chunk_size = 1024
chunk_overlap = 100
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "。", "；", " ", ""]
)


def calculate_file_hash(file_path):
    """计算文件MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_single_document(file_path):
    """加载单个文档文件"""
    filename = os.path.basename(file_path)

    try:
        if file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()

        else:
            return []

        # 添加元数据
        for doc in docs:
            doc.metadata['source'] = filename
            doc.metadata['type'] = 'regulation'

        print(f"  ✓ 成功加载: {filename} ({len(docs)} 个段落)")
        return docs

    except Exception as e:
        print(f"  ✗ 加载失败 {filename}: {e}")
        return []


def incremental_update(new_folder="new"):
    """增量更新函数"""
    print("=" * 80)
    print("知识库增量更新系统")
    print("=" * 80)

    # 检查现有索引
    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        print("\n⚠ 未找到现有索引")
        print("请先运行 cutting.py 创建基础索引，或将文档放入 new 文件夹进行初始化\n")

        return

    # 加载现有索引
    print("\n[1/5] 加载现有索引...")
    vectorstore = FAISS.load_local(
        faiss_index_path,
        embeddings_model,
        allow_dangerous_deserialization=True
    )

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    next_doc_id = max([meta['doc_id'] for meta in metadata_list]) + 1 if metadata_list else 0
    print(f"  ✓ 现有文档块: {len(metadata_list)}")
    print(f"  ✓ 下一个ID: {next_doc_id}")

    # 加载处理日志
    processed_log = {}
    if os.path.exists(processed_log_path):
        with open(processed_log_path, 'r', encoding='utf-8') as f:
            processed_log = json.load(f)

    # 扫描新文件夹
    print(f"\n[2/5] 扫描 {new_folder} 文件夹...")
    if not os.path.exists(new_folder):
        print(f"  ✗ {new_folder} 不存在")
        return

    new_files = glob.glob(os.path.join(new_folder, "*.docx"))

    if not new_files:
        print(f"  ⊘ 未找到文档文件")
        return

    print(f"  ✓ 找到 {len(new_files)} 个文件")

    # 过滤需要处理的文件
    print(f"\n[3/5] 检查文件状态...")
    files_to_process = []

    for file_path in new_files:
        filename = os.path.basename(file_path)
        file_hash = calculate_file_hash(file_path)

        if filename in processed_log:
            old_hash = processed_log[filename].get('hash', '')
            if old_hash == file_hash:
                print(f"  ⊘ 跳过 (未变化): {filename}")
                continue
            else:
                print(f"  ⟳ 更新: {filename}")
        else:
            print(f"  ✓ 新增: {filename}")

        files_to_process.append((file_path, filename, file_hash))

    if not files_to_process:
        print("\n⊘ 没有需要处理的文件")
        return

    # 处理文件
    print(f"\n[4/5] 处理 {len(files_to_process)} 个文件...")
    all_new_docs = []

    for file_path, filename, file_hash in files_to_process:
        print(f"\n处理: {filename}")
        docs = load_single_document(file_path)

        if docs:
            all_new_docs.extend(docs)
            processed_log[filename] = {
                'hash': file_hash,
                'update_time': datetime.now().isoformat(),
                'doc_count': len(docs)
            }

    if not all_new_docs:
        print("\n✗ 没有成功加载任何文档")
        return

    print(f"\n  ✓ 共加载 {len(all_new_docs)} 个文档片段")

    # 分割文档
    print(f"\n[5/5] 分割和向量化...")
    split_docs = text_splitter.split_documents(all_new_docs)
    print(f"  ✓ 分割为 {len(split_docs)} 个文档块")

    # 为每个文档块添加ID
    for i, doc in enumerate(split_docs):
        doc.metadata['doc_id'] = next_doc_id + i

    # 创建新向量库
    print(f"  向量化中...")
    new_vectorstore = FAISS.from_documents(split_docs, embeddings_model)

    # 构建新元数据
    new_metadata = []
    for i, doc in enumerate(split_docs):
        doc_id = next_doc_id + i
        new_metadata.append({
            'doc_id': doc_id,
            'source': doc.metadata.get('source', ''),
            'content': doc.page_content,
            'type': doc.metadata.get('type', '')
        })

    # 合并索引
    print(f"  合并索引...")
    vectorstore.merge_from(new_vectorstore)

    # 更新元数据列表
    metadata_list.extend(new_metadata)

    # 保存
    print(f"\n保存更新...")
    vectorstore.save_local(faiss_index_path)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    with open(processed_log_path, 'w', encoding='utf-8') as f:
        json.dump(processed_log, f, ensure_ascii=False, indent=2)

    # 打印总结
    print(f"\n{'=' * 80}")
    print("✓ 增量更新完成!")
    print(f"{'=' * 80}")
    print(f"  新增文件: {len(files_to_process)} 个")
    print(f"  新增文档块: {len(split_docs)} 个")
    print(f"  总文档块数: {len(metadata_list)} 个")
    print(f"  下一ID起始: {next_doc_id + len(split_docs)}")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    incremental_update()
