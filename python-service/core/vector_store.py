import os
import shutil
import threading
from typing import List, Optional, Dict, Any
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from core.reranker import create_reranker, BaseReranker
from core.config import config

# pymilvus 和 Milvus 是可选依赖，仅在使用 Milvus 时需要
try:
    from pymilvus import connections, utility
    from langchain_community.vectorstores import Milvus
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class VectorStoreManager:
    def __init__(self, persist_directory=None, use_milvus=None):
        """
        初始化向量存储管理器

        Args:
            persist_directory: FAISS持久化目录（仅当use_milvus=False时使用）
            use_milvus: 是否使用Milvus（默认从环境变量读取，默认值为True）
        """
        # 从环境变量读取持久化目录，默认值为./faiss_index
        self.persist_directory = persist_directory or config.VECTOR_STORE_PERSIST_DIR
        # 从环境变量读取use_milvus配置，如果未设置则默认为True
        if use_milvus is None:
            use_milvus = config.USE_MILVUS
        self.use_milvus = use_milvus
        # 从环境变量读取集合名称，默认值为ai_knowledge_collection
        self.collection_name = config.VECTOR_STORE_COLLECTION_NAME
        self.vector_store = None
        self.embeddings = None
        self._initialized = False
        self._initialization_lock = threading.Lock()
        self._mutation_lock = threading.Lock()

        # 初始化Reranker
        self._init_reranker()

        # 如果 pymilvus 未安装，强制使用 FAISS
        if self.use_milvus and not MILVUS_AVAILABLE:
            config.logger.warning("未安装 pymilvus，已降级使用 FAISS")
            self.use_milvus = False

        # Embedding 和向量库在首次实际使用时初始化，避免导入模块或
        # 收集单元测试时加载本地模型、访问网络或连接外部服务。

    def _init_embeddings(self):
        """根据配置初始化 Embedding 模型。"""
        embedding_model = config.EMBEDDING_MODEL.lower()

        if embedding_model == "local":
            # 使用本地中文 Embedding 模型
            local_model = config.LOCAL_EMBEDDING_MODEL
            config.logger.info(f"正在使用本地 HuggingFace 向量模型（{local_model}）")
            self.embeddings = HuggingFaceEmbeddings(model_name=local_model)
        else:
            # 默认使用阿里云 DashScope Embeddings (text-embedding-v1)
            api_key = config.DASHSCOPE_API_KEY
            if api_key:
                config.logger.info("正在使用 DashScope 向量模型（text-embedding-v1）")
                self.embeddings = DashScopeEmbeddings(
                    model="text-embedding-v1",
                    dashscope_api_key=api_key
                )
            else:
                config.logger.warning("未配置 DASHSCOPE_API_KEY，已降级使用本地 HuggingFace 向量模型")
                self.embeddings = HuggingFaceEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL)

    def _ensure_initialized(self):
        """线程安全地延迟初始化 Embedding 与向量数据库。"""
        if self._initialized:
            return

        with self._initialization_lock:
            if self._initialized:
                return

            self._init_embeddings()
            if self.use_milvus:
                self._init_milvus()
            else:
                self._init_faiss()
            self._initialized = True

    def _init_reranker(self):
        """初始化Reranker"""
        reranker_type = config.RERANKER_TYPE
        if reranker_type == "none":
            self.reranker = None
            config.logger.info("重排序功能已关闭")
            return

        try:
            self.reranker = create_reranker(reranker_type)
            config.logger.info(f"重排序器初始化完成：{reranker_type}")
        except Exception as e:
            config.logger.error(f"重排序器初始化失败：{e}")
            self.reranker = None

    def _init_milvus(self):
        """初始化Milvus连接和集合"""
        try:
            # Milvus连接配置
            milvus_host = config.MILVUS_HOST
            milvus_port = config.MILVUS_PORT

            config.logger.info(f"正在连接 Milvus：{milvus_host}:{milvus_port}")
            connections.connect(alias="default", host=milvus_host, port=milvus_port)

            # 检查连接
            if not connections.has_connection("default"):
                raise ConnectionError("Failed to connect to Milvus")

            config.logger.info("Milvus 连接成功")

            # 初始化Milvus向量存储
            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={
                    "host": milvus_host,
                    "port": milvus_port,
                    "alias": "default"
                },
                # 自动创建集合（如果不存在）
                auto_id=True
            )

            config.logger.info(f"Milvus 集合已就绪：{self.collection_name}")

        except Exception as e:
            config.logger.error(f"Milvus 初始化失败：{e}")
            config.logger.info("正在降级使用 FAISS")
            self.use_milvus = False
            self._init_faiss()

    def _init_faiss(self):
        """
        初始化 FAISS 向量存储（作为 Milvus 的 fallback）
        """
        config.logger.info("正在使用 FAISS 向量库")
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
                config.logger.info(f"已从 {self.persist_directory} 加载现有 FAISS 索引")
            except Exception as e:
                config.logger.error(f"现有 FAISS 索引加载失败：{e}")
                config.logger.info("可能是向量模型变更导致维度不一致，将重新初始化空 FAISS 向量库")
                # Backup old index just in case
                if os.path.exists(self.persist_directory + "_backup"):
                    shutil.rmtree(self.persist_directory + "_backup")
                shutil.move(self.persist_directory, self.persist_directory + "_backup")
                self.vector_store = None
        else:
            self.vector_store = None
            config.logger.info("未发现现有 FAISS 索引，将在首次写入时创建")

    def add_documents(self, documents: List[Document]):
        """
        添加文档到向量数据库
        """
        if not documents:
            return

        self._ensure_initialized()

        if self.vector_store is None:
            if self.use_milvus:
                # Milvus会自动创建集合
                milvus_host = config.MILVUS_HOST
                milvus_port = config.MILVUS_PORT
                self.vector_store = Milvus.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args={
                        "host": milvus_host,
                        "port": milvus_port,
                        "alias": "default"
                    }
                )
                config.logger.info(f"Milvus 集合 {self.collection_name} 创建完成，写入 {len(documents)} 个片段")
            else:
                # FAISS
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_store.save_local(self.persist_directory)
                config.logger.info(f"FAISS 索引创建完成，写入 {len(documents)} 个片段")
        else:
            # 添加文档到现有存储
            if self.use_milvus:
                # Milvus添加文档
                self.vector_store.add_documents(documents)
                config.logger.info(f"已向 Milvus 添加 {len(documents)} 个片段")
            else:
                # FAISS添加文档
                self.vector_store.add_documents(documents)
                self.vector_store.save_local(self.persist_directory)
                config.logger.info(f"已向 FAISS 添加 {len(documents)} 个片段")

    def create_pending_embeddings(
        self,
        doc_id: int,
        documents: List[Document],
    ) -> np.ndarray:
        """只计算并缓存待审核文档向量，不写入线上向量库。"""
        if not documents:
            raise ValueError("待审核文档没有可向量化的知识片段")
        self._ensure_initialized()
        if self.use_milvus:
            raise RuntimeError("预发布审核流程当前要求使用 FAISS")

        vectors = np.asarray(
            self.embeddings.embed_documents([doc.page_content for doc in documents]),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise RuntimeError("Embedding 返回的向量数量与知识片段数量不一致")

        os.makedirs(config.PENDING_VECTOR_DIR, exist_ok=True)
        target = self._pending_vector_path(doc_id)
        temp = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
        np.savez_compressed(temp, vectors=vectors)
        os.replace(temp, target)
        config.logger.info(
            "文档 %s 的待审核向量已生成，共 %s 个；未写入线上 FAISS",
            doc_id,
            len(vectors),
        )
        return vectors

    @staticmethod
    def _pending_vector_path(doc_id: int) -> str:
        return os.path.join(config.PENDING_VECTOR_DIR, f"document_{int(doc_id)}.npz")

    def load_pending_embeddings(self, doc_id: int) -> np.ndarray:
        """读取审核阶段缓存的向量，并校验基本结构。"""
        path = self._pending_vector_path(doc_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文档 {doc_id} 的待审核向量缓存不存在")
        with np.load(path, allow_pickle=False) as payload:
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"文档 {doc_id} 的待审核向量格式无效")
        return vectors

    def delete_pending_embeddings(self, doc_id: int) -> None:
        """删除待审核向量缓存。"""
        path = self._pending_vector_path(doc_id)
        if os.path.isfile(path):
            os.remove(path)

    def find_active_matches_by_vectors(
        self,
        query_vectors: np.ndarray,
        active_doc_ids: Optional[List[int]] = None,
        top_k: Optional[int] = None,
        cosine_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """让新向量批量查询已有 ACTIVE FAISS，不重新计算旧知识向量。"""
        top_k = config.DUPLICATE_TOP_K if top_k is None else top_k
        cosine_threshold = (
            config.INSPECTION_COSINE_THRESHOLD
            if cosine_threshold is None else cosine_threshold
        )
        max_candidates = (
            config.DUPLICATE_MAX_CANDIDATES
            if max_candidates is None else max_candidates
        )
        if self.use_milvus:
            raise RuntimeError("预发布语义巡检当前仅支持 FAISS")

        inspection_store = self._get_faiss_inspection_store()
        if inspection_store is None:
            config.logger.info("线上 FAISS 为空，新文档无需进行跨文档语义比对")
            return []
        index = getattr(inspection_store, "index", None)
        total = int(getattr(index, "ntotal", 0) or 0)
        if total == 0:
            return []

        old_vectors = self._reconstruct_faiss_vectors(index, total)
        new_vectors = np.asarray(query_vectors, dtype=np.float32)
        if old_vectors is None or new_vectors.ndim != 2 or len(new_vectors) == 0:
            return []
        if old_vectors.shape[1] != new_vectors.shape[1]:
            raise ValueError("待审核向量与线上 FAISS 的维度不一致")

        old_norms = np.linalg.norm(old_vectors, axis=1, keepdims=True)
        new_norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        valid_old = old_norms[:, 0] > 0
        valid_new = new_norms[:, 0] > 0
        old_normalized = np.zeros_like(old_vectors, dtype=np.float32)
        new_normalized = np.zeros_like(new_vectors, dtype=np.float32)
        old_normalized[valid_old] = old_vectors[valid_old] / old_norms[valid_old]
        new_normalized[valid_new] = new_vectors[valid_new] / new_norms[valid_new]

        import faiss

        cosine_index = faiss.IndexFlatIP(old_normalized.shape[1])
        cosine_index.add(np.ascontiguousarray(old_normalized))
        search_k = min(total, max(1, int(top_k)))
        similarities, indices = cosine_index.search(
            np.ascontiguousarray(new_normalized), search_k
        )

        active_ids = (
            None
            if active_doc_ids is None
            else {str(item) for item in active_doc_ids}
        )
        candidates = []
        seen = set()
        neighbor_count = 0
        config.logger.info(
            "[知识巡检][FAISS TopK] 开始输出逐 Chunk 召回明细：新Chunk数=%s，TopK=%s，余弦阈值=%.4f",
            len(new_vectors),
            search_k,
            cosine_threshold,
        )
        for new_index, (scores, old_indices) in enumerate(zip(similarities, indices)):
            if not valid_new[new_index]:
                config.logger.warning(
                    "[知识巡检][FAISS TopK] 新Chunk=%s 的向量范数为0，跳过检索",
                    new_index,
                )
                continue
            for rank, (score, old_index) in enumerate(zip(scores, old_indices), start=1):
                old_index = int(old_index)
                if old_index < 0 or not valid_old[old_index]:
                    config.logger.info(
                        "[知识巡检][FAISS TopK] 新Chunk=%s，排名=%s，FAISS行=%s，结果=无效向量，已过滤",
                        new_index,
                        rank,
                        old_index,
                    )
                    continue
                neighbor_count += 1
                similarity = float(score)
                old_chunk = self._faiss_chunk_at(old_index, inspection_store)
                if old_chunk is None:
                    config.logger.info(
                        "[知识巡检][FAISS TopK] 新Chunk=%s，排名=%s，FAISS行=%s，cosine=%.6f，阈值=%.4f，阈值结果=%s，结果=元数据缺失，已过滤",
                        new_index,
                        rank,
                        old_index,
                        similarity,
                        cosine_threshold,
                        "通过" if similarity >= cosine_threshold else "未通过",
                    )
                    continue

                passed_threshold = similarity >= cosine_threshold
                is_active = (
                    active_ids is None
                    or str(old_chunk.get("doc_id")) in active_ids
                )
                accepted = passed_threshold and is_active
                config.logger.info(
                    "[知识巡检][FAISS TopK] 新Chunk=%s，排名=%s，旧文档ID=%s，来源=%s，旧Chunk=%s，FAISS行=%s，cosine=%.6f，阈值=%.4f，阈值结果=%s，ACTIVE过滤=%s，候选结果=%s，旧内容预览=%s",
                    new_index,
                    rank,
                    old_chunk.get("doc_id"),
                    old_chunk.get("source") or "未知来源",
                    old_chunk.get("chunk_index"),
                    old_index,
                    similarity,
                    cosine_threshold,
                    "通过" if passed_threshold else "未通过",
                    "通过" if is_active else "未通过",
                    "通过基础过滤" if accepted else "已过滤",
                    old_chunk.get("preview") or "",
                )
                if not passed_threshold:
                    continue
                if not is_active:
                    continue
                pair_key = (new_index, old_chunk["vector_id"])
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                candidates.append({
                    "new_chunk_index": new_index,
                    "old": old_chunk,
                    "similarity": similarity,
                })

        candidates.sort(key=lambda item: item["similarity"], reverse=True)
        result = candidates[:max_candidates]
        for final_rank, candidate in enumerate(result, start=1):
            old_chunk = candidate["old"]
            config.logger.info(
                "[知识巡检][LLM候选] 全局排名=%s，新Chunk=%s，旧文档ID=%s，来源=%s，旧Chunk=%s，cosine=%.6f，结果=进入LLM Judge",
                final_rank,
                candidate["new_chunk_index"],
                old_chunk.get("doc_id"),
                old_chunk.get("source") or "未知来源",
                old_chunk.get("chunk_index"),
                float(candidate["similarity"]),
            )
        if len(candidates) > len(result):
            config.logger.info(
                "[知识巡检][LLM候选] 有%s条通过阈值和ACTIVE过滤的候选因最大候选数=%s未进入LLM Judge",
                len(candidates) - len(result),
                max_candidates,
            )
        config.logger.info(
            "新文档语义巡检完成：近邻%s个，达到阈值且属于已发布知识%s个，最终候选%s个",
            neighbor_count,
            len(candidates),
            len(result),
        )
        return result

    def publish_precomputed_documents(
        self,
        documents: List[Document],
        vectors: np.ndarray,
    ) -> None:
        """将审核通过的文档及其已计算向量写入正式 FAISS。"""
        if not documents:
            raise ValueError("没有可发布的知识片段")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise ValueError("发布向量与知识片段数量不一致")

        self._ensure_initialized()
        if self.use_milvus:
            raise RuntimeError("预计算向量发布当前仅支持 FAISS")

        text_embeddings = [
            (document.page_content, vector.tolist())
            for document, vector in zip(documents, vectors)
        ]
        metadatas = [dict(document.metadata or {}) for document in documents]
        with self._mutation_lock:
            if self.vector_store is None:
                self.vector_store = FAISS.from_embeddings(
                    text_embeddings=text_embeddings,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                )
            else:
                self.vector_store.add_embeddings(
                    text_embeddings=text_embeddings,
                    metadatas=metadatas,
                )
            self.vector_store.save_local(self.persist_directory)
        config.logger.info("已将 %s 个审核通过的片段写入线上 FAISS", len(documents))

    def contains_document(self, doc_id: int) -> bool:
        """检查正式 FAISS 是否已经包含某文档，用于防止重复发布。"""
        if self.use_milvus:
            return False
        store = self._get_faiss_inspection_store()
        if store is None:
            return False
        for document in getattr(getattr(store, "docstore", None), "_dict", {}).values():
            if str(getattr(document, "metadata", {}).get("doc_id")) == str(doc_id):
                return True
        return False

    def search(self, query: str, k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None, use_rerank: bool = True) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 最终返回结果数量，默认使用 KNOWLEDGE_TOP_K
            filter_dict: 过滤条件（仅Milvus支持）
            use_rerank: 是否使用Rerank进行结果重排序
        """
        import time
        start_time = time.time()
        k = k or config.KNOWLEDGE_TOP_K
        self._ensure_initialized()
        
        if self.vector_store is None:
            config.logger.info(f"向量检索完成，耗时 {time.time() - start_time:.4f} 秒；当前没有可用向量库")
            return []

        try:
            # 初始检索数量应该比最终返回的多，以便Rerank有足够的候选
            initial_k = (
                k * config.RETRIEVAL_CANDIDATE_MULTIPLIER
                if use_rerank and self.reranker else k
            )

            search_start = time.time()
            if self.use_milvus and filter_dict:
                # Milvus支持过滤查询
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=initial_k, filter=filter_dict)
            else:
                # FAISS或无条件查询
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=initial_k)
            search_time = time.time() - search_start
            config.logger.info(f"向量召回完成，耗时 {search_time:.4f} 秒，找到 {len(docs_with_scores) if isinstance(docs_with_scores, list) else 0} 个片段")

            # 提取文档
            if isinstance(docs_with_scores, list):
                if len(docs_with_scores) > 0 and isinstance(docs_with_scores[0], tuple):
                    # (doc, score) 格式
                    docs = [doc for doc, score in docs_with_scores]
                else:
                    docs = docs_with_scores
            else:
                docs = docs_with_scores

            # FAISS 默认返回距离，距离与相似度分数的方向不同。
            # 因此召回阶段只返回 top-k 候选，不对距离套用统一阈值。
            if not use_rerank or not self.reranker:
                candidate_docs = [doc for doc, _ in docs_with_scores[:k]]
                config.logger.info(f"检索完成，耗时 {time.time() - start_time:.4f} 秒，返回 {len(candidate_docs)} 个片段")
                return candidate_docs

            # 使用Rerank进行重排序
            try:
                rerank_start = time.time()
                rerank_results = self.reranker.rerank(query, docs, top_k=k)
                rerank_time = time.time() - rerank_start
                config.logger.info(f"重排序完成，耗时 {rerank_time:.4f} 秒，保留前 {len(rerank_results)} 个结果")

                # 将重排序分数随文档返回，供后续充分性判断使用。
                reranked_docs = []
                for result in rerank_results:
                    result.document.metadata = dict(result.document.metadata or {})
                    result.document.metadata["rerank_score"] = result.score
                    reranked_docs.append(result.document)

                config.logger.info(f"检索完成，耗时 {time.time() - start_time:.4f} 秒，返回 {len(reranked_docs)} 个片段")

                return reranked_docs

            except Exception as rerank_error:
                config.logger.error(f"重排序失败：{rerank_error}，已降级使用向量召回顺序")
                # Rerank失败时，回退到 FAISS 的 top-k 候选。
                candidate_docs = [doc for doc, _ in docs_with_scores[:k]]
                config.logger.info(f"降级检索完成，耗时 {time.time() - start_time:.4f} 秒，返回 {len(candidate_docs)} 个片段")
                return candidate_docs

        except Exception as e:
            config.logger.error(f"向量检索异常：{e}")
            # 如果 similarity_search_with_score 失败，回退到普通搜索
            try:
                fallback_start = time.time()
                if self.use_milvus and filter_dict:
                    result = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
                else:
                    result = self.vector_store.similarity_search(query, k=k)
                fallback_time = time.time() - fallback_start
                config.logger.info(f"备用检索完成，耗时 {fallback_time:.4f} 秒，返回 {len(result)} 个片段")
                return result
            except Exception as e2:
                config.logger.error(f"备用检索也失败：{e2}")
                config.logger.info(f"所有检索均失败，总耗时 {time.time() - start_time:.4f} 秒，返回空结果")
                return []

    def find_semantic_duplicate_pairs(
        self,
        top_k: Optional[int] = None,
        cosine_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """复用 FAISS 已有向量，批量召回跨文档语义重复候选。

        该方法不会调用 Embedding，也不会修改线上 FAISS 索引。它从原索引
        reconstruct 向量副本，归一化后放入临时 IndexFlatIP，以内积表示
        cosine similarity。
        """
        top_k = config.DUPLICATE_TOP_K if top_k is None else top_k
        cosine_threshold = (
            config.DUPLICATE_COSINE_THRESHOLD
            if cosine_threshold is None else cosine_threshold
        )
        max_candidates = (
            config.DUPLICATE_MAX_CANDIDATES
            if max_candidates is None else max_candidates
        )

        if top_k <= 0 or max_candidates <= 0:
            return []

        try:
            if self.use_milvus:
                config.logger.warning(
                    "语义重复巡检当前仅支持 FAISS 向量库"
                )
                return []
            inspection_store = self._get_faiss_inspection_store()
            if inspection_store is None:
                config.logger.info("FAISS 为空，跳过语义重复巡检")
                return []

            index = getattr(inspection_store, "index", None)
            total_chunks = int(getattr(index, "ntotal", 0) or 0)
            config.logger.info(
                "语义重复巡检开始，共扫描 %s 个 FAISS 知识片段",
                total_chunks,
            )
            if total_chunks < 2:
                return []

            vectors = self._reconstruct_faiss_vectors(index, total_chunks)
            if vectors is None or len(vectors) < 2:
                return []

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            valid_mask = norms[:, 0] > 0
            if not np.all(valid_mask):
                config.logger.warning(
                    "发现 %s 个零向量，语义重复巡检将跳过这些向量",
                    int((~valid_mask).sum()),
                )

            normalized_vectors = np.zeros_like(vectors, dtype=np.float32)
            normalized_vectors[valid_mask] = (
                vectors[valid_mask] / norms[valid_mask]
            ).astype(np.float32, copy=False)
            normalized_vectors = np.ascontiguousarray(normalized_vectors)

            import faiss

            cosine_index = faiss.IndexFlatIP(normalized_vectors.shape[1])
            cosine_index.add(normalized_vectors)
            search_k = min(total_chunks, top_k + 1)
            similarities, neighbor_indices = cosine_index.search(
                normalized_vectors, search_k
            )

            pair_candidates: Dict[Any, Dict[str, Any]] = {}
            neighbor_count = 0
            threshold_count = 0
            cross_document_count = 0

            for source_index, (scores, indices) in enumerate(
                zip(similarities, neighbor_indices)
            ):
                if not valid_mask[source_index]:
                    continue
                source_item = self._faiss_chunk_at(source_index, inspection_store)
                if source_item is None:
                    continue

                for score, target_index in zip(scores, indices):
                    target_index = int(target_index)
                    if target_index < 0 or target_index == source_index:
                        continue
                    neighbor_count += 1
                    similarity = float(score)
                    if similarity < cosine_threshold:
                        continue
                    threshold_count += 1

                    target_item = self._faiss_chunk_at(target_index, inspection_store)
                    if target_item is None:
                        continue
                    if str(source_item["doc_id"]) == str(target_item["doc_id"]):
                        continue
                    cross_document_count += 1

                    left, right = sorted(
                        (source_item, target_item),
                        key=lambda item: item["vector_id"],
                    )
                    pair_key = (left["vector_id"], right["vector_id"])
                    previous = pair_candidates.get(pair_key)
                    if previous is None or similarity > previous["similarity"]:
                        pair_candidates[pair_key] = {
                            "left": left,
                            "right": right,
                            "similarity": similarity,
                        }

            candidates = sorted(
                pair_candidates.values(),
                key=lambda item: item["similarity"],
                reverse=True,
            )[:max_candidates]
            config.logger.info(
                "语义近邻共%s个，达到阈值%s个，跨文档%s个，"
                "Pair去重后%s对，最终返回%s对",
                neighbor_count,
                threshold_count,
                cross_document_count,
                len(pair_candidates),
                len(candidates),
            )
            return candidates
        except Exception as exc:
            config.logger.warning(
                "语义重复候选召回失败，将返回空候选：%s",
                exc,
            )
            return []

    def _get_faiss_inspection_store(self):
        """获取只读 FAISS 巡检视图，不触发 Embedding 模型初始化。

        服务已经初始化时复用内存中的 LangChain FAISS；否则直接读取
        LangChain 保存的 index.faiss/index.pkl。后者只用于本次巡检，
        不写回 self.vector_store，也不会改变正常知识问答的初始化流程。
        """
        current_store = self.vector_store
        if current_store is not None and getattr(current_store, "index", None) is not None:
            return current_store

        index_path = os.path.join(self.persist_directory, "index.faiss")
        metadata_path = os.path.join(self.persist_directory, "index.pkl")
        if not os.path.isfile(index_path) or not os.path.isfile(metadata_path):
            return None

        try:
            import faiss
            import pickle
            from types import SimpleNamespace

            index = faiss.read_index(index_path)
            # index.pkl 是本项目自己通过 LangChain FAISS.save_local 生成的；
            # 与现有 load_local(...allow_dangerous_deserialization=True) 风险边界一致。
            with open(metadata_path, "rb") as metadata_file:
                docstore, index_to_docstore_id = pickle.load(metadata_file)
            config.logger.info(
                "只读 FAISS 巡检快照加载完成，共 %s 个向量",
                int(getattr(index, "ntotal", 0) or 0),
            )
            return SimpleNamespace(
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )
        except Exception as exc:
            config.logger.warning("只读 FAISS 巡检快照加载失败：%s", exc)
            return None

    @staticmethod
    def _reconstruct_faiss_vectors(index: Any, total_chunks: int) -> Optional[np.ndarray]:
        """兼容 reconstruct_n 的不同签名，并回退到逐条 reconstruct。"""
        try:
            vectors = index.reconstruct_n(0, total_chunks)
            if vectors is not None:
                result = np.asarray(vectors, dtype=np.float32)
                if result.ndim == 2 and result.shape[0] == total_chunks:
                    return result
        except Exception as exc:
            config.logger.warning(
                "FAISS 批量向量重建失败，降级为逐条重建：%s",
                exc,
            )

        try:
            vectors = [
                np.asarray(index.reconstruct(i), dtype=np.float32)
                for i in range(total_chunks)
            ]
            return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
        except Exception as exc:
            config.logger.warning("FAISS 向量重建失败：%s", exc)
            return None

    def _faiss_chunk_at(self, vector_index: int,
                        faiss_store: Any = None) -> Optional[Dict[str, Any]]:
        """按 FAISS 行号解析 LangChain docstore 文档，不依赖 MySQL chunk_id。"""
        store = faiss_store or self.vector_store
        mapping = getattr(store, "index_to_docstore_id", {}) or {}
        vector_id = mapping.get(vector_index)
        if vector_id is None:
            config.logger.warning(
                "FAISS 第 %s 行缺少 index_to_docstore_id 映射",
                vector_index,
            )
            return None

        docstore = getattr(store, "docstore", None)
        document = None
        if docstore is not None and hasattr(docstore, "search"):
            document = docstore.search(vector_id)
        if not isinstance(document, Document):
            document = getattr(docstore, "_dict", {}).get(vector_id)
        if not isinstance(document, Document):
            config.logger.warning("FAISS 第 %s 行找不到对应的 LangChain 文档", vector_index)
            return None

        metadata = dict(document.metadata or {})
        doc_id = metadata.get("doc_id")
        if doc_id is None:
            config.logger.warning("FAISS 第 %s 行缺少 doc_id 元数据，已跳过", vector_index)
            return None

        content = str(document.page_content or "").strip()
        preview = " ".join(content.split())[:160]
        return {
            "vector_id": str(vector_id),
            "doc_id": doc_id,
            "source": metadata.get("source") or "未知来源",
            "chunk_index": metadata.get("chunk_index"),
            "content": content,
            "preview": preview,
        }

    def delete_document(self, doc_id: int):
        """
        根据 doc_id 删除文档向量

        Milvus: 支持高效删除
        FAISS: 标记删除（实际需要重建索引）
        """
        self._ensure_initialized()

        if self.vector_store is None:
            config.logger.warning(f"No vector store available, cannot delete doc_id: {doc_id}")
            return

        if self.use_milvus:
            # Milvus删除逻辑
            try:
                config.logger.info(f"Deleting document with doc_id: {doc_id} from Milvus")

                # 构建删除表达式（使用类型转换确保安全性）
                if not isinstance(doc_id, int):
                    raise ValueError(f"doc_id must be an integer, got {type(doc_id)}")
                delete_expr = f'doc_id in [{doc_id}]'

                # 执行删除
                result = self.vector_store.delete(expr=delete_expr)
                config.logger.info(f"Milvus delete result: {result}")

                # 可选：压缩集合以释放空间
                # utility.compact(collection_name=self.collection_name)

                config.logger.info(f"Successfully deleted document {doc_id} from Milvus")

            except Exception as e:
                config.logger.error(f"Failed to delete document from Milvus: {e}")
                # 尝试其他删除方法
                self._delete_document_fallback(doc_id)

        else:
            # FAISS删除逻辑（效率较低）
            config.logger.info(f"Deleting document with doc_id: {doc_id} from FAISS")
            self._delete_document_faiss(doc_id)

    def _delete_document_fallback(self, doc_id: int):
        """备用删除方法：通过查询找到ID然后删除"""
        try:
            # 先搜索包含该doc_id的文档
            filter_dict = {"doc_id": doc_id}
            docs_to_delete = self.search("", k=1000, filter_dict=filter_dict)

            if not docs_to_delete:
                config.logger.info(f"No documents found with doc_id: {doc_id}")
                return

            # 提取文档ID（假设metadata中有唯一ID）
            ids_to_delete = []
            for doc in docs_to_delete:
                if 'chunk_id' in doc.metadata:
                    ids_to_delete.append(doc.metadata['chunk_id'])

            if ids_to_delete:
                # 执行删除
                self.vector_store.delete(ids=ids_to_delete)
                config.logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id {doc_id}")
            else:
                config.logger.info(f"No deletable chunks found for doc_id {doc_id}")

        except Exception as e:
            config.logger.error(f"Fallback delete failed: {e}")

    def _delete_document_faiss(self, doc_id: int):
        """FAISS删除实现（需要重建索引）"""
        try:
            # 找到所有 metadata['doc_id'] == doc_id 的 ID
            ids_to_delete = []
            for doc_uuid, doc in self.vector_store.docstore._dict.items():
                if doc.metadata.get('doc_id') == doc_id:
                    ids_to_delete.append(doc_uuid)

            if ids_to_delete:
                # FAISS的delete方法可能不彻底，这里尝试删除
                self.vector_store.delete(ids_to_delete)
                self.vector_store.save_local(self.persist_directory)
                config.logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id {doc_id}")

                # 建议：定期重建FAISS索引以提高效率
                if len(ids_to_delete) > 100:
                    config.logger.warning("Large deletion in FAISS. Consider rebuilding index for better performance.")
            else:
                config.logger.info(f"No chunks found for doc_id {doc_id}")

        except Exception as e:
            config.logger.error(f"Failed to delete document from FAISS: {e}")

    def delete_collection(self):
        """
        删除整个向量库 (慎用)
        """
        self._ensure_initialized()

        if self.use_milvus:
            try:
                # 删除Milvus集合
                utility.drop_collection(self.collection_name)
                config.logger.info(f"Successfully deleted Milvus collection '{self.collection_name}'")
            except Exception as e:
                config.logger.error(f"Failed to delete Milvus collection: {e}")
        else:
            # 删除FAISS目录
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            self.vector_store = None
            config.logger.info("Successfully deleted FAISS collection")

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        self._ensure_initialized()

        stats = {
            "initialized": self._initialized,
            "using_milvus": self.use_milvus,
            "collection_name": self.collection_name if self.use_milvus else None,
            "persist_directory": self.persist_directory if not self.use_milvus else None,
        }

        if self.use_milvus and self.vector_store:
            try:
                # 获取Milvus集合信息
                collection_stats = utility.get_collection_stats(self.collection_name)
                stats.update({
                    "row_count": collection_stats.get("row_count", 0),
                    "partitions": collection_stats.get("partitions", []),
                })
            except Exception as e:
                stats["error"] = f"Failed to get Milvus stats: {e}"
        elif not self.use_milvus and self.vector_store:
            # FAISS统计
            stats["doc_count"] = len(self.vector_store.docstore._dict) if hasattr(self.vector_store, 'docstore') else 0

        return stats

    def migrate_faiss_to_milvus(self):
        """将FAISS数据迁移到Milvus"""
        self._ensure_initialized()

        if not self.use_milvus or self.vector_store is None:
            config.logger.warning("Cannot migrate: not using Milvus or no vector store")
            return False

        try:
            config.logger.info("Starting migration from FAISS to Milvus...")

            # 1. 加载FAISS数据
            if os.path.exists(self.persist_directory):
                faiss_store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)

                # 2. 提取所有文档
                all_docs = []
                for doc_uuid, doc in faiss_store.docstore._dict.items():
                    all_docs.append(doc)

                # 3. 添加到Milvus
                if all_docs:
                    self.add_documents(all_docs)
                    config.logger.info(f"Migrated {len(all_docs)} documents from FAISS to Milvus")

                    # 4. 备份原FAISS数据
                    backup_dir = self.persist_directory + "_migrated_backup"
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)
                    shutil.move(self.persist_directory, backup_dir)
                    config.logger.info(f"Backed up FAISS data to {backup_dir}")

                    return True

            return False

        except Exception as e:
            config.logger.error(f"Migration failed: {e}")
            return False


# 创建单例实例
vector_store_manager = VectorStoreManager()

# 导出（保持兼容性，同时保留完整管理器）
vector_store = vector_store_manager
