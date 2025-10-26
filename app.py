import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import gradio as gr
import nltk
import time
from rank_bm25 import BM25Okapi#
from evaluate import load#
import seaborn as sns#
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from evaluate import load
import matplotlib.pyplot as plt


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

PDF_FOLDER = r"D:\RAG"
PDF_FILES = [
    "ARCHIVED-6.06-Solar-hot-water-performance-requirments-250322.pdf",
    "Commercial-and-industrial-heat-pump-water-heaters-technical-guidelines.pdf",
    "Emergency-backstop-factsheet_June-2024.pdf",
    "Fact-Sheet-Solar-Homes-WEB-220823.pdf",
    "Gas-substitution-roadmap-new-homes-factsheet.pdf",
    "NoticetoMarket202425SolarVictoriav2.pdf",
    "NoticetoMarket202526v1.pdf",
    "Plumbing-SH-01-Solar-water-heaters-UNDER-REVIEW-16-May-2024.pdf",
    "SOL131_FactSheet_Battery_23.10.23.pdf",
    "Solar-Homes_FactSheet_09.01.23.pdf",
    "Solar-water-heater-Datasheet.pdf",
    "VEU-water-heating-consumer-factsheet.pdf",
]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "distilgpt2"

def load_pdfs(folder: str, files: List[str]):
    docs = []
    for pdf in files:
        path = os.path.join(folder, pdf)
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Skipped {pdf}: {e}")
    return docs

def build_vectorstore(documents):
    if not documents:
        raise RuntimeError("No documents loaded. Check PDF paths.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def build_llm():
    gen_pipe = pipeline(
        "text-generation",
        model=GEN_MODEL,
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True,
    )

    def gen_wrapper(prompt):
        return gen_pipe(prompt, max_length=len(prompt.split()) + 256)[0]['generated_text']

    return HuggingFacePipeline(pipeline=gen_pipe)

def build_qa_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
    )
    return chain

EVAL_DATA = [
    {
        "question": "What is the purpose of the Solar Homes program?",
        "expected_answer": "It aims to help Victorian households install solar panels and reduce energy costs."
    },
    {
        "question": "Who is eligible for solar water heater rebates?",
        "expected_answer": "Owner-occupiers with a combined household income of less than $210,000 per year."
    },
]

eval_embedder = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
smooth = SmoothingFunction().method1

def evaluate_qa_system(qa_chain, eval_data, sim_threshold=0.6):
    scores = []
    bleu_scores = []
    rouge_scores = []
    precisions = []
    recalls = []
    correctness = []
    coverage = []
    latencies = []

    for item in eval_data:
        question = item["question"]
        expected = item["expected_answer"]

        start_time = time.time()
        predicted = qa_chain.run(question)
        latency = time.time() - start_time
        latencies.append(latency)

        coverage.append(1 if predicted.strip() else 0)

        emb_expected = eval_embedder.encode([expected])
        emb_pred = eval_embedder.encode([predicted])
        sim = cosine_similarity(emb_expected, emb_pred)[0][0]

        bleu = sentence_bleu(
            [expected.split()],
            predicted.split(),
            smoothing_function=smooth
        )

        rouge_l = rouge.score(expected, predicted)["rougeL"].fmeasure

        is_correct = sim >= sim_threshold
        precisions.append(1 if is_correct else 0)
        recalls.append(1 if is_correct else 0)
        correctness.append(1 if is_correct else 0)

        scores.append(sim)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)

        print(f"\nQ: {question}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"🔹 Semantic Similarity: {sim:.3f}")
        print(f"🔹 BLEU: {bleu:.3f}")
        print(f"🔹 ROUGE-L: {rouge_l:.3f}")
        print(f"🔹 Correct: {is_correct}, Latency: {latency:.3f}s")

    print("\n--- Evaluation Summary ---")
    print(f"Average Semantic Similarity: {np.mean(scores):.3f}")
    print(f"Average BLEU: {np.mean(bleu_scores):.3f}")
    print(f"Average ROUGE-L: {np.mean(rouge_scores):.3f}")
    print(f"Precision: {np.mean(precisions):.3f}")
    print(f"Recall: {np.mean(recalls):.3f}")
    print(f"Coverage: {np.mean(coverage):.3f}")
    print(f"Correctness: {np.mean(correctness):.3f}")
    print(f"Latency p95: {np.percentile(latencies, 95):.3f}s")

    return {
        "semantic": np.mean(scores),
        "bleu": np.mean(bleu_scores),
        "rougeL": np.mean(rouge_scores),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "coverage": np.mean(coverage),
        "correctness": np.mean(correctness),
        "latency_p95": np.percentile(latencies, 95)
    }

def evaluate_retriever(vectorstore, eval_data, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    total_precision, total_recall, mrr_scores = [], [], []
    bert_f1_scores, latencies = [], []

    bertscore = load("bertscore")

    for item in eval_data:
        question = item["question"]
        relevant = item["expected_answer"]

        start_time = time.time()
        docs = retriever.get_relevant_documents(question)
        latency = time.time() - start_time
        latencies.append(latency)

        retrieved_texts = [d.page_content for d in docs]
        if not retrieved_texts:
            total_precision.append(0)
            total_recall.append(0)
            mrr_scores.append(0)
            bert_f1_scores.append(0)
            continue

        emb_expected = eval_embedder.encode([relevant])
        emb_retrieved = eval_embedder.encode(retrieved_texts)

        sims = cosine_similarity(emb_expected, emb_retrieved)[0]
        ranked_indices = np.argsort(sims)[::-1]

        relevant_indices = [i for i, sim in enumerate(sims) if sim > 0.6]
        if relevant_indices:
            rank = relevant_indices[0]
            mrr_scores.append(1 / (rank + 1))
        else:
            mrr_scores.append(0)

        precision = recall = 1 if np.max(sims) > 0.6 else 0
        total_precision.append(precision)
        total_recall.append(recall)

        bert_result = bertscore.compute(
            predictions=[retrieved_texts[0]],
            references=[relevant],
            lang="en"
        )
        bert_f1 = bert_result["f1"][0]
        bert_f1_scores.append(bert_f1)

        print(f"\nQ: {question}")
        print(f"Top Similarity: {np.max(sims):.3f}")
        print(f"MRR: {mrr_scores[-1]:.3f}")
        print(f"BERTScore (F1): {bert_f1:.3f}")
        print(f"Latency: {latency:.3f}s")

    print("\n--- Retriever Evaluation Summary ---")
    print(f"Precision@{k}: {np.mean(total_precision):.3f}")
    print(f"Recall@{k}: {np.mean(total_recall):.3f}")
    print(f"Mean Reciprocal Rank: {np.mean(mrr_scores):.3f}")
    print(f"Average BERTScore (F1): {np.mean(bert_f1_scores):.3f}")
    print(f"Latency p95: {np.percentile(latencies, 95):.3f}s")

    sns.histplot(latencies, kde=True)
    plt.title("Retriever Latency Distribution")
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.show()

    return {
        "precision": np.mean(total_precision),
        "recall": np.mean(total_recall),
        "mrr": np.mean(mrr_scores),
        "bertscore_f1": np.mean(bert_f1_scores),
        "latency_p95": np.percentile(latencies, 95)
    }

def build_bm25_retriever(docs):
    tokenized_corpus = [nltk.word_tokenize(doc.page_content.lower()) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

def retrieve_bm25(bm25, tokenized_corpus, query, top_k=3):
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [tokenized_corpus[i] for i in top_indices], [scores[i] for i in top_indices]


def hybrid_retrieval(vectorstore, bm25, tokenized_corpus, query, top_k=3, alpha=0.5):
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    dense_docs = retriever.get_relevant_documents(query)

    dense_embeddings = eval_embedder.encode([d.page_content for d in dense_docs])
    query_emb = eval_embedder.encode([query])
    dense_sims = cosine_similarity(query_emb, dense_embeddings)[0]

    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-9)
    dense_norm = (dense_sims - np.min(dense_sims)) / (np.ptp(dense_sims) + 1e-9)
    hybrid_score = alpha * bm25_norm[:len(dense_norm)] + (1 - alpha) * dense_norm

    ranked_indices = np.argsort(hybrid_score)[::-1][:top_k]
    return [dense_docs[i] for i in ranked_indices], hybrid_score[ranked_indices]

eval_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def compare_retrieval_with_scores(docs, eval_data, k=3, alpha=0.5):
    tokenized_docs = [nltk.word_tokenize(d.page_content.lower()) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever_dense = vectorstore.as_retriever(search_kwargs={"k": k})

    results_summary = []

    for item in eval_data:
        query = item["question"]
        expected = item["expected_answer"]

        tokenized_query = nltk.word_tokenize(query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:k]
        sparse_texts = [docs[i].page_content for i in top_indices]
        sparse_sim = max([cosine_similarity(
            eval_embedder.encode([expected]),
            eval_embedder.encode([text])
        )[0][0] for text in sparse_texts])

        dense_docs = retriever_dense.get_relevant_documents(query)
        dense_texts = [d.page_content for d in dense_docs]
        dense_sim = max([cosine_similarity(
            eval_embedder.encode([expected]),
            eval_embedder.encode([text])
        )[0][0] for text in dense_texts])

        hybrid_texts = sparse_texts[:1] + dense_texts[:2]
        hybrid_sim = max([cosine_similarity(
            eval_embedder.encode([expected]),
            eval_embedder.encode([text])
        )[0][0] for text in hybrid_texts])

        results_summary.append({
            "query": query,
            "sparse_sim": sparse_sim,
            "dense_sim": dense_sim,
            "hybrid_sim": hybrid_sim,
        })

    print("\n--- Retrieval Scores ---")
    for r in results_summary:
        print(f"Query: {r['query']}")
        print(f"  Sparse BM25 Similarity: {r['sparse_sim']:.3f}")
        print(f"  Dense FAISS Similarity: {r['dense_sim']:.3f}")
        print(f"  Hybrid Similarity: {r['hybrid_sim']:.3f}")
        print("-" * 60)

    return results_summary

def main():
    print("Loading PDFs...")
    docs = load_pdfs(PDF_FOLDER, PDF_FILES)
    print(f"Loaded {len(docs)} document pages")

    print(" Chunking + indexing...")
    vectorstore = build_vectorstore(docs)
    print("FAISS index ready")

    print("Spinning up local LLM pipeline...")
    llm = build_llm()

    print("Building RetrievalQA chain...")
    qa_chain = build_qa_chain(vectorstore, llm)

    scores = compare_retrieval_with_scores(docs, EVAL_DATA)

    print("\nRunning evaluation on test questions...")
    evaluate_retriever(vectorstore, EVAL_DATA)
    evaluate_qa_system(qa_chain, EVAL_DATA)

    print("\nComparing retrieval methods...")

    bm25, tokenized_corpus = build_bm25_retriever(docs)
    bm25_docs, _ = retrieve_bm25(bm25, tokenized_corpus, "What is the Solar Homes program?", top_k=3)
    print(f"Sparse BM25 retrieved {len(bm25_docs)} docs")

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    dense_docs = dense_retriever.get_relevant_documents("What is the Solar Homes program?")
    print(f"Dense FAISS retrieved {len(dense_docs)} docs")

    hybrid_docs, _ = hybrid_retrieval(vectorstore, bm25, tokenized_corpus, "What is the Solar Homes program?", top_k=3)
    print(f"Hybrid retrieved {len(hybrid_docs)} docs")

    def ask_question(query: str) -> str:
        if not query or not query.strip():
            return "Please enter a question "
        try:
            return qa_chain.run(query).strip()
        except Exception as e:
            return f"Error: {e}"
    iface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask about Solar Energy Policies and Programs..."
    ),
    outputs=gr.Textbox(
        lines=15,          
        max_lines=20,      
        label="Answer"   
    ),
    title="EcoWatt AI ⚡",
    description="Ask about Solar Energy Policies and Programs...",
    flagging_mode="manual",
    flagging_options=["Save Response"],
    css="""                               
        footer {display: none !important;}
        #footer {display: none !important;}
    """

)

    print("Launching Gradio at http://127.0.0.1:7860 ...")
    iface.launch(share=False)

if __name__ == "__main__":
    main()
