#  Simple Multilingual Retrieval-Augmented Generation (RAG) System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer **Bangla-language questions** from a scanned textbook — *HSC26 Bangla 1st Paper*. It combines OCR, text cleaning, vector search, and large language models to enable intelligent question answering.

Features
- ✅ **Bangla OCR** from scanned PDFs using Tesseract
- ✅ **Text cleaning** with typo correction, fuzzy matching, and formatting
- ✅ **Chunking + Vectorization** using LaBSE embeddings
- ✅ **Long-Term Memory** using FAISS for semantic search
- ✅ **Short-Term Memory** using LangChain conversational buffer
- ✅ **Answer generation** using OpenAI GPT models
- ✅ **Supports Bangla & English queries**

Tools, Libraries, and Packages Used

| Library / Tool                     | Purpose                                                              |
|--------------------------------- |--------------------------------------------------------- |
| pytesseract                        | OCR for Bangla text from PDF images           |
| PyMuPDF (fitz)                  | PDF to image conversion                                 |
| rapidfuzz                            | Fuzzy text correction for OCR errors              |
| LangChain                          | Conversational RAG pipeline and memory   |
| sentence-transformers   | Bangla-compatible embeddings (LaBSE)        |
| FAISS                                  | Fast similarity search over document chunks|
| `OpenAI GPT`                    | Language model for answer generation         |
| `unstructured`                  | Optional PDF/text preprocessing                     |
Sample Queries and Outputs:
Bangla Queries:
| প্রশ্ন                                                                                              | উত্তর          |
| ------------------------------------------------------------------------   | ---------        |
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?                    | শম্ভুনাথ। |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে।     |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?                     | ১৫ বছর।    |

English Queries:

| Question                                                                                | Answer              |
| ---------------------------------------------------------------------- | ------------            |
| Who is considered the ideal man by Anupam?              | Shambhunath. |
| Who did Anupam refer to as his lucky god?                   | His uncle.          |
| What was Kalyani’s real age at the time of marriage? | 15 years.            |





1.	What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

[PyMuPDF (fitz)]
o	Purpose: To read pages and extract images (if the PDF is image-based or scanned).
o	Reason: It supports efficient rendering and image extraction from scanned PDFs.
2.	[Tesseract OCR with Bangla Language Support]
o	Purpose: Optical Character Recognition to extract Bangla text from scanned images.
o	Reason: Tesseract is one of the best open-source OCR engines and supports Bangla (ben) language natively.
3.	[Pillow]
o	Purpose: Image handling (converting PDF pages into images).
o	Reason: Works well with PyMuPDF and is required for Tesseract input.
Why These Choices?
•	Bangla academic books are usually scanned PDFs (image-based), which cannot be parsed by traditional text extractors (like pdfminer or PyPDF2).
•	PyMuPDF + Tesseract is a proven stack for OCR on non-English texts, including Bangla.
•	The combination allows precise control over page extraction, resolution, and language-aware text recognition.

 Formatting Challenges Faced:
Yes, several formatting issues arose due to the nature of scanned academic PDFs:
1.	 Noisy OCR Output:
o	Many OCR errors in Bangla characters (e.g., "ি" misplaced, or conjunct letters misread).
o	Example: “শিক্ষার্থী” might be misread as “শচ়কষি” or “শিখগ্গাী”.
2.	 Table and MCQ Layout Disruption:
o	Original layout like multiple-choice options (ক, খ, গ, ঘ) got merged or misaligned.
o	Solution: Used regex rules and Unicode normalization to restructure the lines.
3.	 Line Breaks and Hyphenation:
o	Broken lines and improper sentence segmentation.
o	Addressed using custom cleanup scripts and tokenization methods.
4.	 Punctuation and Noise Tokens:
o	Irregular Bangla punctuation and unwanted characters (e.g., page headers, numbering).
o	Cleaned using re (regex) and unicodedata normalization.

2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

We used a line-based chunking strategy, grouping approximately 100 lines per chunk after OCR and cleaning.
Why This Strategy?
•	Scanned Bangla PDFs often lack clear paragraph boundaries and have inconsistent sentence segmentation.
•	Chunking by fixed line count ensures:
o	Uniform chunk sizes.
o	Better compatibility with FAISS and embedding models.
o	Reduced risk of breaking semantic meaning mid-question or mid-option (especially for MCQs).
Why It Works for Semantic Retrieval
•	Embedding models (like sentence-transformers) perform well when input chunks are dense with related information.
•	Larger chunks preserve context (e.g., questions with their answer options or adjacent explanations).
•	This improves semantic similarity matching during retrieval, as related concepts are likely to be in the same chunk.

3. What embedding model did you use? Why did you choose it? How does it     capture the meaning of the text?

Embedding Model Used
We used sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 from the Sentence Transformers library.
Why This Model?
•	Multilingual Support: Works well for both Bangla and English, making it ideal for bilingual RAG applications.
•	Compact & Fast: It's lightweight (MiniLM-based), so it’s efficient for use in Colab with limited resources.
•	Semantic Understanding: Fine-tuned on a large dataset for semantic similarity tasks, which makes it well-suited for dense retrieval.

How It Captures Meaning
This model converts each chunk of text into a high-dimensional semantic vector that encodes meaning beyond just keywords. It considers:
•	Word context within the sentence.
•	Cross-lingual alignment (helps retrieve relevant Bangla content using English queries and vice versa).
•	Sentence-level semantics, making it ideal for question-answer matching.
As a result, even if the query and text don't share exact words, relevant chunks can still be retrieved based on their semantic similarity.

4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

How Queries Are Compared to Stored Chunks
We use cosine similarity between the query embedding and stored chunk embeddings in a FAISS vector store.
Why Cosine Similarity?
•	Well-suited for high-dimensional embeddings like those from sentence-transformers.
•	Measures the angle between vectors (not magnitude), which is ideal for comparing semantic closeness of text.
•	Efficient and widely used in retrieval-based NLP applications.

Why FAISS for Storage?
•	Optimized for fast nearest-neighbor search on large-scale embeddings.
•	Handles dense vector indexing efficiently, even on low-resource environments like Google Colab.
•	Easy integration with tools like LangChain and HuggingFace.

 End-to-End Flow
1.	Text Chunk ➜ Embedding (via SentenceTransformer)
2.	Store embeddings in FAISS index
3.	Query ➜ Embedding ➜ FAISS similarity search (cosine)
4.	Top-k most similar chunks returned for response generation
This setup ensures fast, semantically accurate retrieval, which is critical for multilingual QA on OCR-extracted Bangla content.

5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

Ensuring Meaningful Comparison
To ensure meaningful comparison between the user query and document chunks, we apply:
1.	Semantic-Embeddings
We use a multilingual sentence-transformer (paraphrase-multilingual-MiniLM-L12-v2) that captures context and intent, not just keywords. This allows matching even when surface words differ.
2.	Context-Preserving-Chunking
Chunks are created in line-based blocks (~100 lines), ensuring enough surrounding context (e.g., questions + options or explanations) stays together.
3.	Preprocessing-&Normalization
OCR noise, irregular spacing, and punctuation are cleaned so that embeddings are generated from readable and semantically clear text.

 What Happens If the Query Is Vague or Missing Context?
If the user query is vague (e.g., "Explain this" or "What is it?"):
•	The semantic embedding may lack strong signal, resulting in less accurate retrieval.
•	The RAG system might retrieve general or unrelated chunks based on weak similarity.

Possible Solutions (Future Enhancements)
•	Chat History Tracking: Use previous turns to add context.
•	Query Rewriting: Prompt user for clarification or apply automatic rephrasing.
•	Top-k Re-ranking: Add a second scoring layer based on keyword overlap or metadata.

By combining semantic models, context-rich chunks, and text cleaning, we ensure robust matching in most real-world queries—even across Bangla and English.

6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

Do the Results Seem Relevant?
Yes, for most queries, the retrieved results are semantically relevant and contextually accurate—especially when the question is clear and matches content in the document.
The combination of:
•	OCR-cleaned Bangla text
•	Line-based chunking (~100 lines)
•	Multilingual sentence embeddings
•	FAISS + cosine similarity
results in good performance in retrieving related content from the scanned HSC Bangla 1st Paper textbook.

 When Results Are Less Relevant
Some issues can still arise in cases like:
•	Vague or overly short queries
   Lack enough semantic signal for matching
•	OCR errors in chunks
   Misleading context for embedding
•	Concepts spread across multiple chunks
   Partial match or fragmented retrieval

Potential Improvements
To further improve relevance:
1.	Better Chunking Strategy
o	Try semantic or paragraph-based chunking (with sentence tokenization).
o	Dynamically adjust chunk size to retain logical unit (e.g., full question + options).
2.	Stronger Embedding Models
o	Upgrade to sentence-transformers/paraphrase-multilingual-mpnet-base-v2 or bge-m3 for richer context understanding.
3.	Hybrid Retrieval
o	Combine dense (semantic) + sparse (keyword/BM25) retrieval to improve coverage.
4.	Metadata Filtering
o	Tag chunks with section headers (e.g., "Prose", "Poetry", "MCQ") for filtered search.

In summary, the current results are quite good, but advanced chunking, more powerful embeddings, and hybrid retrieval could significantly improve accuracy and robustness—especially for noisy or ambiguous inputs.

