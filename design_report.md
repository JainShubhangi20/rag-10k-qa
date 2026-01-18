# Design Report - RAG System for 10-K Documents

## What I Built

I built a question-answering system that can answer questions about Apple and Tesla's annual reports (10-K filings). You ask it a question, it finds relevant parts from the PDFs, and then uses an AI to generate an answer.

## How It Works (Simple Version)

'''
User Question → Find Similar Text → Re-rank Results → Generate Answer
'''

Basically:
1. Load the PDFs and break them into small pieces
2. Convert those pieces into numbers (embeddings)
3. When someone asks a question, find the most similar pieces
4. Feed those pieces to an AI model to write an answer

## The Parts I Used

### 1. Reading PDFs

I used 'pdfplumber' to read the PDFs because it's good at reading tables. SEC filings have a lot of tables with financial data, so this was important.

The code goes through each page, pulls out the text, and also extracts any tables it finds.

### 2. Breaking Text Into Chunks

I couldn't just feed the whole PDF to the AI - it's way too long. So I broke it into smaller pieces:
- Each chunk is about 1000 characters
- There's 200 characters of overlap between chunks (so we don't lose context at the edges)

I also kept track of which page and section each chunk came from. This way I can tell the user where the answer came from.

Total chunks: ~1000 (about 500 from each PDF)

### 3. Converting Text to Numbers (Embeddings)

I used a model called 'BAAI/bge-small-en-v1.5'. It turns text into a list of 384 numbers. Similar texts get similar numbers.

Why this model? It's small (~130MB) but still good quality. Bigger models would be better but slower.

### 4. Storing and Searching (Vector Store)

I used FAISS to store all the embeddings. When you ask a question:
1. Convert the question to numbers
2. Find the chunks with the most similar numbers
3. Return the top matches

FAISS is really fast at this - it can search through thousands of chunks in milliseconds.

### 5. Re-ranking

The first search is fast but not super accurate. So I added a second step:
1. Get top 20 results from FAISS
2. Use another model to re-score them (this model is smarter but slower)
3. Keep only the top 3

The re-ranker model is 'cross-encoder/ms-marco-MiniLM-L-6-v2'. It's specifically trained to decide if a text answers a question.

### 6. Generating the Answer

Finally, I use 'TinyLlama-1.1B-Chat' to write the answer. I give it:
- The question
- The top 3 chunks I found
- Instructions to only use those chunks

It then writes an answer based on what it read.

Why TinyLlama? It's small enough to run on a regular GPU but still writes decent answers.

## The Prompt I Used

'''
<|system|>
Answer using ONLY the context. Cite as ["Doc", "Section", "p. X"].
Not found: "Not specified in the document."
Unanswerable: "This question cannot be answered based on the provided documents."</s>
<|user|>
Context:
{the chunks I found}

Question: {user's question}</s>
<|assistant|>
'''

I kept it short because TinyLlama only has room for 2048 tokens total.

## Handling Bad Questions

Some questions can't be answered from these documents. I check for:
- Questions about the future ("stock price in 2025")
- Questions about things not in the report ("what color is the building")
- Questions about years after the report was filed

If I detect these, I just return "This question cannot be answered" without even trying.

## Results

The system answers most questions correctly:

| Question | Answer |
|----------|--------|
| Apple's revenue in 2024? | $391,035 million ✓ |
| Tesla's revenue in 2023? | $96,773 million ✓ |
| Tesla vehicles produced? | Model S, X, 3, Y, Cybertruck, Semi ✓ |
| Stock price in 2025? | "Cannot be answered" ✓ |

## What Could Be Better

1. Small context window - TinyLlama can only handle ~2000 tokens, so I can only give it 3 chunks. A bigger model could use more context.

2. Tables aren't perfect - I convert tables to text but sometimes the formatting gets messy.

3. Slow on CPU - Without a GPU, generating answers takes like 30+ seconds.

4. Simple chunking - I just split by character count. Smarter chunking could keep paragraphs together.

## Files

- The main python notebook link: https://colab.research.google.com/drive/1egq4qo_A_ENwVJE3WO1oTXQmTdLlSzGa?usp=sharing

## Libraries Used

- 'pdfplumber' - reading PDFs
- 'sentence-transformers' - embeddings
- 'faiss-cpu' - vector search
- 'transformers' - running the LLM
- 'torch' - deep learning stuff

## How To Run

1. Install the libraries (first cell)
2. Download the PDFs (second cell)
3. Run all cells in order
4. Use 'answer_question("your question here")'
