# AI Notes

## AI Used
- **SentenceTransformers (all-MiniLM-L6-v2)**: To create embeddings for uploaded text documents.
- **GPT-2 / DistilGPT-2**: To generate human-readable answers based on the context retrieved from embeddings.

## How AI Was Used
1. Uploaded documents are split into chunks and converted into embeddings.
2. User questions are also converted into embeddings.
3. Closest matching document chunk is retrieved using ChromaDB.
4. GPT-2 generates an answer using the context chunk and the user question.

## Manual Checks
- Ensured context chunks are correctly retrieved for questions.
- Verified generated answers make sense and match context.
- Adjusted prompt formatting for better GPT-2 output.
