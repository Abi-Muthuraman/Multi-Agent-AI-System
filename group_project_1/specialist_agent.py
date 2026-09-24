import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SpecialistAgent:
    """
    A Specialist Agent that uses RAG with Corrective Rag (CRAG). 
    Steps:
    1. Load corresponding documents
    2. Chunk the documents into smaller pieces
    3. Create embeddings for the chunks
    4. Store the embeddings in a FAISS index
    5. Retrieve relevant chunks based on user query
    6. Grade the retrieved chunks and select the best one
    7. Perform corrective retrieval if necessary
    8. Generate a good response with an LLM
    """
    def __init__(
            self,
            knowledge_base_path="knowledge_base",
            embedding_model_name="all-MiniLM-L6-v2",
            top_k=3,
            relevance_threshold=0.35
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        #load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        #load documents and create chunks
        self.chunks = self.load_and_chunk_documents()
        #create FAISS index and store embeddings
        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(
            texts, 
            normalize_embeddings=True
        )
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.asarray(self.embeddings, dtype=np.float32))

        #client (Groq)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=api_key)


    #1 and 2 --> load docs and chunk them
    def load_and_chunk_documents(self):
        chunks = []
        for file_path in self.knowledge_base_path.glob("*.md"):
            text = file_path.read_text(encoding="utf-8")
            #split text by headings/paragraphs instead of one huge chunk per document
            sections = text.split("\n\n")
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                #keep filename with every chunk for reference
                chunks.append({
                    "source": file_path.name,
                    "text": section
                })
        return chunks


    #3, 4, 5 --> create embeddings, store in FAISS index, retrieve relevant chunks
    def _retrieve(self, query, k):
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True
        )
        scores, indices = self.index.search(
            np.asarray(query_embedding, dtype=np.float32), 
            k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results


    #CRAG --> evaluate retrieved chunks (step 1)
    def _grade_retrieval(self, retrieved_chunks):
        """
        Figure out whether the retrieved chunks have relevant information. 
        This will use the embedding similarity score as a relevance grader (lightweight).
        """
        if not retrieved_chunks:
            return False
        relevant_chunks = [
            chunk for chunk in retrieved_chunks 
            if chunk["score"] >= self.relevance_threshold
        ]
        return len(relevant_chunks) > 0


    #CRAG --> corrective retrieval (step 2)
    def _corrective_retrieval(self, query):
        """
        If initial retrieval is weak, get more candidates.
        CRAG correction:
        1. Increase the number of retrieved chunks
        2. Combine the original query with important keywords
        3. Keep only chunks that meet the relevance threshold
        """
        expanded_query = (
            f"{query} "
            "troubleshooting symptoms resolution "
            "ticket category escalation procedures"
        )
        expanded_results = self._retrieve(
            expanded_query,
            k=min(8, len(self.chunks))
        )
        corrected_results = [
            chunk for chunk in expanded_results 
            if chunk["score"] >= self.relevance_threshold
        ]
        return corrected_results


    # context
    def _build_context(self, chunks):
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"Source: {chunk['source']}\n"
                f"Content: {chunk['text']}"
            )
        return "\n\n---\n\n".join(context_parts)


    # generate grounded response
    def _generate_answer(self, question, retrieved_chunks):
        context = self._build_context(retrieved_chunks)
        prompt = f"""
You are a specialist agent in a technical support system.
Answer the user's question using ONLY information from the provided context.
Do not make up information or procedures. 
return your answer as a valid JSON using exactly these fields:
{{
    "category": "...",
    "resolution": "...",
    "sources": ["...", "..."]
}}
Each source filename should appear only once in the sources list. 
The category must go with a Ticket Category found in the context.
The resolution should summarize the appropriate troubleshooting steps or resolution procedure. 
User request:
{question}
Known context:
{context}
"""
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grounded technical support specialist agent. "
                        "Never use information outside the provided context. "
                )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        answer = response.choices[0].message.content
        return self._parse_response(answer)


    #parse llm JSON response
    def _parse_response(self, answer):
        #remove md  formatting (fences) if model adds them
        answer = answer.strip()
        if answer.startswith("```"):
            answer = answer.replace("```json", "")
            answer = answer.replace("```", "")
            answer = answer.strip()
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            return {
                "category": "Unknown",
                "resolution": answer,
                "sources": []
            }


    #main function
    def answer(self, question):
        #initial retrieval
        retrieved = self._retrieve(
            question,
            self.top_k
        )
        #CRAG check
        retrieval_is_good = self._grade_retrieval(
            retrieved
        )
        #corrective retrieval if initial is bad
        if not retrieval_is_good:
            retrieved = self._corrective_retrieval(
                question
            )
        #no useful information found
        if not retrieved:
            return {
                "category": "Unknown",
                "resolution": (
                    "No useful relevant information was found in the provided context."
                ),
                "sources": []
            }
        #generate grounded response
        result = self._generate_answer(
            question,
            retrieved
        )
        #include the sources
        if not result.get("sources"):
            result["sources"] = list(
                dict.fromkeys(
                    chunk["source"]
                    for chunk in retrieved
                )
            )
        return result