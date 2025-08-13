import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from document_handler import load_and_split_document

# ✅ Gemini Wrapper
class GeminiLLM(LLM):
    model: any

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-pro')

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        print("\n\n====== Prompt sent to Gemini ======\n")
        print(prompt)
        print("\n===================================\n")
        response = self.model.generate_content(prompt)

# ✅ Check for empty candidates or finish_reason
        if not response.candidates or not response.candidates[0].content.parts:
            return "No valid answer could be generated."

        return response.text  # original line 


    @property
    def _llm_type(self) -> str:
        return "gemini"

def answer_question(pdf_path, question):
    # ✅ Step 1: Load and split document
    chunks = load_and_split_document(pdf_path)

    # ✅ Step 2: Generate embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # ✅ Step 3: Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
    llm = GeminiLLM(model=model)

    # ✅ Step 4: Prompt template for better answers
    question_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a highly accurate academic assistant.
Use ONLY the rules, facts, and limits from the provided context to answer the question.
If the question involves a calculation or applying a rule to a new scenario, do the reasoning step-by-step using only the context.
Do not guess beyond the rules provided.
If the rule clearly answers the scenario, apply it and provide the result.

Context:
{context}

Question:
{question}

Answer:
"""
)


    # ✅ Step 5: Setup Retrieval QA Chain
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # You can switch to "map_reduce" for large docs
        retriever=retriever,
        chain_type_kwargs={"prompt": question_prompt}
    )

    # ✅ Step 6: Run the chain
    result = qa_chain.run(question)

    return {
        "answers": [result]
    }
