import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from openai import OpenAI

load_dotenv()

model = OpenAIEmbeddings(model="text-embedding-3-small")
    #(SentenceTransformer('all-MiniLM-L6-v2'))
    #(llama-text-embed-v2)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_provider = "OPENAI"
pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv("PINECONE_ENVIRONMENT"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def create_embeddings(data, provider):
    if provider == 'OPENAI':
        embeds = create_embeddings_with_openai(data)
    elif provider == 'sentence_transformer':
        embeds = create_embeddings_with_st(data)
    elif provider == 'llama':
        embeds = create_embeddings_with_llama(data)

def create_embeddings_with_openai(data):
    message_data = data.get('items')
    res = openai.embeddings.create(
        input=[message_data.get('message')], model=model
    )
    embeds = [i.embedding for i in res.data]
    return embeds

if not pinecone.has_index(INDEX_NAME):
     pinecone.create_index(
         name=INDEX_NAME,
         vector_type="dense",
         dimension=1536,  # Example dimension for OpenAI's text-embedding-3-small
         metric="cosine",
         spec=ServerlessSpec(
             cloud="aws",
             region="us-east-1"
         ),
     )
index = pinecone.Index(INDEX_NAME)

def create_embeddings_with_llama(data):
    message_data = data.get('items')
    print("message_data", message_data)
    index.upsert_records(
        namespace="message_namespace",
        records=message_data
    )

async def create_embeddings_with_st(data):
    print("sample data: ", data)
    vectors_to_upsert = []
    print("data.get(items)", data.get("items"))
    for item in data.get('items'):
        print("item", item)
        unique_id = item.get('id')
        user_id = item.get('user_id')
        user_name = item.get('user_name')
        timestamp = item.get('timestamp')
        text_to_embed = item.get('message')
        if text_to_embed and unique_id and user_id and user_name:

            embedding = openai.embeddings.create(
                input=text_to_embed, model="text-embedding-3-small").data[0].embedding

            metadata = {k: v for k, v in item.items() if
                        k not in ['id']}
            print("embedding:", embedding)
            print("metadata:", metadata)
            vectors_to_upsert.append({
                'id': str(unique_id),
                'values': embedding,
                'metadata': metadata
            })
        else:
            print(f"Skipping item due to missing 'text' or 'id': {item}")
    if vectors_to_upsert:
        try:
            print("upserting vectors", vectors_to_upsert)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                print("batch", batch)
                index.upsert(vectors=batch)
            return "message: Successfully uploaded vectors to Pinecone"
        except Exception as e:
            return "error: Error upserting to Pinecone:"
    else:
        return "message:" "No valid data to upload."
    return "message: Data received successfully"


def create_rag_prompt(question, context_list):
    context_str = "\n---\n".join(context_list)
    prompt = f"""
    Use the following pieces of context to answer the user's question. 
    If you don't know the answer based *only* on the provided context, 
    simply state that you cannot find the answer in the given information.
    Do not use prior knowledge or external information.
    Context:
    ---
    {context_str}
    ---
    Question:
    {question}
    Answer:
    """
    return prompt

def generate_answer_with_llm(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-5-nano", # Or gpt-4, or other capable models
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based strictly on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=1 # Lower temperature for factual, less creative answers
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred during generation: {e}"

def get_answer(user_query: str) -> list:
    query_embedding = openai.embeddings.create(
        input=user_query, model="text-embedding-3-small").data[0].embedding
    context_list = index.query(vector=query_embedding, top_k=10, include_metadata=True)
    for match in context_list['matches']:
        print(match['metadata']['message'])
    context_chunks = [str(match['metadata']) for match in context_list['matches']]
    final_prompt = create_rag_prompt(user_query, context_chunks)
    final_answer = generate_answer_with_llm(final_prompt)
    return final_answer