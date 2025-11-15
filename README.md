### Q&A for chat

Front end/UI:
HTML, CSS, Jinja2 templates.

Back end:
python, Flask.

ML:
pinecone, openai embeddings model: "text-embedding-3-small"

System:
The service has 2 endpoints: Query, Upload.
Upload => User may invoke this endpoint to upload json file with messages. 
Query => This is invoked to submit a Question and retrieve answer based on the context from messages.

The api endpoints can be invoked from swagger docs or html form from the browser.

RAG pipeline:
Experimented with following embedding models:
1. Sentence Transformers "all-MiniLM-L6-v2".
2. llama-text-embed-v2.
3. OpenAI: "text-embedding-3-small".

The input object:
    {
      "id": "9d4b9c82-f13a-4f53-b679-729f756177b7",
      "user_id": "e35ed60a-5190-4a5f-b3cd-74ced7519b4a",
      "user_name": "Fatima El-Tahir",
      "timestamp": "2025-08-18T12:35:31.160458+00:00",
      "message": "Get me front-row seats for the ballet performance on December 9."
    }

    id field represents unique id of the message.
    user_id and user_name fields represent id and name of the user.
    timestamp and message fields represent the time and the content of the exchanged message.

  "message" field is converted into a dense vector embedding using the OpenAI embedding model, and 
  all of the fields including id, embedding and metadata(user_id, user_name, timestamp, message) is upserted into the pinecone index.
  
  A file that has been uploaded once, the embeddings are persisted in the pinecone db and need not be uploaded again.
  User is able to retrieve answers for multiple questions in the same session or a different session.
  
    


