# pip install fastapi uvicorn openai python-dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # Load .env file if present

openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Call the OpenAI ChatCompletion endpoint
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": req.message},
        ],
        # temperature=0.7,
    )
    # Extract assistant message
    assistant_reply = response.choices[0].message.content
    return {"reply": assistant_reply}


@app.post("/assistant")
async def assistant_endpoint(req: MessageRequest):
    # assistant = openai.beta.assistants.create(
    #     name="My Assistant",
    #     instructions="You are a helpful assistant that answers user queries.",
    #     model="gpt-4o",
    # )
    # 우리는 앞서 만든 assistant를 사용합니다.
    assistant = await openai.beta.assistants.retrieve("asst_tc4AhtsAjNJnRtpJmy1gjJOE")

    # Create a new thread with the user's message
    thread = await openai.beta.threads.create(
        messages=[{"role": "user", "content": req.message}]
    )

    # Create a run and poll until completion using the helper method
    run = await openai.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    # Get messages for this specific run
    messages = list(
        await openai.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
    )

    # Process the first message's content and annotations
    message_content = messages[0][1][0].content[0].text
    annotations = message_content.annotations
    citations = []

    # Replace annotations with citation markers and build citations list
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(
            annotation.text, f"[{index}]"
        )
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = await openai.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    # Combine message content with citations if any exist
    assistant_reply = message_content.value
    if citations:
        assistant_reply += "\n\n" + "\n".join(citations)

    return {"reply": assistant_reply}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
