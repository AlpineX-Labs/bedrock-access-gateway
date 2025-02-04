from typing import Annotated

from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse
from api.setting import DEFAULT_MODEL

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


@router.post("/completions", response_model=ChatResponse | ChatStreamResponse, response_model_exclude_unset=True)
async def chat_completions(
        chat_request: Annotated[
            ChatRequest,
            Body(
                examples=[
                    {
                        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello!"},
                        ],
                    }
                ],
            ),
        ]
):
    try:
        if chat_request.model.lower().startswith("gpt-"):
            chat_request.model = DEFAULT_MODEL

        model = BedrockModel()
        model.validate(chat_request)
        
        if chat_request.stream:
            return StreamingResponse(
                content=model.chat_stream(chat_request), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        response = model.chat(chat_request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
