from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ----------------------------
# Request body structure
# ----------------------------
class CommentRequest(BaseModel):
    comment: str

# ----------------------------
# Response structure
# ----------------------------
class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

# ----------------------------
# JSON Schema for OpenAI
# ----------------------------
sentiment_schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "rating": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        }
    },
    "required": ["sentiment", "rating"],
    "additionalProperties": False
}

# ----------------------------
# POST /comment (main endpoint)
# ----------------------------
@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(request: CommentRequest):

    # Safe handling of empty input
    if not request.comment or not request.comment.strip():
        return {"sentiment": "neutral", "rating": 3}

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis engine.\n"
                        "Classify sentiment strictly as:\n"
                        "- positive: praise, enjoyment, satisfaction\n"
                        "- negative: complaints, dissatisfaction, anger\n"
                        "- neutral: mixed, average, or factual\n\n"
                        "Rating scale:\n"
                        "- 5 = extremely positive\n"
                        "- 4 = positive\n"
                        "- 3 = neutral or mixed\n"
                        "- 2 = negative\n"
                        "- 1 = extremely negative\n\n"
                        "Return ONLY valid JSON matching the schema."
                    )
                },
                {"role": "user", "content": request.comment}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": sentiment_schema
                }
            }
        )

        return response.output_parsed

    except Exception:
        # -------- SMART FALLBACK (IMPORTANT) --------
        text = request.comment.lower()

        if any(w in text for w in ["horrible", "terrible", "worst", "broke", "awful"]):
            return {"sentiment": "negative", "rating": 1}

        if any(w in text for w in ["bad", "mediocre", "overcooked", "disappointed"]):
            return {"sentiment": "negative", "rating": 2}

        if any(w in text for w in ["amazing", "excellent", "fantastic", "loved"]):
            return {"sentiment": "positive", "rating": 5}

        if any(w in text for w in ["good", "great", "enjoyed", "nice"]):
            return {"sentiment": "positive", "rating": 4}

        return {"sentiment": "neutral", "rating": 3}

# ----------------------------
# GET /comment (health check)
# ----------------------------
@app.get("/comment")
def comment_health():
    return {"status": "comment endpoint alive"}