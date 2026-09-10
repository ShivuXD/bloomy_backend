import os
import json
from pathlib import Path
import joblib
import pandas as pd
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


# ============================================================
# LOAD ML MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

skill_model = joblib.load(BASE_DIR / "skill_classifier.pkl")

FEATURE_NAMES = [
    "accuracy",
    "reaction_time",
    "hesitation",
    "retries"
]


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Bloomy AI Service")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:4173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:4173",
        "https://bloomly-frontend.vercel.app",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class GameStats(BaseModel):
    accuracy: float = Field(..., ge=0, le=100)
    reaction_time: float = Field(..., ge=0)
    hesitation: int = Field(..., ge=0)
    retries: int = Field(..., ge=0)
    child_name: str = "Friend"


class QuestionChoice(BaseModel):
    text: str
    is_correct: bool


class AdaptiveQuestionRequest(BaseModel):
    difficulty_level: int = Field(..., ge=1, le=10)
    skill_level: str = "Intermediate"
    age: int = Field(default=8, ge=4, le=18)
    question_type: str = "phonics"
    child_name: str = "Friend"


class AdaptiveQuestion(BaseModel):
    instruction: str
    question: str
    choices: list[QuestionChoice]
    peco_dialogue: str


# ============================================================
# PECO FALLBACK
# ============================================================

def fallback_message(
    skill_level: str,
    child_name: str
) -> str:

    if skill_level == "Beginner":
        return f"It's okay, {child_name}. Let's try together!"

    if skill_level == "Advanced":
        return f"Wow, {child_name}! You did amazing!"

    return f"Great effort, {child_name}! Keep going!"


# ============================================================
# PECO GEMINI MESSAGE
# ============================================================

def generate_peco_message(
    skill_level: str,
    accuracy: float,
    retries: int,
    child_name: str
) -> str:

    if client is None:
        return fallback_message(skill_level, child_name)

    prompt = f"""
You are Peco, a warm and supportive learning companion for children.

Give exactly one short encouraging response.

Rules:
- Maximum 2 short sentences.
- Use simple child-friendly language.
- Never mention diagnosis, disorder, ADHD, ASD, dyslexia, medical conditions,
  intelligence, or mental health.
- Do not shame the child.
- Do not mention percentages or technical data.
- Encourage trying again or celebrating success.

Child name: {child_name}
Skill level: {skill_level}
Accuracy: {accuracy}%
Retries: {retries}
"""

    try:
        result = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = result.text.strip()

        if text:
            return text

    except Exception as error:
        print("Gemini Peco error:", error)

    return fallback_message(skill_level, child_name)


# ============================================================
# LOCAL FALLBACK QUESTIONS
# ============================================================

def fallback_question(
    difficulty_level: int,
    question_type: str
) -> dict:

    # --------------------------------------------------------
    # PHONICS FALLBACK
    # --------------------------------------------------------

    if question_type == "phonics":

        if difficulty_level <= 3:
            return {
                "instruction": "Listen carefully and choose the right answer.",
                "question": "Which word starts with B?",
                "choices": [
                    {
                        "text": "Bear",
                        "is_correct": True
                    },
                    {
                        "text": "Cat",
                        "is_correct": False
                    }
                ],
                "peco_dialogue": "Listen carefully. Which word starts with B?"
            }

        elif difficulty_level <= 6:
            return {
                "instruction": "Choose the word that starts with the sound B.",
                "question": "Which word begins with B?",
                "choices": [
                    {
                        "text": "Ball",
                        "is_correct": True
                    },
                    {
                        "text": "Dog",
                        "is_correct": False
                    },
                    {
                        "text": "Fish",
                        "is_correct": False
                    }
                ],
                "peco_dialogue": "Which word begins with B? Take your time and choose the best answer."
            }

        else:
            return {
                "instruction": "Choose the word that begins with the same sound.",
                "question": "Which word starts with the same sound as Butterfly?",
                "choices": [
                    {
                        "text": "Banana",
                        "is_correct": True
                    },
                    {
                        "text": "Tiger",
                        "is_correct": False
                    },
                    {
                        "text": "Moon",
                        "is_correct": False
                    },
                    {
                        "text": "Sun",
                        "is_correct": False
                    }
                ],
                "peco_dialogue": "Listen for the beginning sound in Butterfly. Which word starts with the same sound?"
            }

    # --------------------------------------------------------
    # GENERIC FALLBACK
    # --------------------------------------------------------

    return {
        "instruction": "Choose the best answer.",
        "question": "Which answer is correct?",
        "choices": [
            {
                "text": "Yes",
                "is_correct": True
            },
            {
                "text": "No",
                "is_correct": False
            }
        ],
        "peco_dialogue": "Take your time. Which answer is correct?"
    }


# ============================================================
# CLEAN GEMINI JSON
# ============================================================

def clean_json_response(text: str) -> str:

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ============================================================
# VALIDATE QUESTION
# ============================================================

def validate_question(
    question_data: dict,
    difficulty_level: int
) -> bool:

    if not isinstance(question_data, dict):
        return False

    if "instruction" not in question_data:
        return False

    if "question" not in question_data:
        return False

    if "choices" not in question_data:
        return False

    if "peco_dialogue" not in question_data:
        return False

    if not isinstance(question_data["peco_dialogue"], str):
        return False

    if not question_data["peco_dialogue"].strip():
        return False

    choices = question_data["choices"]

    if not isinstance(choices, list):
        return False

    # Difficulty controls number of choices
    if difficulty_level <= 3:
        expected_choices = 2
    elif difficulty_level <= 6:
        expected_choices = 3
    else:
        expected_choices = 4

    if len(choices) != expected_choices:
        return False

    correct_count = 0

    for choice in choices:

        if not isinstance(choice, dict):
            return False

        if "text" not in choice:
            return False

        if "is_correct" not in choice:
            return False

        if not isinstance(choice["text"], str):
            return False

        if not isinstance(choice["is_correct"], bool):
            return False

        if choice["is_correct"]:
            correct_count += 1

    # Exactly ONE correct answer
    if correct_count != 1:
        return False

    return True


# ============================================================
# GENERATE ADAPTIVE QUESTION
# ============================================================

def generate_adaptive_question(
    difficulty_level: int,
    skill_level: str,
    age: int,
    question_type: str,
    child_name: str
) -> dict:

    # --------------------------------------------------------
    # FALLBACK IF GEMINI IS NOT AVAILABLE
    # --------------------------------------------------------

    if client is None:

        print("Gemini unavailable -> using fallback question")

        return fallback_question(
            difficulty_level,
            question_type
        )

    # --------------------------------------------------------
    # CHOICE COUNT BASED ON DIFFICULTY
    # --------------------------------------------------------

    if difficulty_level <= 3:
        choice_count = 2

    elif difficulty_level <= 6:
        choice_count = 3

    else:
        choice_count = 4

    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the adaptive question generator for Bloomy.

Bloomy is a child-friendly learning application.

Generate ONE educational question.

Learner information:
- Age: {age}
- Current skill level: {skill_level}
- Difficulty level: {difficulty_level}/10
- Question type: {question_type}

Rules:

1. The question must be appropriate for a child around the given age.
2. The question must match the difficulty level.
3. Generate exactly {choice_count} answer choices.
4. Exactly ONE choice must be correct.
5. The other choices must be plausible but wrong.
6. Keep wording short and simple.
7. Do not mention diagnosis, disorders, ADHD, ASD, dyslexia,
   intelligence, mental health, or medical conditions.
8. Do not make the question dependent on cultural knowledge.
9. Do not use emojis.
10. Return ONLY valid JSON.
11. Do NOT wrap the JSON in markdown.

Required JSON format:

{{
  "instruction": "Short instruction for the child",
  "question": "The question",
  "choices": [
    {{
      "text": "Choice 1",
      "is_correct": true
    }},
    {{
      "text": "Choice 2",
      "is_correct": false
    }}
  ],
  "peco_dialogue": "A short natural sentence Peco can say aloud that directly presents the same question."
}}

The "peco_dialogue" MUST be based directly on the generated instruction and question.
It must not introduce a different question or different facts.
Keep it natural and child-friendly.
Make the question genuinely appropriate for difficulty {difficulty_level}/10.
"""

    try:

        response = client.models.generate_content(model=MODEL_NAME,contents=prompt)
        raw_text = response.text.strip()

        cleaned_text = clean_json_response(raw_text)

        question_data = json.loads(cleaned_text)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if validate_question(
            question_data,
            difficulty_level
        ):

            return question_data

        print("Gemini returned invalid question -> fallback")

    except Exception as error:

        print("Gemini question generation error:", error)

    # --------------------------------------------------------
    # SAFE FALLBACK
    # --------------------------------------------------------

    return fallback_question(
        difficulty_level,
        question_type
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "gemini_configured": client is not None
    }


# ============================================================
# ML PREDICTION
# ============================================================

@app.post("/predict")
def predict(stats: GameStats):

    data = pd.DataFrame(
        [[
            stats.accuracy,
            stats.reaction_time,
            stats.hesitation,
            stats.retries
        ]],
        columns=FEATURE_NAMES
    )

    # --------------------------------------------------------
    # 1. ML predicts current skill
    # --------------------------------------------------------

    skill_level = str(
        skill_model.predict(data)[0]
    )

    # --------------------------------------------------------
    # 2. Convert skill -> adaptive difficulty
    # --------------------------------------------------------

    if skill_level == "Beginner":

        difficulty_level = 3

    elif skill_level == "Advanced":

        difficulty_level = 8

    else:

        difficulty_level = 5

    # --------------------------------------------------------
    # 3. Peco response
    # --------------------------------------------------------

    peco_message = generate_peco_message(
        skill_level,
        stats.accuracy,
        stats.retries,
        stats.child_name
    )

    # --------------------------------------------------------
    # 4. Peco state
    # --------------------------------------------------------

    if skill_level == "Beginner" or stats.accuracy < 45:

        peco_state = "comforting"

    elif skill_level == "Advanced":

        peco_state = "proud"

    else:

        peco_state = "encouraging"

    # --------------------------------------------------------
    # 5. RETURN AUTHORITATIVE ADAPTIVE RESULT
    # --------------------------------------------------------

    return {
        "skill_level": skill_level,
        "difficulty_level": difficulty_level,
        "peco_state": peco_state,
        "peco_message": peco_message
    }


# ============================================================
# GEMINI ADAPTIVE QUESTION ENDPOINT
# ============================================================

@app.post("/generate-question")
def generate_question(
    request: AdaptiveQuestionRequest
):

    question = generate_adaptive_question(
        difficulty_level=request.difficulty_level,
        skill_level=request.skill_level,
        age=request.age,
        question_type=request.question_type,
        child_name=request.child_name
    )

    return question


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )
