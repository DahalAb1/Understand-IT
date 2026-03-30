import os
import json
import google.generativeai as genai
from ...domain.models import ModelOutput, RiskLevel


class GeminiAdapter:
    max_input_length = 3000

    def __init__(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def simplify(self, text: str) -> ModelOutput:
        prompt = (
            "You are a legal document analyzer helping general members of the public "
            "understand legal contracts without needing a lawyer.\n\n"
            "Analyze the following clause and respond with a JSON object only, no other text.\n\n"
            "JSON format:\n"
            "{\n"
            '  "title": "name of this legal section (e.g. Indemnification, Termination)",\n'
            '  "simplified": "plain English explanation preserving legal nuance",\n'
            '  "risk_level": "low" | "medium" | "high",\n'
            '  "risk_reason": "why this clause matters to the user, or null if low risk"\n'
            "}\n\n"
            f"Clause:\n{text}"
        )

        response = self.model.generate_content(prompt)
        text_response = response.text.strip()

        if text_response.startswith("```"):
            text_response = text_response.split("```")[1]
            if text_response.startswith("json"):
                text_response = text_response[4:]

        data = json.loads(text_response)
        return ModelOutput(
            title=data["title"],
            simplified=data["simplified"],
            risk_level=RiskLevel(data["risk_level"]),
            risk_reason=data.get("risk_reason"),
        )
