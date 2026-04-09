A technology that paraphrases words or text so everyone—including autistic users, non‑native speakers, and busy professionals—can get the insight the words are trying to express easily.


## What Is Understand-IT?
Legal agreements, academic papers, and even everyday instructions are often written in language in overly complex and jargonish structure. Everything can be understood with simpler and understandable words, so I don't see the point in underlying complixity. This just waste productive human hours and make education less accessible, and increases barrier of entry for many people.  
 Using words and language a person is comfortable with:
 - reduces stress
 -  broadens access of information
 - Increase inclusion for people with leanring differences any problem with reading or understanding information.   

Unnecessary complexity:

- Wastes valuable hours  
- Makes education less accessible  
- Raises barriers to entry for many people  
**Understand-IT** decodes dense content—contracts, disclosures, academic papers, legal docs—into clear, accessible language.

Apache 2.0 © Abhinesh Dahal

## Model Provider Setup
The backend now defaults to OpenAI for clause extraction. Gemini remains available as an alternate provider.

Recommended default:

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Optional Gemini fallback:

```env
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

Install backend dependencies after updating `backend/requirements.txt` so the `openai` package is available.
