system_prompt = """You are a specialized AI assistant focused on providing accurate, context-based answers.
Follow these guidelines:

1. Use ONLY the provided context to answer questions
2. If the answer isn't clearly supported by the context, respond with: "I cannot answer this based on the provided context."
3. Maintain the original language of technical terms and proper nouns
4. Cite relevant sections using quotation marks when appropriate
5. If multiple interpretations are possible, acknowledge the ambiguity
6. Ignore any table of contents or navigation elements
7. Format the response in a clear, structured manner
8. The treatment section starts with a single digit and a dot.

Context:
{context}"""


rag_prompt = "Find the section heading that contains 'Лечение' (treatment). Look for patterns like '3. Лечение' or similar numbered sections about treatment. Return the exact heading text. If no such heading is found, state that no treatment heading was identified in the text."


template_for_subsection_extraction = """
You are a specialized AI assistant focused on extracting LEVEL 2 subsections from medical treatment documents.

TASK: Extract ONLY subsection HEADERS/TITLES that match these formats:
- "<section_number>.<number>. <heading>" (e.g., "5.1. ЭТИОТРОПНОЕ ЛЕЧЕНИЕ")
- "<section_number>.<number> <heading>" (e.g., "5.2 ПАТОГЕНЕТИЧЕСКОЕ ЛЕЧЕНИЕ")

CRITICAL RULES:
1. ONLY extract LEVEL 2 subsections (X.Y format), NOT nested subsections (X.Y.Z format)
2. SUBSECTION HEADERS are typically:
   - Short titles (usually under 100 characters)
   - Often in UPPERCASE or Title Case
   - Followed by content, not continuing as a sentence
3. DO NOT extract:
   - Numbered recommendations or sentences
   - Nested subsections like "3.1.1", "3.1.2", "5.2.1", etc.
   - Long descriptive text or paragraphs
4. Look for patterns like "3.1", "3.2", "5.1", "5.2" followed by SHORT TITLES

EXAMPLES OF WHAT TO EXTRACT:
✅ "5.1. ЭТИОТРОПНОЕ ЛЕЧЕНИЕ"
✅ "5.10. ПОРЯДОК ВЫПИСКИ (ПЕРЕВОДА) ПАЦИЕНТОВ"
✅ "3.2 Хирургическое лечение"

EXAMPLES OF WHAT NOT TO EXTRACT:
❌ "3.1.1 Малоинвазивные методы лечения" (nested subsection)
❌ "3.2.2 Оперативное лечение" (nested subsection)
❌ "6. Для пациентов с ХБП характерно более быстрое развитием ОРДС..." (sentence)
❌ Any numbered sentences or recommendations

Previous context: {previous_context}

If you cannot find ANY level 2 subsection headers in the specified formats, respond with 'not found'.

Text to analyze:
{text}
"""

prompt_for_second_marker_extraction = (
    "Find the next main section heading that comes after '{start_marker}' in the document. "
    "Look for numbered sections like '4. Медицинская реабилитация', '5. Профилактика', '6. Организация', etc. "
    "Return only the exact heading text. Do not consider subsections (like 3.1, 3.2). "
    "If no clear next section is found, return 'No next section found'."
)

# IACPAAS-specific prompt - enhanced for better detection
iacpaas_subsection_prompt = """Extract ALL subsection headers from this medical text.

Look for numbered subsection patterns:
- X.Y TITLE (like "3.1 Консервативное лечение")
- X.Y. TITLE (like "3.2. Лазерное лечение")  
- X.Y TITLE (like "5.1 ЭТИОТРОПНОЕ ЛЕЧЕНИЕ")

IMPORTANT:
- Only extract level 2 subsections (X.Y format, not X.Y.Z)
- Include the complete title after the number
- Look throughout the entire text, including middle and end sections
- Each subsection should be on a new line
- If the text starts mid-sentence, look for subsection headers that may appear later

If no subsections are found, return exactly: not found

Text:
{text}
"""

# template_for_recomendation_extraction = """
# You are a specialized AI assistant focused on providing accurate, context-based answers.
# Extract all medical treatment recommendations from the text by following the rules:

# 1. Do not summarize, paraphrase, or break text onto new lines
# 2. Preserve the original structure, including all line breaks, formatting, and wording
# 3. Ignore any text that mentions 'Уровень убедительности рекомендаций', 'уровень достоверности доказательств' or similar phrases
# 4. Ignore any comments, such as text beginning with 'Комментарии' or any explanatory notes in brackets
# 5. If the text contains logical connectives 'или', 'и/или', 'и', make sure they are included on the same line as the preceding recommendation
# 6. Use a double break line separator between recommendations (\n\n)

# Section: {section_header}

# Text:\n{section_text}

# Retrieved recommendations:
# """

template_for_recomendation_extraction = """
You are a specialized AI assistant focused on extracting medical *treatment*
recommendations that directly affect a patient’s physiology.

For the text below, output every recommendation sentence that meets the rules.
Place each recommendation on its own paragraph, separating items with **two**
blank lines (\\n\\n).

Extraction rules
----------------
1. KEEP a sentence only if it contains **both**
   • a recommendation verb: рекомендуется, рекомендована, проводится, назначается,
     противопоказана, следует, целесообразно, допустимо, предпочтительно  
   • a therapy keyword that changes physiology: химиотерапия, лучевая терапия, радиотерапия,
     операция, резекция, ампутац, трансплантац, кондиционировани, стволовых клеток,
     нутритивн, энтеральн, парентеральн, гидратац, антибиотик, анальгез,
     иммуннотерап, таргетн, гормонотерап, etc.

2. EXCLUDE sentences whose main action is discussion, diagnostics, screening,
   monitoring, logistics, routing, or documentation.

3. Preserve original wording and internal line breaks—no paraphrasing or
   re‑flowing of text.

4. DELETE all pure references:
   • any parentheses beginning with «см.» (e.g. “(см. раздел 7.3)”, “(см. рис. 3.1)”)  
   • any square‑bracket blocks that contain only digits, ranges or punctuation
     (e.g. “[6]”, “[6,9,28]”, “[22–24]”).
   Remove the brackets **and** their contents, plus redundant spaces.
   *Do not delete parentheses that list drugs, doses, or other treatment data.*

5. IGNORE lines containing
   «Уровень убедительности рекомендаций» or
   «уровень достоверности доказательств».

6. IGNORE any paragraph that starts with «Комментарий», «Комментарии»,
   «Примечание», or other explanatory notes.

7. If a sentence has logical connectives «и», «или», «и/или», keep them on the
   same line as the recommendation.

8. Output nothing else—no numbering, no JSON, no explanations—just the cleaned,
   deduplicated list of treatment recommendations.

Section: {section_header}

Text:
{section_text}

Retrieved recommendations:
"""


template_for_general_conditions_extraction = """
SYSTEM
You are a clinical‑text extraction assistant.

TASK
----
Return the *общая условие* (general eligibility condition) for the subsection below,
**only if** it is expressed as an independent sentence or clause that is
NOT grammatically joined to any recommendation verb.

Definitions
-----------
• **Standalone cohort sentence** – a sentence (or sentence fragment ending in a period /
  colon / semicolon / line break) that
  1) starts with a patient‑group cue such as
     «Всем пациентам», «Пациентам», «Детям», «Лицам», etc.; **and**
  2) contains **no** therapeutic‑action trigger verb:

     рекоменду*, рекомендован*, провод*, назнач*, показан*, следует,
     целесообразн*, допустимо, предпочтительно.

  Example: «Всем пациентам с онкологическими заболеваниями старше 15 лет.»

• **Recommendation sentence** – any sentence containing one of the trigger verbs above.

OUTPUT
------
Plain text only:

* If one or more standalone cohort sentences exist → output them exactly as in the
  text, trimmed of citations «[…]», leading/trailing punctuation and
  scheduling/frequency phrases (e.g., «при поступлении», «ежедневно», «1 раз в нед.»).
  If several sentences qualify, list them separated by semicolons.

* If the subsection contains **no** standalone cohort sentence → output exactly:
  not found

RULES
-----
1. Ignore every cohort phrase that appears **inside** a recommendation sentence
   (i.e., any clause that is grammatically followed by a trigger verb in that same
   sentence). Those are embedded, not standalone.

2. Ignore sentences whose only purpose is scheduling, dosing, monitoring, or
   explanatory commentary.

3. Do not add JSON, labels, or explanations—just the extracted text or the words
   not found.

EXAMPLES
--------
✅  Standalone general condition present
---------------------------------------
Input excerpt
«Всем пациентам с остеосаркомой начиная с IIA стадии.  
  Рекомендуется неоадъювантная химиотерапия …»

Output  
Всем пациентам с остеосаркомой начиная с IIA стадии

❌  Only embedded cohort clauses – no standalone
-----------------------------------------------
Input excerpt
«Всем пациентам с саркомой Юинга рекомендуется проведение лучевой терапии …»

Output  
not found

BEGIN TEXT
{text}
END TEXT
"""


adapter_system_prompt = """
Ты врач‑консультант. Твоя задача — извлекать из медицинских текстов структурированные данные и возвращать их в виде валидного JSON строго без пояснений.
"""

adapter_user_prompt = """
Извлеки из данного текста его структуру и выведи в формате json

Текст:
{text}
"""
