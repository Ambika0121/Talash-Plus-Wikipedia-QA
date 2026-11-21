
import re
from functools import lru_cache
from transformers import pipeline


# ---------------------------------------
# LOAD FLAN-T5 MODEL
# ---------------------------------------

@lru_cache(maxsize=1)
def get_qa_pipeline():

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )


# ---------------------------------------
# STOPWORDS
# ---------------------------------------

STOPWORDS = {
    "what", "which", "when", "where", "who", "why",
    "how", "the", "is", "are", "was", "were", "a",
    "an", "and", "or", "of", "in", "on", "to", "for",
    "with", "from", "by", "it", "its", "this", "that",
    "these", "those", "can", "could", "would", "should",
    "explain", "describe", "give", "tell", "about",
    "main", "using", "used", "use", "define"
}


def keywords(text):

    words = set(
        re.findall(r"\b[a-z]{2,}\b", text.lower())
    )

    return words - STOPWORDS


# ---------------------------------------
# EXTRACT TEXT FROM FAISS RESULTS
# ---------------------------------------

def extract_contexts(contexts):

    extracted = []

    for item in contexts:

        if isinstance(item, dict):

            text = item.get("text", "")

        else:

            text = str(item)

        if text.strip():
            extracted.append(text.strip())

    return extracted


# ---------------------------------------
# SELECT RELEVANT SENTENCES
# ---------------------------------------

def select_relevant_sentences(question, contexts):

    question_lower = question.lower()

    question_words = keywords(question)

    all_text = " ".join(contexts)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        all_text
    )

    scored_sentences = []

    # Special handling for AI applications
    if "application" in question_lower:

        for sentence in sentences:

            if (
                re.search(r"\bapplications?\b", sentence, re.I)
                and re.search(
                    r"\bAI\b|artificial intelligence",
                    sentence,
                    re.I
                )
            ):

                return [sentence.strip()]

    # Special handling for AGI questions
    if (
        "agi" in question_lower
        or "artificial general intelligence" in question_lower
    ):

        for sentence in sentences:

            if re.search(
                r"artificial general intelligence|\bAGI\b",
                sentence,
                re.I
            ):

                scored_sentences.append(
                    (10, sentence.strip())
                )

    # General keyword-based sentence selection
    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 25:
            continue

        sentence_words = keywords(sentence)

        score = len(
            question_words.intersection(sentence_words)
        )

        # Give additional importance to matching key phrases
        if "computer vision" in question_lower:
            if "computer vision" in sentence.lower():
                score += 5

        if "robotics" in question_lower:
            if "robot" in sentence.lower() or "robotics" in sentence.lower():
                score += 3

        if score >= 2:

            scored_sentences.append(
                (score, sentence)
            )

    # Sort by relevance
    scored_sentences.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    seen = set()

    for score, sentence in scored_sentences:

        if sentence not in seen:

            selected.append(sentence)
            seen.add(sentence)

        if len(selected) >= 4:
            break

    return selected


# ---------------------------------------
# GENERATE ANSWER
# ---------------------------------------

def get_answer(question, contexts):

    if not contexts:
        return "No relevant information was found."

    try:

        # Extract actual text from FAISS results
        clean_contexts = extract_contexts(contexts)

        if not clean_contexts:

            return "No usable Wikipedia information was found."

        # Select relevant sentences
        selected_sentences = select_relevant_sentences(
            question,
            clean_contexts
        )

        if not selected_sentences:

            return (
                "The available Wikipedia information "
                "is insufficient to answer this question."
            )

        context = "\n".join(selected_sentences)

        # Load FLAN-T5
        generator = get_qa_pipeline()

        prompt = f"""
You are Talash+, an accurate Wikipedia research assistant.

Answer the question using ONLY the context.

Rules:
1. Answer the exact question.
2. Use simple English.
3. Give a complete and meaningful answer.
4. Include examples only when available in the context.
5. Do not mix unrelated topics.
6. Do not repeat the context unnecessarily.
7. If the answer is missing, say the information is insufficient.

Context:
{context}

Question:
{question}

Answer:
"""

        result = generator(
            prompt,
            max_new_tokens=150,
            num_beams=4,
            do_sample=False,
            truncation=True
        )

        answer = result[0]["generated_text"].strip()

        # Validate generated answer
        answer_words = keywords(answer)
        context_words = keywords(context)

        common_words = answer_words.intersection(
            context_words
        )

        if (
            len(answer.split()) < 6
            or len(common_words) < 2
        ):

            return (
                "Based on the retrieved Wikipedia information:\n\n"
                + "\n\n".join(selected_sentences)
            )

        return answer

    except Exception as exc:

        return f"Answer generation failed: {exc}"
