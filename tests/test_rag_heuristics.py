from bot.rag import (
    NAVIGATION_SCOPE_REPLY,
    get_answer,
    is_opinion_or_subjective_question,
    kb_has_section_beyond_sobre_utad,
    looks_like_basic_utad_identity_question,
)


def test_opinion_detection():
    assert is_opinion_or_subjective_question("A UTAD é uma boa universidade?")
    assert is_opinion_or_subjective_question("e boa?")
    assert is_opinion_or_subjective_question("vale a pena estudar aqui?")
    assert not is_opinion_or_subjective_question("Quando começam as aulas?")


def test_kb_beyond_sobre():
    assert not kb_has_section_beyond_sobre_utad({})
    assert not kb_has_section_beyond_sobre_utad({"sobre_utad": {}})
    assert kb_has_section_beyond_sobre_utad({"sobre_utad": {}, "cursos": {}})


def test_identity_question():
    assert looks_like_basic_utad_identity_question("Onde fica a UTAD?")
    assert looks_like_basic_utad_identity_question("o que é a utad")
    assert not looks_like_basic_utad_identity_question("Quanto custa a propina?")


def test_get_answer_opinion_short_circuits_without_llm():
    sid = "pytest-opinion"
    text, sources = get_answer(sid, "Achas que a UTAD é boa?")
    assert "opini" in text.lower() or "não dá" in text.lower() or "não dou" in text.lower()
    assert sources == []


def test_get_answer_off_topic_navigation_reply(monkeypatch):
    """Sem Chroma, só sobre_utad na KB, pergunta fora do âmbito → mensagem de âmbito."""
    monkeypatch.setattr("bot.rag.get_document_count", lambda: 0)
    monkeypatch.setattr("bot.rag.vector_search", lambda *a, **k: [])
    monkeypatch.setattr(
        "bot.rag._select_kb_sections",
        lambda _msg: {"sobre_utad": {"sigla": "UTAD"}},
    )

    sid = "pytest-nav"
    text, sources = get_answer(sid, "Qual é o menu da cantina de Marte no planeta Júpiter?")
    assert text == NAVIGATION_SCOPE_REPLY
    assert sources == []
