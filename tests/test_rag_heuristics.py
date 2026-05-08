from bot.rag import (
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
    def _fake_chat(*_args, **_kwargs):
        return {"message": {"content": "Resposta local de teste"}}

    from bot import rag as rag_module

    sid = "pytest-opinion"
    original_chat = rag_module.ollama.chat
    rag_module.ollama.chat = _fake_chat
    try:
        text, sources = get_answer(sid, "Achas que a UTAD é boa?")
    finally:
        rag_module.ollama.chat = original_chat

    assert text == "Resposta local de teste"
    assert isinstance(sources, list)
    assert sources and sources[0]["category"] == "LocalKnowledge"


def test_get_answer_uses_ollama_model_from_env(monkeypatch):
    captured = {}

    def _fake_chat(*, model, messages):
        captured["model"] = model
        captured["messages"] = messages
        return {"message": {"content": "ok"}}

    from bot import rag as rag_module

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setattr(rag_module.ollama, "chat", _fake_chat)

    text, _sources = get_answer("pytest-model-env", "Olá, consegues ajudar-me?")
    assert text == "ok"
    assert captured["model"] == "llama3.1:8b"
    assert captured["messages"][0]["role"] == "system"
    assert "INFORMAÇÃO UTAD" in captured["messages"][0]["content"]


def test_get_answer_keeps_session_history(monkeypatch):
    def _fake_chat(*, model, messages):
        # Confirma que o histórico da sessão é passado para a chamada seguinte.
        has_previous_turn = any(
            m.get("role") == "assistant" and m.get("content") == "primeira resposta"
            for m in messages
        )
        if has_previous_turn:
            return {"message": {"content": "segunda resposta"}}
        return {"message": {"content": "primeira resposta"}}

    from bot import rag as rag_module

    monkeypatch.setattr(rag_module.ollama, "chat", _fake_chat)

    sid = "pytest-history"
    text1, _ = get_answer(sid, "Primeira pergunta")
    text2, _ = get_answer(sid, "Segunda pergunta")

    assert text1 == "primeira resposta"
    assert text2 == "segunda resposta"


def test_get_answer_returns_fallback_when_ollama_fails(monkeypatch):
    def _raise_error(*_args, **_kwargs):
        raise RuntimeError("ollama offline")

    from bot import rag as rag_module

    monkeypatch.setattr(rag_module.ollama, "chat", _raise_error)
    text, sources = get_answer("pytest-fallback", "Pergunta qualquer")
    assert "Não tenho essa informação" in text
    assert sources and sources[0]["title"] == "Skill.md"

def test_get_answer_rejects_empty_message():
    try:
        get_answer("pytest-empty", "   ")
        assert False, "Expected ValueError for empty message"
    except ValueError as exc:
        assert "não pode estar vazia" in str(exc).lower()


def test_get_answer_contextual_entry_grade_uses_last_course():
    sid = "pytest-context-course"
    text1, _ = get_answer(sid, "vou entrar em engenharia informatica este ano")
    text2, _ = get_answer(sid, "qual é a média de entrada do curso que referi?")

    assert isinstance(text1, str) and text1
    assert "engenharia informática" in text2.lower()
    assert "não tenho a média de entrada" in text2.lower()


def test_get_answer_candidaturas_returns_direct_answer():
    sid = "pytest-candidaturas"
    text, _sources = get_answer(sid, "Como me candidatar à UTAD?")
    assert "portal nacional dges" in text.lower()
    assert "documentos gerais" in text.lower()


def test_get_answer_contacts_is_deterministic():
    sid = "pytest-contactos"
    text, _ = get_answer(sid, "Como contactar os serviços académicos?")
    assert "serviços académicos" in text.lower()
    assert "sautad@utad.pt" in text.lower()


def test_get_answer_propinas_is_deterministic():
    sid = "pytest-propinas"
    text, _ = get_answer(sid, "Quanto custa a propina?")
    assert "propinas" in text.lower()
    assert "259 350 049" in text


def test_get_answer_sanitizes_llm_style(monkeypatch):
    from bot import rag as rag_module

    def _fake_chat(*, model, messages):
        return {
            "message": {
                "content": (
                    "Parabéns pela sua decisão! Se tiver alguma dúvida, não hesite em perguntar. "
                    "Boa sorte nos seus estudos!"
                )
            }
        }

    monkeypatch.setattr(rag_module.ollama, "chat", _fake_chat)
    text, _ = get_answer("pytest-style", "Fala-me da vida universitária em geral")
    assert "não tenho essa informação" in text.lower()


def test_get_answer_context_followup_mentions_previous_course():
    sid = "pytest-context-followup-v3"
    _first, _ = get_answer(sid, "Quero entrar em Engenharia Informática")
    second, _ = get_answer(sid, "e a média de entrada desse curso?")
    assert "engenharia informática" in second.lower()
