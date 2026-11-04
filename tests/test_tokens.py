from src.common.budget import MESSAGE_OVERHEAD_TOKENS, TOKEN_COUNTER


def test_count_messages_adds_overhead_and_counts_tool_calls():
    """Every message costs its content plus the overhead; a tool call costs its name and arguments."""
    plain = [{"role": "user", "content": "hello"}]
    assert TOKEN_COUNTER.count_messages(plain) == TOKEN_COUNTER.count("hello") + MESSAGE_OVERHEAD_TOKENS
    call = {"id": "c", "type": "function", "function": {"name": "search_wikipedia", "arguments": "{\"query\": \"x\"}"}}
    with_call = [{"role": "assistant", "content": None, "tool_calls": [call]}]
    assert TOKEN_COUNTER.count_messages(with_call) == TOKEN_COUNTER.count("search_wikipedia{\"query\": \"x\"}") + MESSAGE_OVERHEAD_TOKENS


def test_truncate_cuts_on_sentence_boundary():
    """Long extracts are cut after a full sentence and fit the limit."""
    text = "First sentence here. Second sentence here. Third sentence here."
    shortened = TOKEN_COUNTER.truncate(text, 9)
    assert shortened.endswith(".") and TOKEN_COUNTER.count(shortened) <= 9 and shortened.startswith("First")
    assert TOKEN_COUNTER.truncate(text, 100) == text
