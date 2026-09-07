"""Spec for carryia/serve/rewrite.py -- the conversation transcript renderer (P0-7).

`render_transcript` shapes the prior chat turns into the plain `Player:/Coach:` block
the answer prompt prepends (so the coach has memory of the chat). Pure, no client.
"""

from carryia.serve import rewrite


HISTORY = [
    {"role": "user", "content": "Why do I keep losing lane as a support?"},
    {"role": "assistant", "content": "Ward the tri-bush earlier. (mobalytics)", "conv_id": "x"},
]


def test_render_transcript_labels_each_turn_by_role():
    out = rewrite.render_transcript(HISTORY)
    assert out == (
        "Player: Why do I keep losing lane as a support?\n"
        "Coach: Ward the tri-bush earlier. (mobalytics)"
    )


def test_render_transcript_of_nothing_is_empty():
    assert rewrite.render_transcript([]) == ""


def test_render_transcript_ignores_extra_keys():
    # message dicts carry conv_id etc.; only role/content are read.
    out = rewrite.render_transcript([{"role": "user", "content": "hi", "conv_id": "z"}])
    assert out == "Player: hi"
