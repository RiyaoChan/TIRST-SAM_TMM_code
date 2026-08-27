import io

import torch

from efficient_sam.microquery_end2end import EndToEndMicroQueryHead


def test_c1_f1_f2_can_load_byte_identical_shared_initialization():
    torch.manual_seed(20260825)
    source = EndToEndMicroQueryHead()
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    states = []
    for _ in range(3):
        buffer.seek(0)
        head = EndToEndMicroQueryHead()
        head.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        states.append(head.state_dict())
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key])
        assert torch.equal(states[0][key], states[2][key])

