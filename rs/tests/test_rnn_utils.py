from rs.rnn_utils import get_pos


def test_get_pos():
    assert get_pos('словаря') == {'S'}
    assert get_pos('сделался') == {'V'}
