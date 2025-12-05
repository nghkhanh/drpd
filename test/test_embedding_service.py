from drpd.core.embeddings import encoder
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.mark.parametrize(
    "text",
    [
        "deep research",
        "xin chao cac ban",
    ],
)
@patch("drpd.core.embeddings.requests.post")
def test_embeding_text_using_url(mock_post, text):
    """Test embedding text using a mocked remote URL."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]}
        ]
    }
    mock_post.return_value = mock_response

    results = encoder.embed(text)

    assert isinstance(results, np.ndarray)
    assert results.shape == (1, 3)
    mock_post.assert_called_once()