import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.evaluate_local import judge_response as judge_response_local
from backend.evaluate_upgrade import judge_response as judge_response_upgrade, run_suite


class TestEvalFailureHandling(unittest.IsolatedAsyncioTestCase):
    async def test_judge_response_failure_returns_judge_error(self):
        """Confirm that when judge API raises an exception, result is judge_error and not accurate."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Simulated judge API failure"))

        # Test evaluate_upgrade judge_response
        result_upgrade = await judge_response_upgrade(
            client=mock_client,
            query="What is the B.Tech fee?",
            answer="The fee is 50,000 INR per year.",
            sources=["doc_chunk_1"],
        )
        self.assertEqual(result_upgrade.get("judge_error"), 1)
        self.assertEqual(result_upgrade.get("accurate"), 0)
        self.assertEqual(result_upgrade.get("is_hallucination"), 0)

        # Test evaluate_local judge_response
        result_local = await judge_response_local(
            client=mock_client,
            query="What is the B.Tech fee?",
            answer="The fee is 50,000 INR per year.",
            sources=["doc_chunk_1"],
        )
        self.assertEqual(result_local.get("judge_error"), 1)
        self.assertEqual(result_local.get("accurate"), 0)
        self.assertEqual(result_local.get("is_hallucination"), 0)

    async def test_eval_suite_judge_failure_not_scored_as_accurate(self):
        """Confirm that in run_suite, judge failure increments judge_errors, not accurate_answers or scored."""
        mock_judge_client = MagicMock()
        mock_judge_client.chat.completions.create = AsyncMock(side_effect=Exception("Judge API timeout"))

        mock_http_response = MagicMock()
        mock_http_response.raise_for_status = MagicMock()
        mock_http_response.json = MagicMock(return_value={"answer": "Some answer", "sources": ["chunk1"]})

        mock_http_client = MagicMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.post = AsyncMock(return_value=mock_http_response)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            res = await run_suite(["test query 1", "test query 2"], "Test Mock Suite", mock_judge_client)

        self.assertEqual(res["total"], 2)
        self.assertEqual(res["judge_errors"], 2)
        self.assertEqual(res["scored"], 0)
        self.assertEqual(res["accuracy"], 0.0)
        self.assertEqual(res["request_errors"], 0)

    async def test_eval_suite_request_failure_recorded_as_request_error(self):
        """Confirm that request failure/timeout is counted as request_error and not silently dropped."""
        mock_judge_client = MagicMock()

        mock_http_client = MagicMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.post = AsyncMock(side_effect=Exception("Connection refused / Timeout"))

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            res = await run_suite(["failing query 1"], "Test Mock Suite", mock_judge_client)

        self.assertEqual(res["total"], 1)
        self.assertEqual(res["request_errors"], 1)
        self.assertEqual(res["scored"], 0)
        self.assertEqual(res["judge_errors"], 0)
        self.assertEqual(res["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
