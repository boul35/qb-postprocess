from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from afterseed3 import Notifier


class TestNotifier:
    """Test cases for the Notifier class."""

    @pytest.fixture
    def config(self):
        """Sample config with API settings."""
        return {
            "radarr": {"url": "http://localhost:7878", "key": "test_radarr_key"},
            "sonarr": {"url": "http://localhost:8989", "key": "test_sonarr_key"},
            "plex": {
                "url": "http://localhost:32400",
                "token": "test_plex_token",
                "section_id": "1",
            },
        }

    @pytest.fixture
    def notifier(self, config):
        """Create a Notifier instance."""
        return Notifier(config, dry_run=False)

    @pytest.fixture
    def dry_run_notifier(self, config):
        """Create a Notifier instance in dry run mode."""
        return Notifier(config, dry_run=True)

    def test_queue_radarr(self, notifier):
        """Test queuing Radarr notifications."""
        test_path = Path("/movies/Test Movie (2023)")
        notifier.queue_radarr(test_path)

        assert str(test_path) in notifier.pending_radarr_paths

    def test_queue_sonarr(self, notifier):
        """Test queuing Sonarr notifications."""
        test_path = Path("/tv/Test Show/Season 01")
        notifier.queue_sonarr(test_path)

        assert str(test_path) in notifier.pending_sonarr_paths

    def test_queue_plex(self, notifier):
        """Test queuing Plex notifications."""
        notifier.queue_plex()

        assert notifier.trigger_plex is True

    @patch("urllib.request.urlopen")
    def test_send_command_success(self, mock_urlopen, notifier):
        """Test successful API command sending."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        cfg = {"url": "http://test.com", "key": "test_key"}
        result = notifier._send_command(
            cfg, "TestCommand", {"param": "value"}, "Test message"
        )

        assert result is True
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_send_command_failure_then_success(self, mock_urlopen, notifier):
        """Test API command with retry logic."""
        # Mock first call fails, second succeeds
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        mock_urlopen.side_effect = [Exception("Connection failed"), mock_response]

        cfg = {"url": "http://test.com", "key": "test_key"}
        result = notifier._send_command(
            cfg, "TestCommand", {"param": "value"}, "Test message"
        )

        assert result is True
        assert mock_urlopen.call_count == 2

    @patch("urllib.request.urlopen")
    def test_send_command_all_failures(self, mock_urlopen, notifier):
        """Test API command that fails all retries."""
        mock_urlopen.side_effect = Exception("Connection failed")

        cfg = {"url": "http://test.com", "key": "test_key"}
        result = notifier._send_command(
            cfg, "TestCommand", {"param": "value"}, "Test message"
        )

        assert result is False
        assert mock_urlopen.call_count == 3  # Max retries

    @patch("urllib.request.urlopen")
    def test_notify_plex_success(self, mock_urlopen, notifier):
        """Test successful Plex notification."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        result = notifier._notify_plex()

        assert result is True

    @patch("urllib.request.urlopen")
    def test_notify_plex_unauthorized(self, mock_urlopen, notifier):
        """Test Plex notification with invalid token."""
        from urllib.error import HTTPError

        mock_error = HTTPError(None, 401, "Unauthorized", None, None)
        mock_error.read.return_value = b"Invalid token"
        mock_urlopen.side_effect = mock_error

        result = notifier._notify_plex()

        assert result is False

    def test_flush_dry_run(self, dry_run_notifier):
        """Test flush in dry run mode."""
        dry_run_notifier.queue_radarr(Path("/test/path"))
        dry_run_notifier.queue_sonarr(Path("/test/path2"))
        dry_run_notifier.queue_plex()

        results = dry_run_notifier.flush()

        assert results == {}
        assert len(dry_run_notifier.pending_radarr_paths) == 1
        assert len(dry_run_notifier.pending_sonarr_paths) == 1
        assert dry_run_notifier.trigger_plex is True

    @patch("urllib.request.urlopen")
    def test_flush_with_notifications(self, mock_urlopen, notifier):
        """Test flush with actual API calls."""
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        notifier.queue_radarr(Path("/movies/Test"))
        notifier.queue_sonarr(Path("/tv/Test"))
        notifier.queue_plex()

        results = notifier.flush()

        assert "radarr" in results
        assert "sonarr" in results
        assert "plex" in results
        assert results["radarr"] is True
        assert results["sonarr"] is True
        assert results["plex"] is True
