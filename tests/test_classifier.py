from pathlib import Path

import pytest

from afterseed3 import (
    Classification,
    Classifier,
    FileCandidate,
    MediaType,
    TorrentContext,
)


class TestClassifier:
    """Test cases for the Classifier class."""

    @pytest.fixture
    def classifier(self):
        """Create a Classifier instance for testing."""
        return Classifier()

    @pytest.fixture
    def sample_context(self):
        """Create a sample TorrentContext for testing."""
        return TorrentContext(
            torrent_name="Test.Torrent.2023.1080p",
            save_path=Path("/downloads"),
            content_path=Path("/downloads/test.mkv"),
            category="movies",
            tags=[],
            info_hash="TESTHASH",
        )

    def test_parse_movie_name_basic(self, classifier):
        """Test basic movie name parsing."""
        filename = "The.Matrix.1999.1080p.BluRay.x264-GROUP"
        result = classifier.parse_movie_name(filename)

        assert result is not None
        assert result["title"] == "The Matrix"
        assert result["year"] == 1999
        assert result["quality"] == "1080p"

    def test_parse_movie_name_no_year(self, classifier):
        """Test movie parsing without year."""
        filename = "Some.Old.Movie.1080p.BluRay"
        result = classifier.parse_movie_name(filename)

        assert result is not None
        assert result["title"] == "Some Old Movie"
        assert result["year"] is None
        assert result["quality"] == "1080p"

    def test_parse_tv_name_season_episode(self, classifier):
        """Test TV show parsing with season and episode."""
        filename = "Breaking.Bad.S01E01.Pilot.1080p.WEB-DL"
        result = classifier.parse_tv_name(filename)

        assert result is not None
        assert result["show"] == "Breaking Bad"
        assert result["season"] == 1
        assert result["episode"] == 1
        assert result["episode_title"] == "Pilot"

    def test_parse_tv_name_season_pack(self, classifier):
        """Test TV season pack parsing."""
        filename = "The.Office.S03.720p.HDTV"
        result = classifier.parse_tv_name(filename)

        assert result is not None
        assert result["show"] == "The Office"
        assert result["season"] == 3
        assert result["episode"] is None
        assert result["episode_title"] is None

    def test_classify_movie(self, classifier, sample_context):
        """Test full classification of a movie."""
        candidate = FileCandidate(
            source_path=Path("/downloads/The.Matrix.1999.1080p.mkv"),
            relative_path=Path("The.Matrix.1999.1080p.mkv"),
            size=1000000,
            extension=".mkv",
            is_video=True,
        )

        result = classifier.classify(candidate, sample_context)

        assert result.media_type == MediaType.MOVIE
        assert result.detected_title == "The Matrix"
        assert result.detected_year == 1999
        assert result.resolution == "1080p"

    def test_classify_tv_episode(self, classifier, sample_context):
        """Test full classification of a TV episode."""
        candidate = FileCandidate(
            source_path=Path("/downloads/Breaking.Bad.S01E01.mkv"),
            relative_path=Path("Breaking.Bad.S01E01.mkv"),
            size=500000,
            extension=".mkv",
            is_video=True,
        )

        context = TorrentContext(
            torrent_name="Breaking.Bad.S01E01",
            save_path=Path("/downloads"),
            content_path=Path("/downloads/Breaking.Bad.S01E01.mkv"),
            category="tv",
            tags=[],
            info_hash="TESTHASH",
        )
        result = classifier.classify(candidate, context)

        assert result.media_type == MediaType.EPISODE
        assert result.detected_title == "Breaking Bad"
        assert result.season == 1
        assert result.episode == 1

    def test_clean_title_string(self, classifier):
        """Test title cleaning functionality."""
        # Test dots and underscores
        assert (
            classifier._clean_title_string("The.Matrix.Reloaded")
            == "The Matrix Reloaded"
        )

        # Test brackets removal
        assert classifier._clean_title_string("Movie.Name.(2023)") == "Movie Name"

        # Test multiple spaces compression
        assert classifier._clean_title_string("Movie  Name   Here") == "Movie Name Here"

        # Test trailing dash
        assert classifier._clean_title_string("Movie Name -") == "Movie Name"
