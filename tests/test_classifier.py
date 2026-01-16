from pathlib import Path

import pytest

from afterseed3 import Classifier, FileCandidate, MediaType, TorrentContext


class TestClassifier:
    """Test cases for the Classifier class."""

    @pytest.fixture
    def classifier(self):
        """Create a Classifier instance."""
        return Classifier()

    @pytest.fixture
    def sample_context(self):
        """Sample torrent context for testing."""
        return TorrentContext(
            torrent_name="Test.Torrent.2023.1080p",
            save_path=Path("/downloads"),
            category="movies",
            tags=[],
            content_path=Path("/downloads/test.mkv"),
        )

    def test_parse_movie_name_basic(self, classifier):
        """Test basic movie name parsing."""
        result = classifier.parse_movie_name("The.Matrix.1999.1080p.BluRay.x264-GROUP")
        assert result is not None
        assert result["title"] == "The Matrix"
        assert result["year"] == 1999
        assert result["quality"] == "1080p"

    def test_parse_movie_name_no_year(self, classifier):
        """Test movie parsing without year."""
        result = classifier.parse_movie_name("Some.Movie.1080p.BluRay.x264")
        assert result is not None
        assert result["title"] == "Some Movie"
        assert result["year"] is None
        assert result["quality"] == "1080p"

    def test_parse_tv_name_season_episode(self, classifier):
        """Test TV show parsing with season and episode."""
        result = classifier.parse_tv_name("Breaking.Bad.S01E01.Pilot.1080p.WEB-DL")
        assert result is not None
        assert result["show"] == "Breaking Bad"
        assert result["season"] == 1
        assert result["episode"] == 1
        assert result["episode_title"] == "Pilot"

    def test_parse_tv_name_season_pack(self, classifier):
        """Test TV season pack parsing."""
        result = classifier.parse_tv_name("The.Office.S03.720p.HDTV.x264")
        assert result is not None
        assert result["show"] == "The Office"
        assert result["season"] == 3
        assert result["episode"] is None

    def test_classify_movie(self, classifier, sample_context):
        """Test full classification of a movie."""
        candidate = FileCandidate(
            source_path=Path("/downloads/The.Matrix.1999.mkv"),
            relative_path=Path("The.Matrix.1999.mkv"),
            size=1000000,
            extension=".mkv",
            is_video=True,
        )

        context = TorrentContext(
            torrent_name="The.Matrix.1999.1080p.BluRay.x264",
            save_path=Path("/downloads"),
            content_path=Path("/downloads/The.Matrix.1999.mkv"),
            category="movies",
            tags=[],
        )

        result = classifier.classify(candidate, context)

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
        )

        result = classifier.classify(candidate, context)

        assert result.media_type == MediaType.EPISODE
        assert result.detected_title == "Breaking Bad"
        assert result.season == 1
        assert result.episode == 1

    def test_clean_title_string(self, classifier):
        """Test title string cleaning."""
        assert classifier._clean_title_string("The.Matrix.1999") == "The Matrix 1999"
        assert (
            classifier._clean_title_string("Breaking.Bad.S01E01")
            == "Breaking Bad S01E01"
        )
        assert classifier._clean_title_string("Show.Name.2020.") == "Show Name 2020"
