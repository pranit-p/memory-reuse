"""Unit tests for memory_reuse.config semantic-cache additions."""

from __future__ import annotations

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.exceptions import ConfigurationError, InvalidTTLError


class TestSemanticDefaults:
    def test_semantic_off_by_default(self) -> None:
        config = CacheConfig()
        assert config.semantic_enabled is False
        assert config.embedding_provider is None
        assert config.embedding_model is None

    def test_default_similarity_threshold(self) -> None:
        assert CacheConfig().similarity_threshold == 0.95

    def test_default_max_vectors_and_store_flag(self) -> None:
        config = CacheConfig()
        assert config.max_vectors_per_namespace == 10_000
        assert config.store_exact_on_semantic_hit is True


class TestThresholdValidation:
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.95, 1.0])
    def test_valid_threshold_accepted(self, threshold: float) -> None:
        config = CacheConfig(similarity_threshold=threshold)
        assert config.similarity_threshold == threshold

    @pytest.mark.parametrize("threshold", [-0.01, 1.01, -1.0, 2.0])
    def test_out_of_range_threshold_raises(self, threshold: float) -> None:
        with pytest.raises(ConfigurationError):
            CacheConfig(similarity_threshold=threshold)

    def test_existing_ttl_validation_still_works(self) -> None:
        with pytest.raises(InvalidTTLError):
            CacheConfig(default_ttl=0)


class TestMissingProviderGuard:
    def test_semantic_enabled_without_provider_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            CacheConfig(semantic_enabled=True)

    def test_semantic_enabled_with_provider_ok(self) -> None:
        config = CacheConfig(semantic_enabled=True, embedding_provider="local")
        assert config.semantic_enabled is True
        assert config.embedding_provider == "local"

    def test_provider_without_semantic_enabled_ok(self) -> None:
        config = CacheConfig(embedding_provider="openai")
        assert config.semantic_enabled is False
        assert config.embedding_provider == "openai"


class TestFromEnv:
    def test_defaults_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in [
            "MEMORY_REUSE_SEMANTIC_ENABLED",
            "MEMORY_REUSE_SIMILARITY_THRESHOLD",
            "MEMORY_REUSE_EMBEDDING_PROVIDER",
            "MEMORY_REUSE_EMBEDDING_MODEL",
        ]:
            monkeypatch.delenv(var, raising=False)
        config = CacheConfig.from_env()
        assert config.semantic_enabled is False
        assert config.similarity_threshold == 0.95
        assert config.embedding_provider is None
        assert config.embedding_model is None

    def test_reads_semantic_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_REUSE_SEMANTIC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_REUSE_SIMILARITY_THRESHOLD", "0.8")
        monkeypatch.setenv("MEMORY_REUSE_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("MEMORY_REUSE_EMBEDDING_MODEL", "text-embedding-3-small")
        config = CacheConfig.from_env()
        assert config.semantic_enabled is True
        assert config.similarity_threshold == pytest.approx(0.8)
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"

    @pytest.mark.parametrize("raw", ["false", "0", "no", "FALSE"])
    def test_semantic_enabled_falsey_values(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_REUSE_SEMANTIC_ENABLED", raw)
        monkeypatch.delenv("MEMORY_REUSE_EMBEDDING_PROVIDER", raising=False)
        config = CacheConfig.from_env()
        assert config.semantic_enabled is False

    def test_env_out_of_range_threshold_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_REUSE_SIMILARITY_THRESHOLD", "1.5")
        monkeypatch.delenv("MEMORY_REUSE_SEMANTIC_ENABLED", raising=False)
        with pytest.raises(ConfigurationError):
            CacheConfig.from_env()

    def test_env_semantic_without_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_REUSE_SEMANTIC_ENABLED", "true")
        monkeypatch.delenv("MEMORY_REUSE_EMBEDDING_PROVIDER", raising=False)
        with pytest.raises(ConfigurationError):
            CacheConfig.from_env()
