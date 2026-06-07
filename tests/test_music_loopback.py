"""Tests for the MLX server loopback guard in src.music.

These cover the security invariant that the MLX integration only ever talks to
a loopback host (so it cannot be turned into an SSRF primitive). The cases use
IP literals or a mocked resolver, so no real DNS/network access is required.
"""

from unittest import mock

import pytest

import src.music as music


class TestIsLoopbackHost:
    def test_canonical_literals_accepted(self) -> None:
        assert music._is_loopback_host("127.0.0.1") is True
        assert music._is_loopback_host("localhost") is True
        assert music._is_loopback_host("::1") is True

    def test_literals_are_normalized(self) -> None:
        assert music._is_loopback_host("  127.0.0.1  ") is True
        assert music._is_loopback_host("LOCALHOST") is True

    def test_loopback_range_ip_accepted(self) -> None:
        # 127.0.0.0/8 is all loopback; resolved as a numeric literal (no DNS).
        assert music._is_loopback_host("127.0.0.2") is True

    def test_non_loopback_ip_rejected(self) -> None:
        assert music._is_loopback_host("8.8.8.8") is False
        assert music._is_loopback_host("10.0.0.5") is False
        assert music._is_loopback_host("192.168.1.1") is False

    def test_resolution_failure_fails_closed(self) -> None:
        with mock.patch.object(
            music.socket, "getaddrinfo", side_effect=music.socket.gaierror
        ):
            assert music._is_loopback_host("not-a-real-host.example") is False

    def test_mixed_resolution_rejected(self) -> None:
        # If a name resolves to BOTH loopback and a public IP, reject it.
        infos = [
            (music.socket.AF_INET, 0, 0, "", ("127.0.0.1", 0)),
            (music.socket.AF_INET, 0, 0, "", ("93.184.216.34", 0)),
        ]
        with mock.patch.object(music.socket, "getaddrinfo", return_value=infos):
            assert music._is_loopback_host("dual.example") is False

    def test_all_loopback_resolution_accepted(self) -> None:
        infos = [(music.socket.AF_INET, 0, 0, "", ("127.0.0.5", 0))]
        with mock.patch.object(music.socket, "getaddrinfo", return_value=infos):
            assert music._is_loopback_host("alias.example") is True

    def test_default_host_constant_is_loopback(self) -> None:
        # Default MLX_SERVER_HOST is 127.0.0.1, resolved once at import.
        assert music._MLX_HOST_IS_LOOPBACK is True


class TestResolveLoopbackIp:
    """The pinned-IP resolver (closes the DNS-rebinding window)."""

    def test_literal_ip_returned_as_is(self) -> None:
        assert music._resolve_loopback_ip("127.0.0.1") == "127.0.0.1"
        # The whole 127.0.0.0/8 block is loopback (RFC 1122), not just .1.
        assert music._resolve_loopback_ip("127.0.0.9") == "127.0.0.9"
        assert music._resolve_loopback_ip("::1") == "::1"

    def test_localhost_pins_to_127_without_dns(self) -> None:
        assert music._resolve_loopback_ip("localhost") == "127.0.0.1"
        assert music._resolve_loopback_ip("  LOCALHOST ") == "127.0.0.1"

    def test_non_loopback_returns_none(self) -> None:
        assert music._resolve_loopback_ip("8.8.8.8") is None
        assert music._resolve_loopback_ip("192.168.1.1") is None

    def test_hostname_pins_resolved_loopback_ip(self) -> None:
        infos = [(music.socket.AF_INET, 0, 0, "", ("127.0.0.7", 0))]
        with mock.patch.object(music.socket, "getaddrinfo", return_value=infos):
            assert music._resolve_loopback_ip("alias.example") == "127.0.0.7"

    def test_hostname_with_any_public_ip_returns_none(self) -> None:
        # Mixed resolution (rebinding-prone) is rejected entirely.
        infos = [
            (music.socket.AF_INET, 0, 0, "", ("127.0.0.1", 0)),
            (music.socket.AF_INET, 0, 0, "", ("8.8.8.8", 0)),
        ]
        with mock.patch.object(music.socket, "getaddrinfo", return_value=infos):
            assert music._resolve_loopback_ip("rebind.example") is None

    def test_resolution_failure_returns_none(self) -> None:
        with mock.patch.object(
            music.socket, "getaddrinfo", side_effect=music.socket.gaierror
        ):
            assert music._resolve_loopback_ip("nope.invalid") is None


class TestMlxBaseUrl:
    """base_url is built from the pinned IP, with IPv6 bracketing."""

    def test_default_is_pinned_ipv4(self, monkeypatch) -> None:
        # Pin the import-time constant so the assertion is deterministic
        # regardless of how the runner resolves loopback.
        monkeypatch.setattr(music, "_MLX_LOOPBACK_IP", "127.0.0.1")
        assert music._mlx_base_url() == f"http://127.0.0.1:{music.MLX_SERVER_PORT}"

    def test_ipv6_is_bracketed(self, monkeypatch) -> None:
        monkeypatch.setattr(music, "_MLX_LOOPBACK_IP", "::1")
        assert music._mlx_base_url() == f"http://[::1]:{music.MLX_SERVER_PORT}"

    def test_raises_without_pinned_ip(self, monkeypatch) -> None:
        # Fail fast instead of silently targeting 127.0.0.1 if the loopback
        # guard is ever bypassed.
        monkeypatch.setattr(music, "_MLX_LOOPBACK_IP", None)
        with pytest.raises(RuntimeError):
            music._mlx_base_url()


class TestMlxRequest:
    """The centralized blocking HTTP helper (loopback-pinned, nosem'd)."""

    def test_returns_status_and_body(self, monkeypatch) -> None:
        import urllib.request

        monkeypatch.setattr(music, "_MLX_LOOPBACK_IP", "127.0.0.1")
        resp = mock.MagicMock()
        resp.status = 200
        resp.read.return_value = b"hello"
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = resp
        with mock.patch.object(urllib.request, "urlopen", return_value=ctx) as uo:
            status, body = music._mlx_request("/api/models", 1.0)
        assert status == 200
        assert body == b"hello"
        # Connects to the pinned loopback base URL, never a re-resolved name.
        sent_req = uo.call_args.args[0]
        assert sent_req.full_url == f"{music._mlx_base_url()}/api/models"

    def test_urlerror_propagates_as_oserror(self) -> None:
        import urllib.error
        import urllib.request

        with mock.patch.object(
            urllib.request, "urlopen", side_effect=urllib.error.URLError("boom")
        ):
            with pytest.raises(OSError):
                music._mlx_request("/api/models", 1.0)

    def test_http_error_propagates(self) -> None:
        # urllib raises HTTPError (an OSError subclass) for non-2xx responses,
        # so a 5xx surfaces as an exception, not a (status, body) tuple — the
        # caller's `except OSError` handles it.
        import urllib.error
        import urllib.request

        err = urllib.error.HTTPError(
            "http://127.0.0.1/api/models", 503, "Service Unavailable", {}, None
        )
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            with pytest.raises(urllib.error.HTTPError):
                music._mlx_request("/api/models", 1.0)


class TestGetCapabilities:
    """The sync capability probe uses _mlx_request directly (no to_thread)."""

    def test_mlx_available_true_on_200(self, monkeypatch) -> None:
        monkeypatch.setattr(music, "_MLX_HOST_IS_LOOPBACK", True)
        monkeypatch.setattr(music, "_mlx_request", lambda path, timeout: (200, b""))
        assert music.get_capabilities()["mlx"] is True

    def test_mlx_unavailable_on_exception(self, monkeypatch) -> None:
        monkeypatch.setattr(music, "_MLX_HOST_IS_LOOPBACK", True)

        def _boom(path, timeout):
            raise OSError("connection refused")

        monkeypatch.setattr(music, "_mlx_request", _boom)
        assert music.get_capabilities()["mlx"] is False
