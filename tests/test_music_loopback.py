"""Tests for the MLX server loopback guard in src.music.

These cover the security invariant that the MLX integration only ever talks to
a loopback host (so it cannot be turned into an SSRF primitive). The cases use
IP literals or a mocked resolver, so no real DNS/network access is required.
"""

from unittest import mock

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

    def test_default_is_pinned_ipv4(self) -> None:
        assert music._mlx_base_url() == f"http://127.0.0.1:{music.MLX_SERVER_PORT}"

    def test_ipv6_is_bracketed(self, monkeypatch) -> None:
        monkeypatch.setattr(music, "_MLX_LOOPBACK_IP", "::1")
        assert music._mlx_base_url() == f"http://[::1]:{music.MLX_SERVER_PORT}"
