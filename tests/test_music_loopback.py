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
