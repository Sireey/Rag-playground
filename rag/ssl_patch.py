# rag/ssl_patch.py
#
# Corporate SSL interception fix.
#
# On this machine, the corporate proxy rewrites TLS certificates.
# Python's SSL verifier rejects them, causing certificate errors for
# any HTTPS call — including HuggingFace model downloads.
#
# This module applies a global monkey-patch that disables SSL verification
# at the requests.HTTPAdapter level, which catches all requests including
# redirects to CDN hosts (like HuggingFace's Xet Storage).
#
# Call apply() once at process startup (done by RAGFactory automatically
# when ssl_patch: true is set in the YAML config).

import os
import ssl
import urllib3
import requests
import requests.adapters

_applied = False  # guard against double-patching


def apply() -> None:
    """Apply the SSL verification bypass. Safe to call multiple times."""
    global _applied
    if _applied:
        return

    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    _original_send = requests.adapters.HTTPAdapter.send

    def _patched_send(self, request, **kwargs):
        kwargs["verify"] = False
        return _original_send(self, request, **kwargs)

    requests.adapters.HTTPAdapter.send = _patched_send
    _applied = True
