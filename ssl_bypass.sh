#!/bin/bash
# SSL bypass for Hugging Face model downloads
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export SSL_VERIFY=false
export PYTHONHTTPSVERIFY=0

echo "✅ SSL bypass environment variables set"
echo "Run: source ssl_bypass.sh before starting your application"
