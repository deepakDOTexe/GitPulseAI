
import ssl
import urllib.request

def disable_ssl_verification():
    """Disable SSL verification for model downloads."""
    # Create unverified SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Install globally
    urllib.request.install_opener(
        urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context)
        )
    )
    
    # Also disable for requests library
    import requests
    requests.packages.urllib3.disable_warnings()
    
    print("✅ SSL verification disabled for this session")

# Call this before importing sentence-transformers
disable_ssl_verification()
