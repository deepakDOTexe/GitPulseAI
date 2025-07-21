import os
import requests

# Import Google Generative AI with error handling
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Force requests to use custom certificate bundle
CUSTOM_CA_BUNDLE = "/Users/dkatre/GitPulseAI/custom_cacert.pem"

def debug_certificate_setup():
    """Debug certificate configuration"""
    print("🔍 **Certificate Debug Information**")
    print("=" * 50)
    
    # Check environment variables
    print(f"REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE', 'NOT SET')}")
    print(f"SSL_CERT_FILE: {os.environ.get('SSL_CERT_FILE', 'NOT SET')}")
    print(f"Custom bundle exists: {os.path.exists(CUSTOM_CA_BUNDLE)}")
    
    if os.path.exists(CUSTOM_CA_BUNDLE):
        size = os.path.getsize(CUSTOM_CA_BUNDLE)
        with open(CUSTOM_CA_BUNDLE, 'r') as f:
            content = f.read()
            cert_count = content.count('-----BEGIN CERTIFICATE-----')
        print(f"Custom bundle size: {size:,} bytes")
        print(f"Certificate count: {cert_count}")
    print()

def test_requests_with_custom_cert():
    """Test requests library with explicit custom certificate"""
    print("🧪 **Testing Requests with Custom Certificate**")
    print("=" * 50)
    
    # Test 1: Basic HTTPS with custom certificate
    try:
        print("Test 1: Basic HTTPS to google.com with custom cert...")
        response = requests.get('https://google.com', verify=CUSTOM_CA_BUNDLE, timeout=10)
        print(f"✅ Success: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}...")
    
    # Test 2: Google API endpoint with custom certificate
    try:
        print("Test 2: Google API endpoint with custom cert...")
        response = requests.get('https://generativelanguage.googleapis.com', verify=CUSTOM_CA_BUNDLE, timeout=10)
        print(f"✅ Success: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}...")
    
    # Test 3: Using environment variable (original approach)
    try:
        print("Test 3: Using environment variable...")
        # Force environment variable to be used
        os.environ['REQUESTS_CA_BUNDLE'] = CUSTOM_CA_BUNDLE
        response = requests.get('https://google.com', timeout=10)
        print(f"✅ Success: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}...")
    print()

def test_gemini_with_custom_cert():
    """Test Gemini API with custom certificate handling"""
    print("🤖 **Testing Gemini API with Custom Certificate**")
    print("=" * 50)
    
    if genai is None:
        print("❌ google-generativeai not installed or not available")
        return
    
    try:
        # Configure Gemini with REST transport
        if hasattr(genai, 'configure'):
            genai.configure(api_key="AIzaSyCafiMftGuBqFTumWP9ZBWJ0UARy5-Svac", transport='rest')
        else:
            print("❌ genai.configure not available")
            return
        
        # Force requests to use custom certificate for all HTTPS calls
        original_get = requests.get
        original_post = requests.post
        
        def custom_get(*args, **kwargs):
            kwargs['verify'] = CUSTOM_CA_BUNDLE
            return original_get(*args, **kwargs)
        
        def custom_post(*args, **kwargs):
            kwargs['verify'] = CUSTOM_CA_BUNDLE
            return original_post(*args, **kwargs)
        
        # Monkey patch requests
        requests.get = custom_get
        requests.post = custom_post
        
        # Test Gemini API
        if hasattr(genai, 'GenerativeModel'):
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Hello! Just testing the API.")
            print(f"✅ Gemini API Success!")
            print(f"Response: {response.text[:100]}...")
        else:
            print("❌ genai.GenerativeModel not available")
        
        # Restore original requests
        requests.get = original_get
        requests.post = original_post
        
    except Exception as e:
        print(f"❌ Gemini API Failed: {str(e)[:100]}...")
        # Restore original requests
        requests.get = original_get if 'original_get' in locals() else requests.get
        requests.post = original_post if 'original_post' in locals() else requests.post

def test_session_with_custom_cert():
    """Test using a requests session with custom certificate"""
    print("📡 **Testing Session with Custom Certificate**")
    print("=" * 50)
    
    try:
        # Create session with custom certificate
        session = requests.Session()
        session.verify = CUSTOM_CA_BUNDLE
        
        # Test with session
        response = session.get('https://generativelanguage.googleapis.com', timeout=10)
        print(f"✅ Session test success: {response.status_code}")
        
    except Exception as e:
        print(f"❌ Session test failed: {str(e)}...")
    print()

if __name__ == "__main__":
    print("🔧 **SSL Certificate Test with Custom Bundle**")
    print("=" * 60)
    print()
    
    debug_certificate_setup()
    test_requests_with_custom_cert()
    test_session_with_custom_cert()
    test_gemini_with_custom_cert()
    
    print("🎯 **Summary**")
    print("=" * 50)
    print("If any test above succeeded, your certificate bundle works!")
    print("The issue is forcing Python requests to use it consistently.")
