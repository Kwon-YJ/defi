#!/usr/bin/env python3
"""
Test script for rate limiter functionality
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set the log level before importing
os.environ['LOG_LEVEL'] = 'INFO'

def test_rate_limiter():
    """Test the rate limiter functionality"""
    try:
        # Import the rate limiter
        from rate_limiter import RateLimiter, APIQuotaManager
        
        # Test RateLimiter
        print("Testing RateLimiter...")
        rate_limiter = RateLimiter()
        
        # Test initialization
        assert hasattr(rate_limiter, 'limits')
        assert hasattr(rate_limiter, 'request_history')
        assert 'coingecko' in rate_limiter.limits
        print("  ✓ RateLimiter initialization")
        
        # Test is_allowed without requests
        assert rate_limiter.is_allowed('coingecko') == True
        assert rate_limiter.is_allowed('unknown_endpoint') == True
        print("  ✓ is_allowed without requests")
        
        # Test recording requests
        endpoint = 'coingecko'
        rate_limiter.record_request(endpoint)
        assert rate_limiter.is_allowed(endpoint) == True
        print("  ✓ Recording requests")
        
        # Test usage stats
        stats = rate_limiter.get_usage_stats(endpoint)
        assert 'requests_per_minute' in stats
        assert 'requests_per_second' in stats
        print("  ✓ Usage stats")
        
        # Test APIQuotaManager
        print("Testing APIQuotaManager...")
        quota_manager = APIQuotaManager()
        
        # Test initialization
        assert hasattr(quota_manager, 'quota_limits')
        assert hasattr(quota_manager, 'usage_tracking')
        assert 'coingecko' in quota_manager.quota_limits
        print("  ✓ APIQuotaManager initialization")
        
        # Test recording API calls
        api_name = 'coingecko'
        api_key = 'test_key'
        quota_manager.record_api_call(api_name, api_key)
        
        usage = quota_manager.usage_tracking[api_key]
        assert usage['daily_count'] == 1
        assert usage['monthly_count'] == 1
        print("  ✓ Recording API calls")
        
        # Test quota availability
        assert quota_manager.is_quota_available(api_name, api_key) == True
        print("  ✓ Quota availability")
        
        # Test quota usage stats
        usage_stats = quota_manager.get_quota_usage(api_name, api_key)
        assert 'daily_used' in usage_stats
        assert 'monthly_used' in usage_stats
        print("  ✓ Quota usage stats")
        
        print("\n✓ All rate limiter tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Rate limiter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rate_limiter()
    sys.exit(0 if success else 1)