import pytest
import asyncio
import time
import sys
import os
from unittest.mock import patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rate_limiter import RateLimiter, APIQuotaManager

class TestRateLimiter:
    
    @pytest.fixture
    def rate_limiter(self):
        """Rate limiter fixture"""
        return RateLimiter()
    
    @pytest.fixture
    def api_quota_manager(self):
        """API quota manager fixture"""
        return APIQuotaManager()
    
    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initialization"""
        assert hasattr(rate_limiter, 'limits')
        assert hasattr(rate_limiter, 'request_history')
        assert 'coingecko' in rate_limiter.limits
        assert 'coinpaprika' in rate_limiter.limits
        assert 'cryptocompare' in rate_limiter.limits
    
    def test_is_allowed_without_requests(self, rate_limiter):
        """Test is_allowed method without any requests"""
        assert rate_limiter.is_allowed('coingecko') == True
        assert rate_limiter.is_allowed('unknown_endpoint') == True  # Unknown endpoints are allowed
    
    def test_record_request_and_is_allowed(self, rate_limiter):
        """Test recording requests and checking allowance"""
        endpoint = 'coingecko'
        
        # Record a request
        rate_limiter.record_request(endpoint)
        
        # Should still be allowed (we haven't hit the limit yet)
        assert rate_limiter.is_allowed(endpoint) == True
        
        # Record more requests to hit the limit
        for _ in range(rate_limiter.limits[endpoint]['requests_per_second'] - 1):
            rate_limiter.record_request(endpoint)
        
        # Should still be allowed (exactly at the limit)
        assert rate_limiter.is_allowed(endpoint) == True
        
        # Record one more to exceed the limit
        rate_limiter.record_request(endpoint)
        
        # Should not be allowed now
        assert rate_limiter.is_allowed(endpoint) == False
    
    def test_clean_old_requests(self, rate_limiter):
        """Test cleaning old requests"""
        endpoint = 'coingecko'
        
        # Record some requests
        for _ in range(5):
            rate_limiter.record_request(endpoint)
        
        # Manually add an old request (61 seconds ago)
        old_time = time.time() - 61
        rate_limiter.request_history[endpoint].appendleft(old_time)
        
        # Clean old requests
        rate_limiter._clean_old_requests(endpoint)
        
        # Should have cleaned the old request
        assert len(rate_limiter.request_history[endpoint]) == 5
    
    def test_get_usage_stats(self, rate_limiter):
        """Test getting usage statistics"""
        endpoint = 'coingecko'
        
        # Record some requests
        for _ in range(3):
            rate_limiter.record_request(endpoint)
        
        # Get stats
        stats = rate_limiter.get_usage_stats(endpoint)
        
        assert 'requests_per_minute' in stats
        assert 'requests_per_second' in stats
        assert 'minute_limit' in stats
        assert 'second_limit' in stats
        assert 'minute_usage_percent' in stats
        assert 'second_usage_percent' in stats
        assert stats['requests_per_minute'] == 3
        assert stats['requests_per_second'] == 3
    
    def test_api_quota_manager_initialization(self, api_quota_manager):
        """Test API quota manager initialization"""
        assert hasattr(api_quota_manager, 'quota_limits')
        assert hasattr(api_quota_manager, 'usage_tracking')
        assert 'coingecko' in api_quota_manager.quota_limits
        assert 'coinpaprika' in api_quota_manager.quota_limits
        assert 'cryptocompare' in api_quota_manager.quota_limits
    
    def test_api_quota_recording(self, api_quota_manager):
        """Test recording API calls"""
        api_name = 'coingecko'
        api_key = 'test_key'
        
        # Record a call
        api_quota_manager.record_api_call(api_name, api_key)
        
        # Check usage tracking
        usage = api_quota_manager.usage_tracking[api_key]
        assert usage['daily_count'] == 1
        assert usage['monthly_count'] == 1
    
    def test_api_quota_availability(self, api_quota_manager):
        """Test checking API quota availability"""
        api_name = 'coingecko'
        api_key = 'test_key'
        
        # Should be available initially
        assert api_quota_manager.is_quota_available(api_name, api_key) == True
        
        # Record many calls to exceed limits
        limits = api_quota_manager.quota_limits[api_name]
        for _ in range(limits['daily_limit'] + 1):
            api_quota_manager.record_api_call(api_name, api_key)
        
        # Should not be available now
        assert api_quota_manager.is_quota_available(api_name, api_key) == False
    
    def test_api_quota_usage_stats(self, api_quota_manager):
        """Test getting API quota usage statistics"""
        api_name = 'coingecko'
        api_key = 'test_key'
        
        # Record some calls
        for _ in range(5):
            api_quota_manager.record_api_call(api_name, api_key)
        
        # Get usage stats
        usage = api_quota_manager.get_quota_usage(api_name, api_key)
        
        assert 'daily_used' in usage
        assert 'daily_limit' in usage
        assert 'daily_percent' in usage
        assert 'monthly_used' in usage
        assert 'monthly_limit' in usage
        assert 'monthly_percent' in usage
        assert usage['daily_used'] == 5
        assert usage['monthly_used'] == 5

if __name__ == "__main__":
    pytest.main([__file__])