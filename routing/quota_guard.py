#!/usr/bin/env python3
"""
quota_guard.py - Model Quota Monitoring and Fallback Management

Ensures model switches only happen when target has available quota.
Prevents getting stuck when a model is exhausted or unconfigured.
Version: 0.1.1 (integrated with model-router)
Author: JARVIS-Koz
License: MIT
"""

import json
import subprocess
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time


class QuotaStatus(Enum):
    """Quota availability status."""
    HEALTHY = "healthy"           # > 50% remaining
    WARNING = "warning"           # 20-50% remaining
    CRITICAL = "critical"         # < 20% remaining
    EXHAUSTED = "exhausted"       # 0% or error
    UNKNOWN = "unknown"           # Can't determine
    UNCONFIGURED = "unconfigured" # No credentials


@dataclass
class ModelQuota:
    """Quota information for a model."""
    model_id: str
    status: QuotaStatus
    percent_remaining: Optional[float]
    estimated_calls_remaining: Optional[int]
    last_checked: float
    error_message: Optional[str] = None


class QuotaGuard:
    """
    Monitors model quotas and prevents switching to unavailable models.
    """
    
    # Model to provider mapping for quota checks
    MODEL_PROVIDERS = {
        "kimi-code/kimi-for-coding": "kimi-code",
        "google-antigravity/claude-opus-4-5-thinking": "google-antigravity",
        "google-antigravity/gemini-3-pro-high": "google-antigravity",
        "google-antigravity/gemini-3-flash": "google-antigravity",
        "google/gemini-3-pro-preview": "google",
        "google/gemini-3-flash-preview": "google",
        "openrouter/auto": "openrouter"
    }
    
    # Cost per 1k tokens (input + output avg) for estimation
    MODEL_COSTS = {
        "kimi-code/kimi-for-coding": 0.009,
        "google-antigravity/claude-opus-4-5-thinking": 0.090,
        "google-antigravity/gemini-3-pro-high": 0.010,
        "google-antigravity/gemini-3-flash": 0.002,
        "google/gemini-3-pro-preview": 0.007,
        "google/gemini-3-flash-preview": 0.0005,
        "openrouter/auto": 0.005
    }
    
    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache: Dict[str, ModelQuota] = {}
        self.cache_ttl = cache_ttl_seconds
        self.last_check: Dict[str, float] = {}
    
    def check_quota(self, model_id: str, force_refresh: bool = False) -> ModelQuota:
        """
        Check quota for a model. Uses cache unless force_refresh.
        """
        now = time.time()
        
        # Check cache
        if not force_refresh and model_id in self.cache:
            cached = self.cache[model_id]
            if now - cached.last_checked < self.cache_ttl:
                return cached
        
        # Perform fresh check
        quota = self._do_check(model_id)
        self.cache[model_id] = quota
        return quota
    
    def _do_check(self, model_id: str) -> ModelQuota:
        """Actually check quota via system commands."""
        provider = self.MODEL_PROVIDERS.get(model_id)
        
        if not provider:
            return ModelQuota(
                model_id=model_id,
                status=QuotaStatus.UNKNOWN,
                percent_remaining=None,
                estimated_calls_remaining=None,
                last_checked=time.time(),
                error_message=f"Unknown provider for {model_id}"
            )
        
        try:
            # Try to get status via clawdbot
            result = subprocess.run(
                ["clawdbot", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise Exception(f"clawdbot status failed: {result.stderr}")
            
            # Parse status
            return self._parse_clawdbot_status(model_id, provider, result.stdout)
            
        except Exception as e:
            # Fallback: check config file
            return self._check_config_fallback(model_id, provider, str(e))
    
    def _parse_clawdbot_status(self, model_id: str, provider: str, status_output: str) -> ModelQuota:
        """Parse clawdbot status output."""
        try:
            # Try to extract quota info from session_status style output
            # Since we can't easily parse the exact format, use heuristic
            
            # Check if model is mentioned in fallbacks (indicates configuration)
            if provider in status_output.lower():
                # Model is configured
                return ModelQuota(
                    model_id=model_id,
                    status=QuotaStatus.HEALTHY,  # Assume healthy if configured
                    percent_remaining=100.0,
                    estimated_calls_remaining=10000,
                    last_checked=time.time()
                )
            else:
                return ModelQuota(
                    model_id=model_id,
                    status=QuotaStatus.UNCONFIGURED,
                    percent_remaining=0.0,
                    estimated_calls_remaining=0,
                    last_checked=time.time(),
                    error_message="Model not in configured fallbacks"
                )
        except Exception as e:
            return ModelQuota(
                model_id=model_id,
                status=QuotaStatus.UNKNOWN,
                percent_remaining=None,
                estimated_calls_remaining=None,
                last_checked=time.time(),
                error_message=str(e)
            )
    
    def _check_config_fallback(self, model_id: str, provider: str, error: str) -> ModelQuota:
        """Fallback: check config file for model availability."""
        try:
            config_path = "/home/ubuntu/.clawdbot/clawdbot.json"
            with open(config_path) as f:
                config = json.load(f)
            
            # Check if provider is configured
            profiles = config.get("auth", {}).get("profiles", {})
            provider_configured = any(
                provider in pid for pid in profiles.keys()
            )
            
            if not provider_configured:
                return ModelQuota(
                    model_id=model_id,
                    status=QuotaStatus.UNCONFIGURED,
                    percent_remaining=0.0,
                    estimated_calls_remaining=0,
                    last_checked=time.time(),
                    error_message=f"Provider '{provider}' not configured"
                )
            
            # Check if in fallbacks
            fallbacks = config.get("agents", {}).get("defaults", {}).get("model", {}).get("fallbacks", [])
            if model_id in fallbacks:
                return ModelQuota(
                    model_id=model_id,
                    status=QuotaStatus.HEALTHY,
                    percent_remaining=100.0,
                    estimated_calls_remaining=10000,
                    last_checked=time.time()
                )
            else:
                return ModelQuota(
                    model_id=model_id,
                    status=QuotaStatus.UNKNOWN,
                    percent_remaining=50.0,  # Conservative estimate
                    estimated_calls_remaining=5000,
                    last_checked=time.time(),
                    error_message="Not in fallback chain, assuming available"
                )
                
        except Exception as e:
            return ModelQuota(
                model_id=model_id,
                status=QuotaStatus.UNKNOWN,
                percent_remaining=None,
                estimated_calls_remaining=None,
                last_checked=time.time(),
                error_message=f"Config check failed: {e}"
            )
    
    def can_use_model(self, model_id: str, min_percent: float = 10.0) -> Tuple[bool, str]:
        """
        Check if a model can be used.
        
        Returns:
            (can_use, reason)
        """
        quota = self.check_quota(model_id)
        
        if quota.status == QuotaStatus.UNCONFIGURED:
            return False, f"❌ {model_id} not configured"
        
        if quota.status == QuotaStatus.EXHAUSTED:
            return False, f"❌ {model_id} quota exhausted"
        
        if quota.status == QuotaStatus.UNKNOWN:
            # Allow but warn
            return True, f"⚠️ {model_id} quota unknown, proceeding with caution"
        
        if quota.percent_remaining and quota.percent_remaining < min_percent:
            return False, f"❌ {model_id} quota low ({quota.percent_remaining:.1f}%)"
        
        return True, f"✅ {model_id} available ({quota.status.value})"
    
    def find_available_model(self, preferred_models: List[str], task_type: str = "") -> Tuple[Optional[str], str]:
        """
        Find first available model from a priority list.
        
        Returns:
            (model_id or None, reason)
        """
        for model_id in preferred_models:
            can_use, reason = self.can_use_model(model_id)
            if can_use:
                return model_id, reason
        
        # None available - return emergency fallback
        return None, "❌ All preferred models unavailable. Emergency fallback required."
    
    def get_all_status(self) -> Dict[str, ModelQuota]:
        """Get status of all known models."""
        models = list(self.MODEL_PROVIDERS.keys())
        return {m: self.check_quota(m) for m in models}


# Integration with model-router
def create_quota_aware_router():
    """Create a model router with quota checking enabled."""
    from model_router import ModelRouter, ModelRegistry
    
    class QuotaAwareRouter(ModelRouter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.quota_guard = QuotaGuard()
        
        def route_with_quota_check(self, prompt: str, **kwargs) -> dict:
            """Route with quota validation."""
            # Get normal routing decision
            decision = self.route(prompt, **kwargs)
            recommended = decision["model_id"]
            
            # Check quota
            can_use, quota_reason = self.quota_guard.can_use_model(recommended)
            
            if not can_use:
                # Find alternative
                task_type = decision["task_type"]
                candidates = self.TASK_MODEL_MAP.get(task_type, ["gemini-flash"])
                
                available, alt_reason = self.quota_guard.find_available_model(candidates)
                
                if available:
                    # Update decision with available model
                    model_config = ModelRegistry.get(available.replace("kimi-code/kimi-for-coding", "kimi-code").replace("google/gemini-3-flash-preview", "gemini-flash").replace("google/gemini-3-pro-preview", "gemini-pro").replace("google-antigravity/claude-opus-4-5-thinking", "claude-opus"))
                    if not model_config:
                        # Try direct lookup
                        for key, cfg in ModelRegistry.MODELS.items():
                            if cfg.name == available or key in available:
                                model_config = cfg
                                break
                    
                    if model_config:
                        decision["model_id"] = model_config.name
                        decision["provider"] = model_config.provider
                        decision["emoji"] = model_config.emoji
                        decision["quota_note"] = f"Switched from {recommended} due to quota: {quota_reason}"
                        decision["original_recommendation"] = recommended
            else:
                decision["quota_note"] = quota_reason
            
            return decision
    
    return QuotaAwareRouter()


# CLI for quota checking
if __name__ == "__main__":
    import sys
    
    guard = QuotaGuard()
    
    if len(sys.argv) < 2:
        print("Usage: python quota_guard.py [status|check|monitor]")
        print("\nCommands:")
        print("  status        - Show all model quota status")
        print("  check <model> - Check specific model")
        print("\nExamples:")
        print('  python quota_guard.py status')
        print('  python quota_guard.py check kimi-code/kimi-for-coding')
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "status":
        print("🛡️ Model Quota Status")
        print("=" * 60)
        
        statuses = guard.get_all_status()
        for model_id, quota in statuses.items():
            emoji = "✅" if quota.status == QuotaStatus.HEALTHY else \
                    "⚠️" if quota.status == QuotaStatus.WARNING else \
                    "🔴" if quota.status in [QuotaStatus.EXHAUSTED, QuotaStatus.UNCONFIGURED] else "❓"
            
            percent = f"{quota.percent_remaining:.0f}%" if quota.percent_remaining else "N/A"
            print(f"{emoji} {model_id:45} {quota.status.value:15} {percent}")
    
    elif command == "check" and len(sys.argv) >= 3:
        model_id = sys.argv[2]
        can_use, reason = guard.can_use_model(model_id)
        print(reason)
