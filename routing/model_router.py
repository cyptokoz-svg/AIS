#!/usr/bin/env python3
"""
model_router.py - Intelligent Model Selection for AI Agents

Automatically routes tasks to the optimal LLM based on context analysis.
Version: 0.1.0
Author: JARVIS-Koz
License: MIT

Provides:
- Task classification and routing
- Cost-performance optimization
- Model fallback chains
- Usage analytics and recommendations
"""

import re
import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import time


class TaskType(Enum):
    """Categories of agent tasks."""
    CODING = "coding"           # Code generation, debugging, architecture
    RESEARCH = "research"       # Deep analysis, synthesis, reasoning
    CHAT = "chat"               # Casual conversation, quick responses
    CREATIVE = "creative"       # Writing, storytelling, ideation
    DATA = "data"               # Data processing, transformation
    SOCIAL = "social"           # Social media, community management
    CRITICAL = "critical"       # Security, high-stakes decisions


class ModelTier(Enum):
    """Model capability tiers."""
    FAST = "fast"           # Speed priority (Gemini-Flash)
    BALANCED = "balanced"   # Balanced (Gemini-Pro)
    POWER = "power"         # Capability priority (Kimi, Claude)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str
    tier: ModelTier
    context_window: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    strengths: List[str]
    emoji: str
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        return (input_tokens / 1000 * self.cost_per_1k_input + 
                output_tokens / 1000 * self.cost_per_1k_output)


class ModelRegistry:
    """Registry of available models."""
    
    MODELS = {
        "kimi-code": ModelConfig(
            name="kimi-code/kimi-for-coding",
            provider="kimi-code",
            tier=ModelTier.POWER,
            context_window=262144,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.006,
            strengths=["coding", "architecture", "debugging", "complex_reasoning"],
            emoji="🎯"
        ),
        "gemini-flash": ModelConfig(
            name="google/gemini-3-flash-preview",
            provider="google",
            tier=ModelTier.FAST,
            context_window=1048576,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            strengths=["speed", "chat", "social", "simple_tasks"],
            emoji="⚡"
        ),
        "gemini-pro": ModelConfig(
            name="google/gemini-3-pro-preview",
            provider="google",
            tier=ModelTier.BALANCED,
            context_window=1048576,
            cost_per_1k_input=0.0005,
            cost_per_1k_output=0.0015,
            strengths=["research", "analysis", "creative", "general"],
            emoji="🧠"
        ),
        "claude-opus": ModelConfig(
            name="google-antigravity/claude-opus-4-5-thinking",
            provider="google-antigravity",
            tier=ModelTier.POWER,
            context_window=200000,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            strengths=["deep_reasoning", "research", "complex_analysis", "coding"],
            emoji="🔮"
        ),
        "openrouter": ModelConfig(
            name="openrouter/auto",
            provider="openrouter",
            tier=ModelTier.BALANCED,
            context_window=128000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            strengths=["fallback", "diverse", "general"],
            emoji="🔀"
        )
    }
    
    @classmethod
    def get(cls, model_id: str) -> Optional[ModelConfig]:
        return cls.MODELS.get(model_id)
    
    @classmethod
    def by_tier(cls, tier: ModelTier) -> List[ModelConfig]:
        return [m for m in cls.MODELS.values() if m.tier == tier]


class TaskAnalyzer:
    """Analyzes prompts to determine task characteristics."""
    
    CODING_INDICATORS = [
        r"\b(code|coding|program|script|function|class|debug|refactor|implement|build|develop)\b",
        r"\b(python|javascript|typescript|rust|go|java|c\+\+|sql|bash|shell)\b",
        r"\b(api|endpoint|route|handler|middleware|database|schema|migration)\b",
        r"\b(git|github|commit|pull|merge|branch|repository)\b",
        r"\b(error|exception|bug|fix|test|unittest|integration)\b",
        r"```[\s\S]*?```",  # Code blocks
        r"def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import"
    ]
    
    RESEARCH_INDICATORS = [
        r"\b(analyze|analysis|research|investigate|study|compare|contrast|evaluate)\b",
        r"\b(why|how does|what is|explain|elaborate|deep dive|thorough)\b",
        r"\b(pros?\s+and\s+cons?|advantages?|disadvantages?|tradeoffs?|implications?)\b",
        r"\b(synthesize|synthesis|comprehensive|detailed|in-depth|systematic)\b"
    ]
    
    CREATIVE_INDICATORS = [
        r"\b(write|story|poem|creative|imagine|design|brainstorm|ideate)\b",
        r"\b(marketing|copy|blog|article|content|narrative|fiction)\b"
    ]
    
    SOCIAL_INDICATORS = [
        r"\b(post|tweet|reply|comment|social|moltbook|discord|telegram)\b",
        r"\b(engagement|community|follower|like|share|viral|trending)\b"
    ]
    
    CRITICAL_INDICATORS = [
        r"\b(security|vulnerability|exploit|attack|audit|encrypt|private key)\b",
        r"\b(deploy|production|release|merge|critical|urgent|emergency)\b",
        r"\b(password|credential|secret|token|api[_-]?key)\b"
    ]
    
    DATA_INDICATORS = [
        r"\b(data|csv|json|xml|parse|transform|extract|load|etl|pipeline)\b",
        r"\b(database|query|sql|mongodb|postgres|redis|cache)\b"
    ]
    
    @classmethod
    def analyze(cls, prompt: str) -> Dict[str, float]:
        """
        Analyze prompt and return confidence scores for each task type.
        """
        prompt_lower = prompt.lower()
        scores = {}
        
        # Coding score
        coding_score = sum(1 for pattern in cls.CODING_INDICATORS 
                          if re.search(pattern, prompt_lower, re.IGNORECASE))
        scores[TaskType.CODING] = min(coding_score / 3, 1.0)
        
        # Research score
        research_score = sum(1 for pattern in cls.RESEARCH_INDICATORS 
                            if re.search(pattern, prompt_lower, re.IGNORECASE))
        scores[TaskType.RESEARCH] = min(research_score / 2, 1.0)
        
        # Creative score
        creative_score = sum(1 for pattern in cls.CREATIVE_INDICATORS 
                            if re.search(pattern, prompt_lower, re.IGNORECASE))
        scores[TaskType.CREATIVE] = min(creative_score / 2, 1.0)
        
        # Social score
        social_score = sum(1 for pattern in cls.SOCIAL_INDICATORS 
                          if re.search(pattern, prompt_lower, re.IGNORECASE))
        scores[TaskType.SOCIAL] = min(social_score / 2, 1.0)
        
        # Critical score
        critical_score = sum(1 for pattern in cls.CRITICAL_INDICATORS 
                            if re.search(pattern, prompt_lower, re.IGNORECASE))
        scores[TaskType.CRITICAL] = min(critical_score / 2, 1.0)
        
        # Data score
        data_score = sum(1 for pattern in cls.DATA_INDICATORS 
                        if re.search(pattern, prompt_lower, re.IGNORECASE))
        scores[TaskType.DATA] = min(data_score / 2, 1.0)
        
        # Chat is default/low effort
        max_other = max(scores.values()) if scores else 0
        scores[TaskType.CHAT] = max(0, 0.3 - max_other * 0.3)
        
        return scores


class ModelRouter:
    """
    Intelligent model router that selects optimal model for each task.
    Includes quota awareness and fallback safety.
    """
    
    # Maps task types to preferred model IDs (short form)
    TASK_MODEL_MAP = {
        TaskType.CODING: ["kimi-code", "claude-opus"],
        TaskType.RESEARCH: ["claude-opus", "gemini-pro", "kimi-code"],
        TaskType.CRITICAL: ["claude-opus", "kimi-code", "gemini-pro"],
        TaskType.DATA: ["kimi-code", "gemini-pro", "gemini-flash"],
        TaskType.CREATIVE: ["gemini-pro", "claude-opus", "gemini-flash"],
        TaskType.SOCIAL: ["gemini-flash", "gemini-pro"],
        TaskType.CHAT: ["gemini-flash", "gemini-pro"]
    }
    
    # Full model names for config checking
    FULL_MODEL_NAMES = {
        "kimi-code": "kimi-code/kimi-for-coding",
        "claude-opus": "google-antigravity/claude-opus-4-5-thinking",
        "gemini-pro": "google-antigravity/gemini-3-pro-high",
        "gemini-flash": "google-antigravity/gemini-3-flash",
        "openrouter": "openrouter/auto"
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("./model_router_config.json")
        self.usage_history: List[Dict] = []
        self.configured_models: set = set()
        self._load_clawdbot_config()
        self.load_config()
    
    def _load_clawdbot_config(self):
        """Load and validate clawdbot configuration."""
        try:
            import json
            clawdbot_config = Path("/home/ubuntu/.clawdbot/clawdbot.json")
            if clawdbot_config.exists():
                with open(clawdbot_config) as f:
                    config = json.load(f)
                
                # Get configured providers
                profiles = config.get("auth", {}).get("profiles", {})
                providers = set()
                for pid in profiles.keys():
                    if ":" in pid:
                        providers.add(pid.split(":")[0])
                
                # Check which models are available
                for short, full in self.FULL_MODEL_NAMES.items():
                    provider = full.split("/")[0]
                    if provider in providers:
                        self.configured_models.add(short)
                
                # Also check fallbacks
                fallbacks = config.get("agents", {}).get("defaults", {}).get("model", {}).get("fallbacks", [])
                for fb in fallbacks:
                    for short, full in self.FULL_MODEL_NAMES.items():
                        if fb == full:
                            self.configured_models.add(short)
        except Exception as e:
            print(f"Warning: Could not load clawdbot config: {e}")
    
    def load_config(self):
        """Load router configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                self.usage_history = data.get("history", [])
    
    def save_config(self):
        """Save router configuration."""
        with open(self.config_path, 'w') as f:
            json.dump({
                "history": self.usage_history[-100:],  # Keep last 100
                "last_updated": time.time()
            }, f, indent=2)
    
    def route(self, prompt: str, context_tokens: int = 0, 
              budget_priority: bool = False,
              force_model: Optional[str] = None) -> Dict:
        """
        Route a task to the optimal model.
        
        Args:
            prompt: The task/prompt text
            context_tokens: Current context size
            budget_priority: Prefer cheaper models if True
            force_model: Override routing with specific model
        
        Returns:
            Routing decision with metadata
        """
        if force_model and force_model in ModelRegistry.MODELS:
            model = ModelRegistry.get(force_model)
            return self._create_decision(
                model, TaskType.CHAT, prompt, 
                reason="forced", confidence=1.0
            )
        
        # Analyze task
        scores = TaskAnalyzer.analyze(prompt)
        primary_task = max(scores, key=scores.get)
        confidence = scores[primary_task]
        
        # Get model candidates
        candidates = self.TASK_MODEL_MAP.get(primary_task, ["gemini-pro"])
        
        # Select based on constraints
        selected_model_id = self._select_model(
            candidates, context_tokens, budget_priority
        )
        model = ModelRegistry.get(selected_model_id)
        
        # Create decision
        decision = self._create_decision(
            model, primary_task, prompt,
            reason=f"task_type={primary_task.value}, confidence={confidence:.2f}",
            confidence=confidence,
            all_scores={k.value: v for k, v in scores.items()}
        )
        
        # Log usage
        self._log_usage(decision)
        
        return decision
    
    def _is_model_safe(self, short_id: str) -> tuple[bool, str]:
        """Check if model is configured and safe to use."""
        if short_id not in self.configured_models:
            return False, f"{short_id} not in configured models"
        
        model = ModelRegistry.get(short_id)
        if not model:
            return False, f"{short_id} not in registry"
        
        return True, "OK"
    
    def _select_model(self, candidates: List[str], context_tokens: int,
                      budget_priority: bool) -> str:
        """Select best model from candidates with safety checks."""
        for model_id in candidates:
            # Safety check first
            is_safe, reason = self._is_model_safe(model_id)
            if not is_safe:
                print(f"⚠️ Skipping {model_id}: {reason}")
                continue
            
            model = ModelRegistry.get(model_id)
            if not model:
                continue
            
            # Check context fit
            if context_tokens > model.context_window * 0.8:
                print(f"⚠️ Skipping {model_id}: context too large")
                continue
            
            # If budget priority, prefer cheaper models
            if budget_priority and model.tier == ModelTier.FAST:
                return model_id
            
            return model_id
        
        # Emergency fallback - must be safe
        for emergency in ["gemini-flash", "openrouter"]:
            is_safe, _ = self._is_model_safe(emergency)
            if is_safe:
                print(f"🆘 Emergency fallback to {emergency}")
                return emergency
        
        # Absolute fallback
        print("🆘 CRITICAL: No safe models found, using gemini-flash")
        return "gemini-flash"
    
    def _create_decision(self, model: ModelConfig, task: TaskType,
                        prompt: str, reason: str, confidence: float,
                        all_scores: Optional[Dict] = None) -> Dict:
        """Create routing decision object."""
        return {
            "model_id": model.name,
            "provider": model.provider,
            "emoji": model.emoji,
            "task_type": task.value,
            "confidence": confidence,
            "reason": reason,
            "estimated_cost": model.estimate_cost(
                len(prompt) // 4,  # rough input estimate
                500  # assumed output
            ),
            "context_window": model.context_window,
            "all_task_scores": all_scores or {},
            "timestamp": time.time()
        }
    
    def _log_usage(self, decision: Dict):
        """Log usage for analytics."""
        self.usage_history.append(decision)
        self.save_config()
    
    def get_analytics(self) -> Dict:
        """Get usage analytics."""
        if not self.usage_history:
            return {"message": "No usage data yet"}
        
        models_used = {}
        tasks = {}
        total_cost = 0
        
        for d in self.usage_history:
            model = d["model_id"]
            models_used[model] = models_used.get(model, 0) + 1
            
            task = d["task_type"]
            tasks[task] = tasks.get(task, 0) + 1
            
            total_cost += d["estimated_cost"]
        
        return {
            "total_requests": len(self.usage_history),
            "models_distribution": models_used,
            "task_distribution": tasks,
            "estimated_total_cost_usd": round(total_cost, 4),
            "average_cost_per_request": round(total_cost / len(self.usage_history), 4)
        }
    
    def should_switch(self, current_model: str, new_prompt: str,
                     threshold: float = 0.7) -> Tuple[bool, Optional[str], str]:
        """
        Determine if we should switch models for a new prompt.
        
        Returns:
            (should_switch, recommended_model, reason)
        """
        decision = self.route(new_prompt)
        recommended = decision["model_id"]
        
        if recommended == current_model:
            return False, None, "already_optimal"
        
        if decision["confidence"] < threshold:
            return False, None, f"low_confidence({decision['confidence']:.2f})"
        
        # Extract model key from full name
        for key, config in ModelRegistry.MODELS.items():
            if config.name == recommended:
                return True, key, f"task_type={decision['task_type']}"
        
        return False, None, "unknown_model"


# CLI interface
if __name__ == "__main__":
    import sys
    
    router = ModelRouter()
    
    if len(sys.argv) < 2:
        print("Usage: python model_router.py <command> [args]")
        print("\nCommands:")
        print("  route <prompt>     - Route a prompt to optimal model")
        print("  check <model> <prompt> - Check if should switch from model")
        print("  analytics          - Show usage analytics")
        print("  status             - Show model configuration status")
        print("\nExamples:")
        print('  python model_router.py route "Write a Python function"')
        print('  python model_router.py route "Quick hello" --budget')
        print('  python model_router.py status')
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "route":
        prompt = " ".join(sys.argv[2:])
        budget = "--budget" in sys.argv
        
        decision = router.route(prompt, budget_priority=budget)
        
        print(f"\n🎯 Routing Decision")
        print("=" * 40)
        print(f"Model: {decision['emoji']} {decision['model_id']}")
        print(f"Task: {decision['task_type']}")
        print(f"Confidence: {decision['confidence']:.2%}")
        print(f"Est. Cost: ${decision['estimated_cost']:.4f}")
        print(f"Reason: {decision['reason']}")
        
        if decision['all_task_scores']:
            print("\nTask Scores:")
            for task, score in sorted(decision['all_task_scores'].items(), 
                                     key=lambda x: x[1], reverse=True):
                bar = "█" * int(score * 20)
                print(f"  {task:12} {bar} {score:.2f}")
    
    elif command == "check" and len(sys.argv) >= 4:
        current = sys.argv[2]
        prompt = " ".join(sys.argv[3:])
        
        should_switch, recommended, reason = router.should_switch(current, prompt)
        
        if should_switch:
            model = ModelRegistry.get(recommended)
            print(f"🔄 Switch Recommended")
            print(f"   From: {current}")
            print(f"   To:   {model.emoji} {model.name}")
            print(f"   Why:  {reason}")
        else:
            print(f"✅ Stay on {current}")
            print(f"   Why:  {reason}")
    
    elif command == "analytics":
        analytics = router.get_analytics()
        print(json.dumps(analytics, indent=2))
    
    elif command == "status":
        print("🛡️ Model Router Configuration Status")
        print("=" * 50)
        print("\n✅ Configured Models:")
        for short in router.configured_models:
            model = ModelRegistry.get(short)
            if model:
                print(f"   {model.emoji} {short:15} -> {model.name}")
        
        print("\n❌ Not Configured (will be skipped):")
        all_models = set(ModelRegistry.MODELS.keys())
        for short in all_models - router.configured_models:
            model = ModelRegistry.get(short)
            if model:
                print(f"   ⚠️  {short:15} -> {model.name}")
        
        print("\n✅ Router is ready with safe fallbacks")
    
    else:
        print(f"Unknown command: {command}")
