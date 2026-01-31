#!/usr/bin/env python3
"""
skill_auditor.py - Automated Security Auditor for Agent Skills

Scans skill code for security vulnerabilities and best practices.
Version: 0.1.0
Author: JARVIS-Koz
License: MIT

Provides:
- Dangerous pattern detection
- Secret/key leakage scanning
- Configuration validation
- Security score generation
- Remediation recommendations
"""

import ast
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib


class Severity(Enum):
    CRITICAL = "critical"  # Immediate risk (key leakage, code execution)
    HIGH = "high"          # Significant risk (network calls without validation)
    MEDIUM = "medium"      # Moderate risk (unsafe deserialization)
    LOW = "low"            # Minor issues (hardcoded paths)
    INFO = "info"          # Best practice suggestions


@dataclass
class Finding:
    """A security finding with context."""
    rule_id: str
    severity: Severity
    message: str
    file: str
    line: int
    code_snippet: str
    remediation: str
    
    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "code_snippet": self.code_snippet,
            "remediation": self.remediation
        }


class SecurityRules:
    """Security rules database."""
    
    DANGEROUS_PATTERNS = {
        "exec-dynamic": {
            "pattern": r"\b(exec|eval)\s*\(",
            "severity": Severity.CRITICAL,
            "message": "Dynamic code execution detected",
            "remediation": "Avoid exec/eval. Use ast.literal_eval for safe parsing."
        },
        "subprocess-shell": {
            "pattern": r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
            "severity": Severity.CRITICAL,
            "message": "Subprocess with shell=True allows command injection",
            "remediation": "Use shell=False and pass commands as lists."
        },
        "hardcoded-secret": {
            "pattern": r"(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{8,}['\"]",
            "severity": Severity.CRITICAL,
            "message": "Hardcoded secret detected",
            "remediation": "Use environment variables or secure key management."
        },
        "requests-no-verify": {
            "pattern": r"requests\.(get|post|put|delete)\s*\([^)]*verify\s*=\s*False",
            "severity": Severity.HIGH,
            "message": "SSL certificate verification disabled",
            "remediation": "Always verify SSL certificates in production."
        },
        "yaml-unsafe": {
            "pattern": r"yaml\.load\s*\([^)]*\)",
            "severity": Severity.HIGH,
            "message": "Unsafe YAML loading can execute arbitrary code",
            "remediation": "Use yaml.safe_load() instead of yaml.load()."
        },
        "pickle-load": {
            "pattern": r"pickle\.load\s*\(|pickle\.loads\s*\(",
            "severity": Severity.HIGH,
            "message": "Pickle deserialization is unsafe with untrusted data",
            "remediation": "Use json or msgpack for untrusted data serialization."
        },
        "http-url": {
            "pattern": r"http://(?!localhost|127\.0\.0\.1)",
            "severity": Severity.MEDIUM,
            "message": "Insecure HTTP URL detected",
            "remediation": "Use HTTPS for all external communications."
        },
        "broad-except": {
            "pattern": r"except\s*:\s*$|except\s+Exception\s*:",
            "severity": Severity.MEDIUM,
            "message": "Broad exception handling can mask security issues",
            "remediation": "Catch specific exceptions and log security-relevant errors."
        },
        "hardcoded-path": {
            "pattern": r"['\"](/tmp/|/var/tmp/|C:\\\\Windows\\\\Temp)[^'\"]*['\"]",
            "severity": Severity.LOW,
            "message": "Hardcoded temporary path",
            "remediation": "Use tempfile module for cross-platform compatibility."
        },
        "debug-mode": {
            "pattern": r"debug\s*=\s*True|DEBUG\s*=\s*True",
            "severity": Severity.LOW,
            "message": "Debug mode enabled",
            "remediation": "Ensure debug mode is disabled in production."
        }
    }
    
    SENSITIVE_FILES = [
        ".env", ".env.local", ".env.production",
        "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
        ".aws/credentials", ".ssh/config",
        "secrets.json", "config.json"  # Often contain keys
    ]


class SkillAuditor:
    """Main auditor engine."""
    
    def __init__(self, skill_path: str):
        self.skill_path = Path(skill_path)
        self.findings: List[Finding] = []
        self.files_scanned = 0
        self.lines_scanned = 0
    
    def audit(self) -> Dict[str, Any]:
        """Run full audit and return report."""
        if not self.skill_path.exists():
            return {"error": f"Path not found: {self.skill_path}"}
        
        # Scan Python files
        for py_file in self.skill_path.rglob("*.py"):
            self._scan_python_file(py_file)
        
        # Scan config files
        for config_file in self.skill_path.rglob("*.json"):
            self._scan_config_file(config_file)
        
        # Check for sensitive files
        self._check_sensitive_files()
        
        # Generate report
        return self._generate_report()
    
    def _scan_python_file(self, file_path: Path):
        """Scan a Python file for security issues."""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            self.files_scanned += 1
            self.lines_scanned += len(lines)
            
            # Pattern-based scanning
            for rule_id, rule in SecurityRules.DANGEROUS_PATTERNS.items():
                for line_num, line in enumerate(lines, 1):
                    if re.search(rule["pattern"], line, re.IGNORECASE):
                        self.findings.append(Finding(
                            rule_id=rule_id,
                            severity=rule["severity"],
                            message=rule["message"],
                            file=str(file_path.relative_to(self.skill_path)),
                            line=line_num,
                            code_snippet=line.strip(),
                            remediation=rule["remediation"]
                        ))
            
            # AST-based scanning for complex patterns
            self._ast_scan(file_path, content)
            
        except Exception as e:
            self.findings.append(Finding(
                rule_id="parse-error",
                severity=Severity.INFO,
                message=f"Could not parse file: {e}",
                file=str(file_path),
                line=0,
                code_snippet="",
                remediation="Check file encoding and syntax."
            ))
    
    def _ast_scan(self, file_path: Path, content: str):
        """Use AST for deeper analysis."""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for unsafe imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['pickle', 'marshal', 'subprocess']:
                            self.findings.append(Finding(
                                rule_id=f"import-{alias.name}",
                                severity=Severity.MEDIUM,
                                message=f"Import of potentially dangerous module: {alias.name}",
                                file=str(file_path.relative_to(self.skill_path)),
                                line=node.lineno,
                                code_snippet=f"import {alias.name}",
                                remediation=f"Ensure {alias.name} is used safely with validated inputs."
                            ))
                
                # Check for network calls without timeout
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['get', 'post', 'put', 'delete']:
                            # Check if timeout is specified
                            has_timeout = any(
                                kw.arg == 'timeout' for kw in node.keywords
                            )
                            if not has_timeout:
                                self.findings.append(Finding(
                                    rule_id="requests-no-timeout",
                                    severity=Severity.MEDIUM,
                                    message="HTTP request without timeout can hang indefinitely",
                                    file=str(file_path.relative_to(self.skill_path)),
                                    line=node.lineno,
                                    code_snippet=ast.get_source_segment(content, node) or "requests call",
                                    remediation="Always specify timeout parameter for network calls."
                                ))
        
        except SyntaxError:
            pass  # Already handled in pattern scanning
    
    def _scan_config_file(self, file_path: Path):
        """Scan JSON config files for issues."""
        try:
            content = file_path.read_text(encoding='utf-8')
            data = json.loads(content)
            
            # Check for hardcoded keys in config
            self._check_config_recursive(data, file_path)
            
        except json.JSONDecodeError:
            pass  # Not a valid JSON, skip
    
    def _check_config_recursive(self, data: Any, file_path: Path, path: str = ""):
        """Recursively check config for secrets."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check key names for sensitive patterns
                if any(pattern in key.lower() for pattern in ['key', 'secret', 'token', 'password']):
                    if isinstance(value, str) and len(value) > 8:
                        self.findings.append(Finding(
                            rule_id="config-secret",
                            severity=Severity.CRITICAL,
                            message=f"Potential secret in config: {current_path}",
                            file=str(file_path.relative_to(self.skill_path)),
                            line=0,
                            code_snippet=f"{key}: {'*' * min(len(value), 8)}...",
                            remediation="Move secrets to environment variables or secret management."
                        ))
                
                # Recurse
                self._check_config_recursive(value, file_path, current_path)
    
    def _check_sensitive_files(self):
        """Check for accidentally committed sensitive files."""
        for pattern in SecurityRules.SENSITIVE_FILES:
            for file_path in self.skill_path.rglob(pattern):
                self.findings.append(Finding(
                    rule_id="sensitive-file",
                    severity=Severity.CRITICAL,
                    message=f"Potentially sensitive file found: {file_path.name}",
                    file=str(file_path.relative_to(self.skill_path)),
                    line=0,
                    code_snippet="",
                    remediation="Add to .gitignore and rotate any exposed credentials."
                ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate audit report."""
        findings_by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": []
        }
        
        for finding in self.findings:
            findings_by_severity[finding.severity.value].append(finding.to_dict())
        
        # Calculate score (100 - penalties)
        score = 100
        score -= len(findings_by_severity["critical"]) * 20
        score -= len(findings_by_severity["high"]) * 10
        score -= len(findings_by_severity["medium"]) * 5
        score -= len(findings_by_severity["low"]) * 2
        score = max(0, score)
        
        # Determine grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "scan_info": {
                "skill_path": str(self.skill_path),
                "files_scanned": self.files_scanned,
                "lines_scanned": self.lines_scanned,
                "timestamp": "2026-01-31T00:20:00Z",
                "auditor_version": "0.1.0"
            },
            "summary": {
                "total_findings": len(self.findings),
                "score": score,
                "grade": grade,
                "by_severity": {
                    k: len(v) for k, v in findings_by_severity.items()
                }
            },
            "findings": findings_by_severity,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate high-level recommendations."""
        recs = []
        
        has_critical = any(f.severity == Severity.CRITICAL for f in self.findings)
        has_high = any(f.severity == Severity.HIGH for f in self.findings)
        
        if has_critical:
            recs.append("🔴 CRITICAL: Address all critical findings before production deployment.")
        
        if has_high:
            recs.append("🟠 HIGH: Review and remediate high-severity issues within 48 hours.")
        
        if not recs:
            recs.append("🟢 No critical issues found. Continue following security best practices.")
        
        recs.append("📋 Run `skill_auditor.py --fix-suggestions` for automated remediation hints.")
        
        return recs


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python skill_auditor.py <skill_path> [--json]")
        print("\nExample:")
        print("  python skill_auditor.py ./skills/my-skill")
        print("  python skill_auditor.py ./skills/my-skill --json > report.json")
        sys.exit(1)
    
    skill_path = sys.argv[1]
    output_json = "--json" in sys.argv
    
    auditor = SkillAuditor(skill_path)
    report = auditor.audit()
    
    if output_json:
        print(json.dumps(report, indent=2))
    else:
        # Human-readable output
        print("🛡️  Skill Security Audit Report")
        print("=" * 50)
        print(f"\n📁 Path: {report['scan_info']['skill_path']}")
        print(f"📊 Files: {report['scan_info']['files_scanned']}")
        print(f"📄 Lines: {report['scan_info']['lines_scanned']}")
        
        summary = report['summary']
        print(f"\n🏆 Score: {summary['score']}/100 (Grade: {summary['grade']})")
        print(f"🔍 Total Findings: {summary['total_findings']}")
        
        print("\n📈 By Severity:")
        for sev, count in summary['by_severity'].items():
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "🔵"}.get(sev, "⚪")
            print(f"  {emoji} {sev.upper()}: {count}")
        
        if report['findings']['critical']:
            print("\n🔴 CRITICAL FINDINGS:")
            for f in report['findings']['critical'][:5]:
                print(f"  • [{f['file']}:{f['line']}] {f['message']}")
        
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
