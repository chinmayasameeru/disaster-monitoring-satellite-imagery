#!/usr/bin/env python3
"""
Alert Manager
Sends notifications when hazard thresholds are exceeded.
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Hazard alert rule."""

    hazard_type: str
    threshold: float
    severity: str  # "LOW", "MODERATE", "HIGH", "CRITICAL", "EXTREME"
    recipients: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=lambda: ["email", "webhook"])


class AlertManager:
    """Manages hazard alerts and notifications."""

    def __init__(
        self,
        smtp_server: str = "",
        smtp_port: int = 587,
        sender: str = "",
        sender_password: str = "",
        recipients: List[str] = None,
        webhook_url: str = "",
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.sender_password = sender_password
        self.recipients = recipients or []
        self.webhook_url = webhook_url
        self.rules: List[AlertRule] = []
        self.alert_history: List[Dict] = []

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)

    def check_triggers(self, results: list) -> List[Dict]:
        """Check results against alert rules and return triggered alerts."""
        triggered = []
        for result in results:
            for rule in self.rules:
                if result.hazard_type == rule.hazard_type:
                    summary = result.summary()
                    if summary["max_risk"] >= rule.threshold:
                        alert = {
                            "timestamp": datetime.now().isoformat(),
                            "hazard_type": result.hazard_type,
                            "severity": rule.severity,
                            "max_risk": summary["max_risk"],
                            "mean_risk": summary["mean_risk"],
                            "coverage_pct": summary["coverage_pct"],
                        }
                        triggered.append(alert)
                        self.alert_history.append(alert)
        return triggered

    def send_alert(self, alert: Dict):
        """Send an alert via configured channels."""
        if self.smtp_server:
            self._send_email(alert)
        if self.webhook_url:
            self._send_webhook(alert)

    def _send_email(self, alert: Dict):
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = (
                f"[HAZARD ALERT] {alert['hazard_type'].upper()} - {alert['severity']}"
            )

            body = f"""
Hazard Alert - {alert['timestamp']}

Type: {alert['hazard_type']}
Severity: {alert['severity']}
Max Risk Score: {alert['max_risk']:.3f}
Mean Risk Score: {alert['mean_risk']:.3f}
Affected Area: {alert['coverage_pct']:.1f}%

This is an automated alert from the Disaster Monitoring System.
            """
            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender, self.sender_password)
            server.sendmail(self.sender, self.recipients, msg.as_string())
            server.quit()
            logger.info(f"Email alert sent to {self.recipients}")
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def _send_webhook(self, alert: Dict):
        """Send alert to webhook (Slack/Discord)."""
        try:
            import httpx

            payload = {
                "text": f"🚨 *{alert['severity']} ALERT*: {alert['hazard_type']}\n"
                f"Max Risk: {alert['max_risk']:.3f} | Coverage: {alert['coverage_pct']:.1f}%"
            }
            response = httpx.post(self.webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Webhook alert sent")
            else:
                logger.warning(f"Webhook failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    def save_alert_history(self, filepath: Path):
        """Save alert history to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.alert_history, f, indent=2)
