# -*- coding: utf-8 -*-
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Any

from lib.core.data import options
from lib.utils.file import FileUtils


# Pre-calculated z-scores for common confidence levels (avoid scipy dependency)
Z_SCORES = {
    0.90: 1.645,
    0.95: 1.96,
    0.975: 2.24,
    0.99: 2.576,
    0.995: 2.807,
    0.999: 3.291,
}


def get_z_score(confidence: float) -> float:
    """
    Get z-score for confidence level.
    Uses pre-calculated values for common levels.
    
    Args:
        confidence: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Z-score for the confidence level
    """
    if confidence in Z_SCORES:
        return Z_SCORES[confidence]
    
    # Find closest pre-calculated value
    closest = min(Z_SCORES.keys(), key=lambda x: abs(x - confidence))
    return Z_SCORES[closest]


def wilson_score_lower_bound(hits: int, total: int, confidence: float = 0.95) -> float:
    """
    Calculate lower bound of Wilson score confidence interval.
    Returns conservative estimate of success rate.
    
    This is used by Reddit, HackerNews, and other sites for ranking.
    It provides a statistically sound lower bound on the true success rate.
    
    Args:
        hits: Number of successful observations
        total: Total observations (hits + misses)
        confidence: Confidence level (default 0.95 for 95%)
    
    Returns:
        Lower bound of Wilson score interval (0.0 to 1.0)
    """
    if total == 0:
        return 0.0
    
    z = get_z_score(confidence)
    p = hits / total
    
    denominator = 1 + z**2 / total
    center = p + z**2 / (2 * total)
    margin = z * sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    
    return max(0.0, (center - margin) / denominator)


class RatingManager:
    """
    Manages Bayesian success rate ratings for wordlist entries.
    
    Uses Bayesian inference with Beta priors to estimate path discovery probabilities.
    Ranks paths using Wilson Score confidence intervals for statistical soundness.
    Includes safeguards against false positives from suspicious scans.
    """

    def __init__(self, wordlist_path: str) -> None:
        self.wordlist_path = wordlist_path
        self.wordlist_name = Path(wordlist_path).name
        self.rating_file_path = self._get_rating_file_path()
        
        # Bayesian priors (default: 5% prior success rate)
        self.priors = {
            "alpha": options.get("prior_alpha", 5),
            "beta": options.get("prior_beta", 95),
        }
        
        # Rating data structure
        self.metadata = {
            "version": 2,  # Version 2 = Bayesian ratings
            "total_scans": 0,
            "last_scan": None,
        }
        self.words = {}
        self.pending_review = None
        
        # Tracking for current scan
        self.current_scan_results = defaultdict(lambda: {"status": None, "is_hit": False})
        self.status_code_counts = defaultdict(int)
        self.total_requests = 0
        self.rate_limit_count = 0
        
        # Load existing rating data if available
        self._load()

    def _get_rating_file_path(self) -> str:
        """Determine the rating file path based on configuration"""
        # Support both old 'elo_directory' and new 'ratings_directory' for backward compat
        rating_dir = options.get("ratings_directory") or options.get("elo_directory", ".")
        
        # Handle %temp_folder% placeholder
        if rating_dir == "%temp_folder%":
            rating_dir = tempfile.gettempdir()
        # Handle relative paths
        elif not os.path.isabs(rating_dir):
            rating_dir = os.path.abspath(rating_dir)
        
        # Ensure directory exists
        FileUtils.create_dir(rating_dir)
        
        # Keep .elo.json extension for backward compatibility
        rating_filename = f"{self.wordlist_name}.elo.json"
        return os.path.join(rating_dir, rating_filename)

    def _load(self) -> None:
        """Load rating data from file with backward compatibility"""
        if not os.path.exists(self.rating_file_path):
            return
        
        try:
            with open(self.rating_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check if old ELO format (version 1) or new Bayesian format (version 2)
                metadata = data.get("metadata", {})
                if "version" not in metadata or metadata.get("version") == 1:
                    # Migrate from ELO to Bayesian
                    self._migrate_from_elo(data)
                else:
                    # Load new format
                    self.metadata = metadata
                    self.words = data.get("words", {})
                    self.priors = data.get("priors", self.priors)
                    self.pending_review = data.get("pending_review")
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, start fresh
            pass

    def _migrate_from_elo(self, old_data: dict) -> None:
        """
        Convert old ELO format to Bayesian format.
        Preserves hits/misses data, discards ELO scores.
        """
        self.metadata = old_data.get("metadata", {})
        self.metadata["version"] = 2
        
        for word, old_word_data in old_data.get("words", {}).items():
            # Keep hits/misses, discard ELO score
            # NOTE: Scores are NOT stored - they're calculated on-demand
            self.words[word] = {
                "hits": old_word_data.get("hits", 0),
                "misses": old_word_data.get("misses", 0),
            }

    def save(self) -> None:
        """Save rating data to file"""
        data = {
            "metadata": self.metadata,
            "priors": self.priors,
            "words": self.words,
        }
        
        if self.pending_review:
            data["pending_review"] = self.pending_review
        
        try:
            with open(self.rating_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            # Log error but don't crash
            from lib.core.logger import logger
            logger.error(f"Failed to save rating file: {e}")

    def apply_time_decay_to_word(self, word_data: dict, days_since_scan: int) -> dict:
        """
        Apply time decay to observation counts if enabled.
        
        Args:
            word_data: Word statistics with hits/misses
            days_since_scan: Days since last scan for this word
        
        Returns:
            Updated word_data with decayed hits/misses
        """
        if not options.get("time_decay_enabled", False):
            return word_data
        
        threshold_days = options.get("time_decay_threshold_days", 30)
        
        # Only decay if past threshold
        if days_since_scan <= threshold_days:
            return word_data
        
        # Calculate decay factor
        decay_rate = options.get("time_decay_rate", 0.01)  # per month
        months_past_threshold = (days_since_scan - threshold_days) / 30.0
        decay_factor = (1 - decay_rate) ** months_past_threshold
        
        # Apply decay to observation counts (not scores directly)
        # This preserves the Bayesian nature while aging old data
        word_data["hits"] = max(0, int(word_data["hits"] * decay_factor))
        word_data["misses"] = max(0, int(word_data["misses"] * decay_factor))
        
        return word_data

    def calculate_scores(self, hits: int, misses: int) -> tuple[float, float]:
        """
        Calculate Bayesian score and Wilson score on-demand.
        
        Scores are calculated dynamically from stored hits/misses to ensure
        they're always current with the configured priors and confidence level.
        
        Args:
            hits: Number of successful observations
            misses: Number of failed observations
        
        Returns:
            Tuple of (bayesian_mean, wilson_score)
        """
        # Bayesian calculation with priors
        alpha = self.priors["alpha"] + hits
        beta = self.priors["beta"] + misses
        
        # Bayesian mean (expected success rate)
        bayesian_mean = alpha / (alpha + beta)
        
        # Wilson score on Bayesian posterior (not on raw observations)
        # This ensures Wilson is more conservative than Bayesian mean
        total_posterior = alpha + beta
        confidence = options.get("confidence_level", 0.95)
        wilson = wilson_score_lower_bound(alpha, total_posterior, confidence)
        
        return bayesian_mean, wilson

    def calculate_bayesian_score(self, word_data: dict, last_scan_date: str = None) -> dict:
        """
        [DEPRECATED] Use calculate_scores() instead.
        Calculate Bayesian success rate - kept for backward compatibility.
        
        Note: This method does NOT store scores in word_data to maintain
        data consistency when configuration changes.
        """
        # Apply time decay if enabled and last scan date available
        if last_scan_date and options.get("time_decay_enabled", False):
            try:
                last_scan = datetime.fromisoformat(last_scan_date)
                days_since = (datetime.now() - last_scan).days
                word_data = self.apply_time_decay_to_word(word_data, days_since)
            except (ValueError, TypeError):
                # If date parsing fails, skip time decay
                pass
        
        return word_data

    def initialize_word(self, word: str) -> None:
        """Add a new word with initial data if it doesn't exist"""
        if word not in self.words:
            self.words[word] = {
                "hits": 0,
                "misses": 0,
            }

    def sync_wordlist(self, wordlist: list[str]) -> None:
        """Ensure all words in wordlist exist in rating data"""
        for word in wordlist:
            self.initialize_word(word)

    def sort_wordlist(self, wordlist: list[str]) -> list[str]:
        """
        Sort wordlist by Wilson scores (descending - highest scores first).
        
        Wilson scores are calculated on-demand from current hits/misses,
        ensuring they reflect current configuration (priors, confidence level).
        """
        # Ensure all words are initialized
        self.sync_wordlist(wordlist)
        
        # Sort by Wilson score (descending - highest first)
        # Scores are calculated on-demand, not stored
        def get_wilson_score(word: str) -> float:
            word_data = self.words.get(word, {"hits": 0, "misses": 0})
            _, wilson = self.calculate_scores(
                word_data.get("hits", 0),
                word_data.get("misses", 0)
            )
            return wilson
        
        return sorted(wordlist, key=get_wilson_score, reverse=True)

    def record_result(self, path: str, status_code: int, is_hit: bool) -> None:
        """Record a scan result for the current scan"""
        self.current_scan_results[path] = {
            "status": status_code,
            "is_hit": is_hit,
        }
        self.status_code_counts[status_code] += 1
        self.total_requests += 1
        
        # Track rate limiting
        if status_code == 429:
            self.rate_limit_count += 1

    def finalize_scan(self) -> None:
        """Finalize the current scan and update rating scores"""
        if self.total_requests == 0:
            return
        
        # Check for suspicious behavior
        if self._is_scan_suspicious():
            self._mark_for_review()
        else:
            # Apply changes directly
            self._apply_scan_results()
        
        # Update metadata
        self.metadata["total_scans"] += 1
        self.metadata["last_scan"] = datetime.now().isoformat()
        
        # Reset current scan tracking
        self.current_scan_results.clear()
        self.status_code_counts.clear()
        self.total_requests = 0
        self.rate_limit_count = 0
        
        # Save to file
        self.save()

    def _is_scan_suspicious(self) -> bool:
        """Detect if the current scan shows suspicious behavior"""
        if self.total_requests == 0:
            return False
        
        # Check for rate limiting
        rate_limit_threshold = options.get("rate_limit_threshold") or options.get("elo_rate_limit_threshold", 5)
        if self.rate_limit_count >= rate_limit_threshold:
            return True
        
        # Check for dominant status code
        suspicious_threshold = options.get("suspicious_threshold") or options.get("elo_suspicious_threshold", 0.8)
        max_count = max(self.status_code_counts.values()) if self.status_code_counts else 0
        
        if max_count / self.total_requests >= suspicious_threshold:
            return True
        
        return False

    def _get_dominant_status_code(self) -> int | None:
        """Get the most common status code from the current scan"""
        if not self.status_code_counts:
            return None
        return max(self.status_code_counts.items(), key=lambda x: x[1])[0]

    def _mark_for_review(self) -> None:
        """Mark current scan results for user review"""
        dominant_status = self._get_dominant_status_code()
        
        # Categorize results
        spam_changes = {}
        legit_changes = {}
        
        for path, result in self.current_scan_results.items():
            change_data = {
                "status": result["status"],
                "is_hit": result["is_hit"],
            }
            
            # Categorize based on whether it matches dominant status
            if result["status"] == dominant_status:
                spam_changes[path] = change_data
            else:
                legit_changes[path] = change_data
        
        # Determine reason
        rate_limit_threshold = options.get("rate_limit_threshold") or options.get("elo_rate_limit_threshold", 5)
        if self.rate_limit_count >= rate_limit_threshold:
            reason = f"rate_limiting_detected ({self.rate_limit_count} 429 responses)"
        else:
            dominant_percentage = (self.status_code_counts[dominant_status] / self.total_requests) * 100
            reason = f"{dominant_percentage:.0f}% responses with status {dominant_status}"
        
        self.pending_review = {
            "scan_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "dominant_status": dominant_status,
            "spam_changes": spam_changes,
            "legit_changes": legit_changes,
        }

    def _apply_scan_results(self, changes: dict[str, dict] | None = None) -> None:
        """Apply scan results to update rating scores"""
        if changes is None:
            changes = {}
            for path, result in self.current_scan_results.items():
                changes[path] = result
        
        for path, data in changes.items():
            self.initialize_word(path)
            
            # Update hits/misses based on result
            # NOTE: Scores are NOT stored - they're calculated on-demand
            if data.get("is_hit", False):
                self.words[path]["hits"] += 1
            else:
                self.words[path]["misses"] += 1

    def has_pending_review(self) -> bool:
        """Check if there are pending review items"""
        return self.pending_review is not None

    def get_pending_review_summary(self) -> dict[str, Any]:
        """Get a summary of pending review for display"""
        if not self.pending_review:
            return {}
        
        spam_count = len(self.pending_review.get("spam_changes", {}))
        legit_count = len(self.pending_review.get("legit_changes", {}))
        
        # Get sample words for each category
        spam_samples = {}
        legit_samples = {}
        
        for path, data in list(self.pending_review.get("spam_changes", {}).items())[:10]:
            status = data.get("status", "?")
            spam_samples.setdefault(status, []).append(path)
        
        for path, data in list(self.pending_review.get("legit_changes", {}).items())[:10]:
            status = data.get("status", "?")
            legit_samples.setdefault(status, []).append(path)
        
        return {
            "scan_id": self.pending_review.get("scan_id"),
            "timestamp": self.pending_review.get("timestamp"),
            "reason": self.pending_review.get("reason"),
            "dominant_status": self.pending_review.get("dominant_status"),
            "spam_count": spam_count,
            "legit_count": legit_count,
            "spam_samples": spam_samples,
            "legit_samples": legit_samples,
        }

    def apply_pending_review(self, choice: str) -> None:
        """Apply pending review based on user choice"""
        if not self.pending_review:
            return
        
        if choice == "a":  # Accept all
            all_changes = {}
            for path, data in self.pending_review.get("spam_changes", {}).items():
                all_changes[path] = data
            for path, data in self.pending_review.get("legit_changes", {}).items():
                all_changes[path] = data
            self._apply_scan_results(all_changes)
        elif choice == "k":  # Keep legitimate only
            self._apply_scan_results(self.pending_review.get("legit_changes", {}))
        elif choice == "d":  # Discard all
            pass  # Do nothing
        
        # Clear pending review
        self.pending_review = None
        self.save()

    def reset(self) -> None:
        """Reset rating file - clear all data"""
        self.metadata = {
            "version": 2,
            "total_scans": 0,
            "last_scan": None,
        }
        self.words = {}
        self.pending_review = None
        self.save()

    @staticmethod
    def list_rating_files(rating_directory: str = None) -> list[str]:
        """List all rating files in the specified directory"""
        if rating_directory is None:
            # Support both old and new option names
            rating_directory = options.get("ratings_directory") or options.get("elo_directory", ".")
        
        # Handle %temp_folder% placeholder
        if rating_directory == "%temp_folder%":
            rating_directory = tempfile.gettempdir()
        # Handle relative paths
        elif not os.path.isabs(rating_directory):
            rating_directory = os.path.abspath(rating_directory)
        
        if not os.path.exists(rating_directory):
            return []
        
        rating_files = []
        for filename in os.listdir(rating_directory):
            if filename.endswith(".elo.json"):
                full_path = os.path.join(rating_directory, filename)
                rating_files.append(full_path)
        
        return sorted(rating_files)


# Backward compatibility: Create an alias for old code
EloManager = RatingManager

