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
from pathlib import Path
from typing import Any

from lib.core.data import options
from lib.utils.file import FileUtils


class EloManager:
    """Manages ELO ratings for wordlist entries with safeguards against false positives"""

    def __init__(self, wordlist_path: str) -> None:
        self.wordlist_path = wordlist_path
        self.wordlist_name = Path(wordlist_path).name
        self.elo_file_path = self._get_elo_file_path()
        
        # ELO data structure
        self.metadata = {
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
        
        # Load existing ELO data if available
        self._load()

    def _get_elo_file_path(self) -> str:
        """Determine the ELO file path based on configuration"""
        elo_dir = options.get("elo_directory", ".")
        
        # Handle %temp_folder% placeholder
        if elo_dir == "%temp_folder%":
            elo_dir = tempfile.gettempdir()
        # Handle relative paths
        elif not os.path.isabs(elo_dir):
            elo_dir = os.path.abspath(elo_dir)
        
        # Ensure directory exists
        FileUtils.create_dir(elo_dir)
        
        # Create ELO filename
        elo_filename = f"{self.wordlist_name}.elo.json"
        return os.path.join(elo_dir, elo_filename)

    def _load(self) -> None:
        """Load ELO data from file"""
        if not os.path.exists(self.elo_file_path):
            return
        
        try:
            with open(self.elo_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data.get("metadata", self.metadata)
                self.words = data.get("words", {})
                self.pending_review = data.get("pending_review")
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, start fresh
            pass

    def save(self) -> None:
        """Save ELO data to file"""
        data = {
            "metadata": self.metadata,
            "words": self.words,
        }
        
        if self.pending_review:
            data["pending_review"] = self.pending_review
        
        try:
            with open(self.elo_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            # Log error but don't crash
            from lib.core.logger import logger
            logger.error(f"Failed to save ELO file: {e}")

    def initialize_word(self, word: str) -> None:
        """Add a new word with initial ELO score if it doesn't exist"""
        if word not in self.words:
            self.words[word] = {
                "elo": options.get("elo_initial", 1000),
                "hits": 0,
                "misses": 0,
            }

    def sync_wordlist(self, wordlist: list[str]) -> None:
        """Ensure all words in wordlist exist in ELO data"""
        for word in wordlist:
            self.initialize_word(word)

    def apply_time_decay(self) -> None:
        """Apply time decay to all ELO scores"""
        decay_rate = options.get("elo_time_decay_per_scan", 0.01)
        initial_elo = options.get("elo_initial", 1000)
        
        for word_data in self.words.values():
            # Move ELO towards initial value
            current_elo = word_data["elo"]
            word_data["elo"] = current_elo - (current_elo - initial_elo) * decay_rate

    def sort_wordlist(self, wordlist: list[str]) -> list[str]:
        """Sort wordlist by ELO scores (descending)"""
        # Ensure all words are initialized
        self.sync_wordlist(wordlist)
        
        # Sort by ELO score
        return sorted(
            wordlist,
            key=lambda word: self.words.get(word, {}).get("elo", options.get("elo_initial", 1000)),
            reverse=True
        )

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
        """Finalize the current scan and update ELO scores"""
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
        
        # Apply time decay
        self.apply_time_decay()
        
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
        rate_limit_threshold = options.get("elo_rate_limit_threshold", 5)
        if self.rate_limit_count >= rate_limit_threshold:
            return True
        
        # Check for dominant status code
        suspicious_threshold = options.get("elo_suspicious_threshold", 0.8)
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
            if result["is_hit"]:
                change_data = {
                    "status": result["status"],
                    "elo_gain": options.get("elo_hit_gain", 100),
                }
                
                # Categorize based on whether it matches dominant status
                if result["status"] == dominant_status:
                    spam_changes[path] = change_data
                else:
                    legit_changes[path] = change_data
            else:
                # Misses
                change_data = {
                    "status": result["status"],
                    "elo_penalty": options.get("elo_miss_penalty", 25),
                }
                
                if result["status"] == dominant_status:
                    spam_changes[path] = change_data
                else:
                    legit_changes[path] = change_data
        
        # Determine reason
        if self.rate_limit_count >= options.get("elo_rate_limit_threshold", 5):
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
        """Apply scan results to update ELO scores"""
        if changes is None:
            changes = {}
            for path, result in self.current_scan_results.items():
                changes[path] = result
        
        for path, data in changes.items():
            self.initialize_word(path)
            
            # Check if this is a hit or miss
            if "elo_gain" in data:
                # Hit - increase ELO
                gain = data.get("elo_gain", options.get("elo_hit_gain", 100))
                self.words[path]["elo"] += gain
                self.words[path]["hits"] += 1
            elif "elo_penalty" in data:
                # Miss - decrease ELO
                penalty = data.get("elo_penalty", options.get("elo_miss_penalty", 25))
                self.words[path]["elo"] -= penalty
                self.words[path]["misses"] += 1
            elif data.get("is_hit", False):
                # Hit from current scan results
                gain = options.get("elo_hit_gain", 100)
                self.words[path]["elo"] += gain
                self.words[path]["hits"] += 1
            else:
                # Miss from current scan results
                penalty = options.get("elo_miss_penalty", 25)
                self.words[path]["elo"] -= penalty
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
        """Reset ELO file - clear all data"""
        self.metadata = {
            "total_scans": 0,
            "last_scan": None,
        }
        self.words = {}
        self.pending_review = None
        self.save()

    @staticmethod
    def list_elo_files(elo_directory: str = None) -> list[str]:
        """List all ELO files in the specified directory"""
        if elo_directory is None:
            elo_directory = options.get("elo_directory", ".")
        
        # Handle %temp_folder% placeholder
        if elo_directory == "%temp_folder%":
            elo_directory = tempfile.gettempdir()
        # Handle relative paths
        elif not os.path.isabs(elo_directory):
            elo_directory = os.path.abspath(elo_directory)
        
        if not os.path.exists(elo_directory):
            return []
        
        elo_files = []
        for filename in os.listdir(elo_directory):
            if filename.endswith(".elo.json"):
                full_path = os.path.join(elo_directory, filename)
                elo_files.append(full_path)
        
        return sorted(elo_files)

