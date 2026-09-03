#!/usr/bin/env python3
"""
LMCache LFU Cache Policy

This is the Least Frequently Used (LFU) cache policy from LMCache.
Source: https://github.com/LMCache/LMCache

Current Implementation:
- Uses SortedDict to map frequency → keys (O(log N) operations)
- TODO comment in original code (line 25-28) notes this could be O(1)
  by tracking min frequency instead

OpenEvolve Target:
Optimize the LFU cache policy implementation to achieve O(1) operations
for cache hits, puts, and evictions while maintaining correctness.
"""

from typing import Any, Optional
from sortedcontainers import SortedDict


class CacheEngineKey:
    """Simplified CacheEngineKey for standalone testing"""
    def __init__(self, key_id: int):
        self.key_id = key_id

    def __hash__(self):
        return hash(self.key_id)

    def __eq__(self, other):
        return self.key_id == other.key_id

    def __repr__(self):
        return f"Key({self.key_id})"


class CacheEntry:
    """Cache entry with eviction control"""
    def __init__(self, value: Any, can_evict: bool = True):
        self.value = value
        self.can_evict = can_evict


# EVOLVE-BLOCK-START id="lfu-cache-complexity"
class LFUCachePolicy:
    """
    LFU cache policy implementation.

    Current approach uses SortedDict which provides O(log N) operations.
    The TODO in the original implementation suggests optimizing to O(1)
    by tracking minimum frequency.
    """

    def __init__(self):
        # SortedDict provides O(log N) operations for insertion, deletion, lookup
        # freq → {key → None} mapping
        # Using dict as a set (value is None)
        self.freq_to_keys: SortedDict = SortedDict()

        # Track frequency for each key for quick lookup
        self.key_to_freq: dict[CacheEngineKey, int] = {}

    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, CacheEntry],
    ) -> None:
        """
        Update internal state when a cache entry is accessed (cache hit).
        Increment the frequency of the accessed key.
        """
        curr_freq = self.key_to_freq[key]

        # Remove from current frequency bucket
        self.freq_to_keys[curr_freq].pop(key)
        if not self.freq_to_keys[curr_freq]:
            self.freq_to_keys.pop(curr_freq)

        # Add to next frequency bucket
        curr_freq += 1
        self.key_to_freq[key] = curr_freq

        if curr_freq not in self.freq_to_keys:
            self.freq_to_keys[curr_freq] = {key: None}
        else:
            self.freq_to_keys[curr_freq][key] = None

    def update_on_put(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Update internal state when a new cache entry is stored.
        Initialize the frequency for the new key to 1.
        """
        # Initialize the frequency for the new key
        self.key_to_freq[key] = 1

        if 1 not in self.freq_to_keys:
            self.freq_to_keys[1] = {key: None}
        else:
            self.freq_to_keys[1][key] = None

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Update internal state when a cache entry is force evicted.
        Remove all tracking for this key.
        """
        freq = self.key_to_freq.pop(key, None)
        if not freq:
            return

        self.freq_to_keys[freq].pop(key)
        if not self.freq_to_keys[freq]:
            self.freq_to_keys.pop(freq)

    def get_evict_candidates(
        self,
        cache_dict: dict[CacheEngineKey, CacheEntry],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        """
        Get keys to evict based on LFU policy.

        Evicts entries with lowest frequency first.
        Within same frequency, uses FIFO (first inserted gets evicted first).
        Respects can_evict flag on cache entries.

        Note: We do best effort to get eviction candidates so the number
        of returned keys might be smaller than num_candidates.
        """
        evict_keys = []
        evict_freqs = []

        # Iterate through frequencies from lowest to highest
        # SortedDict maintains sorted order
        for curr_min_freq, fifo_keys in self.freq_to_keys.items():
            for key in fifo_keys:
                # Skip pinned entries
                if not cache_dict[key].can_evict:
                    continue

                evict_keys.append(key)
                evict_freqs.append(curr_min_freq)
                self.key_to_freq.pop(key)

                if len(evict_keys) == num_candidates:
                    break

            if len(evict_keys) == num_candidates:
                break

        # Clean up frequency buckets
        for freq, key in zip(evict_freqs, evict_keys, strict=False):
            self.freq_to_keys[freq].pop(key)
            if not self.freq_to_keys[freq]:
                self.freq_to_keys.pop(freq)

        return evict_keys
# EVOLVE-BLOCK-END


# Test interface for evaluator
def create_policy():
    """Factory function to create LFU policy instance"""
    return LFUCachePolicy()


if __name__ == "__main__":
    # Basic smoke test
    policy = create_policy()
    cache_dict = {}

    # Add some entries
    for i in range(5):
        key = CacheEngineKey(i)
        cache_dict[key] = CacheEntry(f"value_{i}")
        policy.update_on_put(key)

    # Access some entries multiple times
    key0 = CacheEngineKey(0)
    key1 = CacheEngineKey(1)
    policy.update_on_hit(key0, cache_dict)
    policy.update_on_hit(key0, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    # Get eviction candidates
    candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    print(f"Eviction candidates: {candidates}")
    print(f"Expected: keys with lowest frequency (2, 3, or 4)")
