"""
ContinuousOptimalBinning transform caching testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2025

import time

import numpy as np

from optbinning import ContinuousOptimalBinning


def test_cache_creation_on_first_transform():
    np.random.seed(42)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.5

    optb = ContinuousOptimalBinning(dtype='numerical')
    optb.fit(x, y)

    # Cache should be empty initially
    assert len(optb._transform_cache) == 0

    # First transform should create cache entry
    result1 = optb.transform(x, metric='mean')
    assert len(optb._transform_cache) == 1
    assert result1.shape == (1000,)


def test_cache_reuse_on_second_transform():
    np.random.seed(42)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.5

    optb = ContinuousOptimalBinning(dtype='numerical')
    optb.fit(x, y)

    # First transform
    result1 = optb.transform(x, metric='mean')
    cache_size_after_first = len(optb._transform_cache)

    # Second transform with same parameters
    result2 = optb.transform(x, metric='mean')

    # Cache size should remain the same
    assert len(optb._transform_cache) == cache_size_after_first
    assert len(optb._transform_cache) == 1

    # Results should be identical
    assert np.allclose(result1, result2)


def test_multiple_cache_entries_for_different_metrics():
    np.random.seed(42)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.5

    optb = ContinuousOptimalBinning(dtype='numerical')
    optb.fit(x, y)

    # Transform with different metrics
    result_mean = optb.transform(x, metric='mean')
    assert len(optb._transform_cache) == 1

    result_indices = optb.transform(x, metric='indices')
    assert len(optb._transform_cache) == 2

    result_bins = optb.transform(x, metric='bins')
    assert len(optb._transform_cache) == 3

    # Each should produce different results
    assert not np.allclose(result_mean, result_indices)


def test_cache_key_includes_all_parameters():
    np.random.seed(42)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.5

    optb = ContinuousOptimalBinning(dtype='numerical')
    optb.fit(x, y)

    # Same metric but different metric_special
    result1 = optb.transform(x, metric='mean', metric_special=0)
    result2 = optb.transform(x, metric='mean', metric_special=1)
    assert len(optb._transform_cache) == 2

    # Same metric but different metric_missing
    result3 = optb.transform(x, metric='mean', metric_missing=1)
    assert len(optb._transform_cache) == 3

    # Same metric but different show_digits
    result4 = optb.transform(x, metric='bins', show_digits=2)
    result5 = optb.transform(x, metric='bins', show_digits=3)
    assert len(optb._transform_cache) == 5


def test_cached_results_match_original_implementation():
    np.random.seed(42)
    x = np.random.randn(500)
    y = x + np.random.randn(500) * 0.5

    optb = ContinuousOptimalBinning(dtype='numerical')
    optb.fit(x, y)

    # Generate some test data
    x_test = np.random.randn(100)

    # First transform (creates cache)
    result1 = optb.transform(x_test, metric='mean')

    # Second transform (uses cache)
    result2 = optb.transform(x_test, metric='mean')

    # Results should be exactly identical (not just close)
    assert np.array_equal(result1, result2)


def test_cache_with_different_input_arrays():
    np.random.seed(42)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.5

    optb = ContinuousOptimalBinning(dtype='numerical')
    optb.fit(x, y)

    # Create two different test arrays
    x_test1 = np.random.randn(100)
    x_test2 = np.random.randn(150)

    # Transform both arrays with same parameters
    result1 = optb.transform(x_test1, metric='mean')
    result2 = optb.transform(x_test2, metric='mean')

    # Cache should have only one entry (same parameters)
    assert len(optb._transform_cache) == 1

    # Results should have different shapes
    assert result1.shape == (100,)
    assert result2.shape == (150,)


def test_cache_performance_benefit():
    np.random.seed(42)
    x_train = np.random.randn(1000)
    y_train = x_train + np.random.randn(1000) * 0.5

    optb = ContinuousOptimalBinning(dtype='numerical')
    optb.fit(x_train, y_train)

    # Generate 1000 single-record test samples
    x_test_samples = [np.random.randn(1) for _ in range(1000)]

    # Test WITH caching: 1000 calls, cache persists
    print("\n=== WITH CACHING ===")
    start = time.time()
    results_cached = []
    for x_test in x_test_samples:
        results_cached.append(optb.transform(x_test, metric='mean'))
    time_with_cache = time.time() - start
    print(f"1000 calls with cache: {time_with_cache:.6f}s")
    print(f"Cache size: {len(optb._transform_cache)}")
    print(f"Average per call: {time_with_cache / 1000 * 1000:.3f}ms")

    # Test WITHOUT caching: 1000 calls, clear cache after each
    print("\n=== WITHOUT CACHING ===")
    start = time.time()
    results_uncached = []
    for x_test in x_test_samples:
        optb._transform_cache.clear()  # Force cache miss
        results_uncached.append(optb.transform(x_test, metric='mean'))
    time_without_cache = time.time() - start
    print(f"1000 calls without cache: {time_without_cache:.6f}s")
    print(f"Average per call: {time_without_cache / 1000 * 1000:.3f}ms")

    # Calculate speedup
    speedup = time_without_cache / time_with_cache
    print(f"\n=== RESULTS ===")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {time_without_cache - time_with_cache:.6f}s ({(1 - 1/speedup) * 100:.1f}% faster)")

    # Results should match (same random seed ensures same samples)
    for i in range(len(results_cached)):
        assert np.array_equal(results_cached[i], results_uncached[i])

    # Caching should provide measurable speedup (typically ~2-3x observed)
    assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"
