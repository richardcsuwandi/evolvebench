# Tasks

EvolveBench draws tasks from real open-source projects. Each task packages an editable program, a correctness oracle, a performance harness, and configurations for comparing evaluator designs. Every task below shows the actual initial code inside its `EVOLVE-BLOCK`, the exact region a system is allowed to change.

Want to add one? See [Contributing](contributing.md).

### `BayesianOptimization`

Optimize acquisition function random sampling in Bayesian optimization for improved performance

**Category:** algorithm-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.4, performance: 0.6 &nbsp;·&nbsp; **Tags:** bayesian-optimization, acquisition-function, performance, vectorization

??? example "View initial code (`bayes_opt/acquisition.py`)"

    ```python
        def _random_sample_minimize(
            self,
            acq: Callable[[NDArray[Float]], NDArray[Float]],
            space: TargetSpace,
            random_state: RandomState,
            n_random: int,
            n_x_seeds: int = 0,
        ) -> tuple[NDArray[Float] | None, float, NDArray[Float]]:
            """Random search to find the minimum of `acq` function.

            Parameters
            ----------
            acq : Callable
                Acquisition function to use. Should accept an array of parameters `x`.

            space : TargetSpace
                The target space over which to optimize.

            random_state : RandomState
                Random state to use for the optimization.

            n_random : int
                Number of random samples to use.

            n_x_seeds : int
                Number of top points to return, for use as starting points for L-BFGS-B.

            Returns
            -------
            x_min : np.ndarray
                Random sample minimizing the acquisition function.

            min_acq : float
                Acquisition function value at `x_min`
            """
            if n_random == 0:
                return None, np.inf, space.random_sample(n_x_seeds, random_state=random_state)
            x_tries = space.random_sample(n_random, random_state=random_state)
            ys = acq(x_tries)
            x_min = x_tries[ys.argmin()]
            min_acq = ys.min()
            if n_x_seeds != 0:
                idxs = np.argsort(ys)[:n_x_seeds]
                x_seeds = x_tries[idxs]
            else:
                x_seeds = []
            return x_min, min_acq, x_seeds
    ```

[Source repository](https://github.com/fmfn/BayesianOptimization){ .md-button }

---

### `difflib`

Optimize Python difflib Differ._fancy_replace WINDOW-based search to reduce redundant ratio() calls and improve performance while preserving correctness.

**Category:** performance-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.5, performance: 0.5 &nbsp;·&nbsp; **Tags:** difflib, recursion, algorithm, performance

??? example "View initial code (`difflib.py`)"

    ```python
            # Target: Optimize Differ._fancy_replace WINDOW-based search algorithm
            # Baseline: Overall score 0.365, RECURSION ERRORS at 500+ similar lines
            # Pathological case #119105: Size 100: 0.059s, Size 500: 7.015s (recursion error)
            #
            # Current algorithm (lines 912-984 in original difflib.py):
            # - Uses WINDOW=10 for searching corresponding line ranges
            # - For each line in b, searches within WINDOW in a
            # - Performs THREE ratio computations per comparison (real_quick_ratio, quick_ratio, ratio)
            # - Overall complexity: O(N*WINDOW*ratio_cost)
            #
            # Known bottlenecks:
            # - Nested loops (for j ... for i in arange) with expensive computations
            # - SequenceMatcher ratio computations called repeatedly
            # - No early termination heuristics
            # - WINDOW size is fixed, not adaptive
            # - Character-level sequence matching (lines 960-975) is expensive
            #
            # Optimization ideas:
            # - Adaptive WINDOW sizing based on input characteristics
            # - Caching of ratio computations
            # - Better early exit conditions
            # - Parallel ratio computations
            # - Alternative matching algorithms (e.g., hash-based)
            # - Limit character-level matching to truly similar lines
            # - Use cheaper similarity metrics for initial filtering
            cutoff = 0.74999
            cruncher = SequenceMatcher(self.charjunk)
            crqr = cruncher.real_quick_ratio
            cqr = cruncher.quick_ratio
            cr = cruncher.ratio

            WINDOW = 10
            best_i = best_j = None
            dump_i, dump_j = alo, blo
            for j in range(blo, bhi):
                cruncher.set_seq2(b[j])
                aequiv = alo + (j - blo)
                arange = range(max(aequiv - WINDOW, dump_i),
                               min(aequiv + WINDOW + 1, ahi))
                if not arange:
                    break
                best_ratio = cutoff
                for i in arange:
                    cruncher.set_seq1(a[i])
                    if (crqr() > best_ratio
                          and cqr() > best_ratio
                          and cr() > best_ratio):
                        best_i, best_j, best_ratio = i, j, cr()

                if best_i is None:
                    continue

                yield from self._fancy_helper(a, dump_i, best_i,
                                              b, dump_j, best_j)
                aelt, belt = a[best_i], b[best_j]
                if aelt != belt:
                    atags = btags = ""
                    cruncher.set_seqs(aelt, belt)
                    for tag, ai1, ai2, bj1, bj2 in cruncher.get_opcodes():
                        la, lb = ai2 - ai1, bj2 - bj1
                        if tag == 'replace':
                            atags += '^' * la
                            btags += '^' * lb
                        elif tag == 'delete':
                            atags += '-' * la
                        elif tag == 'insert':
                            btags += '+' * lb
                        elif tag == 'equal':
                            atags += ' ' * la
                            btags += ' ' * lb
                        else:
                            raise ValueError('unknown tag %r' % (tag,))
                    yield from self._qformat(aelt, belt, atags, btags)
                else:
                    yield '  ' + aelt
                dump_i, dump_j = best_i + 1, best_j + 1
                best_i = best_j = None

            yield from self._fancy_helper(a, dump_i, ahi,
                                          b, dump_j, bhi)
    ```

[Source repository](https://github.com/python/cpython/tree/main/Lib/difflib.py){ .md-button }

---

### `jsonschema`

Optimize python-jsonschema equality checking functions (_mapping_equal, _sequence_equal, equal) to address Python 3.12 performance regression by reducing isinstance overhead and improving type-aware dispatch

**Category:** performance-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.7, performance: 0.3 &nbsp;·&nbsp; **Tags:** jsonschema, equality-checking, type-dispatch, python-312-regression, validation, recursive-algorithms

??? example "View initial code (`jsonschema.py`)"

    ```python
    def _mapping_equal(one, two):
        """
        Check if two mappings are equal using the semantics of `equal`.

        This is the PRIMARY BOTTLENECK identified in profiling:
        - 32.6% of total execution time
        - O(n * m) where n is number of keys, m is average value complexity
        - No caching or memoization of results
        - Recursive calls for nested structures multiply the cost

        BASELINE PERFORMANCE:
        - Python 3.10: 27.8s cumulative time (56.4M calls)
        - Python 3.12: 37.2s cumulative time (56.4M calls)
        - 33% regression in Python 3.12

        OPPORTUNITY:
        This function has significant room for optimization through:
        - Caching comparison results
        - Hash-based quick rejection
        - Structural comparison before value comparison
        - Type-aware early termination
        """
        if len(one) != len(two):
            return False
        return all(
            key in two and equal(value, two[key])
            for key, value in one.items()
        )


    def _sequence_equal(one, two):
        """
        Check if two sequences are equal using the semantics of `equal`.

        Part of the recursive equality checking system.
        Also shows up in profiling but less critical than _mapping_equal.
        """
        if len(one) != len(two):
            return False
        return all(equal(i, j) for i, j in zip(one, two))


    def equal(one, two):
        """
        Check if two things are equal evading some Python type hierarchy semantics.

        Specifically in JSON Schema, evade `bool` inheriting from `int`,
        recursing into sequences to do the same.

        This is the main entry point for equality checking and dispatches to
        specialized functions based on type. The isinstance() checks themselves
        show up as bottlenecks in Python 3.12 profiling.

        PROFILING DATA:
        - Total cumulative time: 33.2s (Python 3.12) vs 28.4s (Python 3.10)
        - isinstance() calls: 34.9s (Python 3.12) vs 23.1s (Python 3.10)

        The cascading isinstance() checks are expensive when called millions of times.
        """
        if one is two:
            return True
        if isinstance(one, str) or isinstance(two, str):
            return one == two
        if isinstance(one, Sequence) and isinstance(two, Sequence):
            return _sequence_equal(one, two)
        if isinstance(one, Mapping) and isinstance(two, Mapping):
            return _mapping_equal(one, two)
        return unbool(one) == unbool(two)
    ```

[Source repository](https://github.com/python-jsonschema/jsonschema){ .md-button }

---

### `lmcache`

Optimize LMCache LFU cache policy from O(log N) to O(1) by tracking minimum frequency instead of using SortedDict

**Category:** performance-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.5, performance: 0.5 &nbsp;·&nbsp; **Tags:** lmcache, cache, lfu, llm-serving, complexity-reduction, sigcomm-2024

??? example "View initial code (`lmcache.py`)"

    ```python
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
    ```

[Source repository](https://github.com/LMCache/LMCache){ .md-button }

---

### `marko`

Optimize nested loops in marko parser's parse_source method for improved performance

**Category:** performance-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.5, performance: 0.5 &nbsp;·&nbsp; **Tags:** markdown, parser, performance, nested-loops

??? example "View initial code (`marko/source.py`)"

    ```python
    def _preprocess_text(text: str) -> str:
        return text.replace("\r\n", "\n")


    class Source:
        """Wrapper class on content to be parsed"""

        parser: Parser

        def __init__(self, text: str) -> None:
            self._buffer = _preprocess_text(text)
            self.pos = 0
            self._anchor = 0
            self._states: list[BlockElement] = []
            self.match: Match[str] | None = None
            #: Store temporary data during parsing.
            self.context = types.SimpleNamespace()

        @property
        def state(self) -> BlockElement:
            """Returns the current element state."""
            if not self._states:
                raise RuntimeError("Need to push a state first.")
            return self._states[-1]

        @property
        def root(self) -> Document:
            """Returns the root element, which is at the bottom of self._states."""
            if not self._states:
                raise RuntimeError("Need to push a state first.")
            return cast(Document, self._states[0])

        def push_state(self, element: BlockElement) -> None:
            """Push a new state to the state stack."""
            self._states.append(element)

        def pop_state(self) -> BlockElement:
            """Pop the top most state."""
            return self._states.pop()

        @contextmanager
        def under_state(self, element: BlockElement) -> Generator[Source, None, None]:
            """A context manager to enable a new state temporarily."""
            self.push_state(element)
            yield self
            self.pop_state()

        @property
        def exhausted(self) -> bool:
            """Indicates whether the source reaches the end."""
            return self.pos >= len(self._buffer)

        @property
        def prefix(self) -> str:
            """The prefix of each line when parsing."""
            return "".join(s._prefix for s in self._states)

        def _expect_re(self, regexp: Pattern[str] | str, pos: int) -> Match[str] | None:
            if isinstance(regexp, str):
                regexp = re.compile(regexp)
            return regexp.match(self._buffer, pos)

        @staticmethod
        @functools.lru_cache
        def match_prefix(prefix: str, line: str) -> int:
            """Check if the line starts with given prefix and
            return the position of the end of prefix.
            If the prefix is not matched, return -1.
            """
            m = re.match(prefix, line.expandtabs(4))
            if not m:
                if re.match(prefix, line.expandtabs(4).replace("\n", " " * 99 + "\n")):
                    return len(line) - 1
                return -1
            pos = m.end()
            if pos == 0:
                return 0
            for i in range(1, len(line) + 1):
                if len(line[:i].expandtabs(4)) >= pos:
                    return i
            return -1  # pragma: no cover

        def expect_re(self, regexp: Pattern[str] | str) -> Match[str] | None:
            """Test against the given regular expression and returns the match object.
            :param regexp: the expression to be tested.
            :returns: the match object.
            """
            prefix_len = self.match_prefix(
                self.prefix, self.next_line(require_prefix=False)  # type: ignore
            )
            if prefix_len >= 0:
                match = self._expect_re(regexp, self.pos + prefix_len)
                self.match = match
                return match
            else:
                return None

        @overload
        def next_line(self, require_prefix: Literal[False] = ...) -> str: ...

        @overload
        def next_line(self, require_prefix: Literal[True] = ...) -> str | None: ...

        def next_line(self, require_prefix: bool = True) -> str | None:
            """Return the next line in the source.

            :param require_prefix:  if False, the whole line will be returned.
                otherwise, return the line with prefix stripped or None if the prefix
                is not matched.
            """
            if require_prefix:
                m = self.expect_re(r"(?m)[^\n]*?$\n?")
            else:
                m = self._expect_re(r"(?m)[^\n]*$\n?", self.pos)
            self.match = m
            if m:
                return m.group()
            return None

        def consume(self) -> None:
            """Consume the body of source. ``pos`` will move forward."""
            if self.match:
                self.pos = self.match.end()
                if self.match.group()[-1:] == "\n":
                    self._update_prefix()
                self.match = None

        def anchor(self) -> None:
            """Pin the current parsing position."""
            self._anchor = self.pos

        def reset(self) -> None:
            """Reset the position to the last anchor."""
            self.pos = self._anchor

        def _update_prefix(self) -> None:
            for s in self._states:
                if hasattr(s, "_second_prefix"):
                    s._prefix = s._second_prefix  # type: ignore
    ```

[Source repository](https://github.com/frostming/marko){ .md-button }

---

### `networkx`

Optimize graph algorithms in NetworkX for improved performance

**Category:** algorithm-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.5, performance: 0.5 &nbsp;·&nbsp; **Tags:** graph, network, algorithm, performance

??? example "View initial code (`networkx/algorithms/centrality/betweenness.py`)"

    ```python
        S = []
        P = {}
        for v in G:
            P[v] = []
        sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
        D = {}
        sigma[s] = 1.0
        D[s] = 0
        Q = deque([s])
        while Q:  # use BFS to find shortest paths
            v = Q.popleft()
            S.append(v)
            Dv = D[v]
            sigmav = sigma[v]
            for w in G[v]:
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:  # this is a shortest path, count paths
                    sigma[w] += sigmav
                    P[w].append(v)  # predecessors
        return S, P, sigma, D
    ```

[Source repository](https://github.com/networkx/networkx){ .md-button }

---

### `pandas_rolling_rank`

Optimize pandas rolling_rank from O(n log w) C skiplist to O(n·w) Numba JIT for small windows, discovering that constant factors trump asymptotic complexity

**Category:** performance-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.7, performance: 0.3 &nbsp;·&nbsp; **Tags:** pandas, rolling-rank, jit-compilation, numba, constant-factors, counter-intuitive-optimization

??? example "View initial code (`pandas_rolling_rank.py`)"

    ```python
        def compute(self, values):
            """
            Compute rolling rank using pandas' C-based skiplist implementation.

            BASELINE PERFORMANCE:
            - Uses O(n log w) skiplist algorithm
            - Implemented in Cython for speed
            - Works well for all window sizes

            OPPORTUNITY FOR IMPROVEMENT:
            - For small windows (w < 300), JIT compilation can be faster
            - Simpler algorithm with lower constant factors
            - Potential 2-3x speedup for typical use cases
            """
            values = np.asarray(values, dtype=np.float64)

            # Use pandas' implementation directly
            series = pd.Series(values)
            result = series.rolling(
                window=self.window_size,
                min_periods=self.window_size
            ).rank(
                method=self.method,
                ascending=self.ascending,
                pct=self.pct
            ).values

            return result
    ```

[Source repository](https://github.com/pandas-dev/pandas){ .md-button }

---

### `pymoo`

Optimize non-dominated sorting in pymoo to improve performance (especially bi-objective) while preserving correctness of Pareto fronts.

**Category:** algorithm-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.6, performance: 0.4 &nbsp;·&nbsp; **Tags:** pymoo, multi-objective, non-dominated-sorting, performance

??? example "View initial code (`pymoo/functions/standard/non_dominated_sorting.py`)"

    ```python
        """Fast non-dominated sorting algorithm."""
        if "dominator" in kwargs:
            M = Dominator.calc_domination_matrix(F)
        else:
            M = dominator.calc_domination_matrix(F)

        # calculate the dominance matrix
        n = M.shape[0]

        fronts = []

        if n == 0:
            return fronts

        # final rank that will be returned
        n_ranked = 0
        ranked = np.zeros(n, dtype=int)

        # for each individual a list of all individuals that are dominated by this one
        is_dominating = [[] for _ in range(n)]

        # storage for the number of solutions dominated this one
        n_dominated = np.zeros(n)

        current_front = []

        for i in range(n):

            for j in range(i + 1, n):
                rel = M[i, j]
                if rel == 1:
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif rel == -1:
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0
                n_ranked += 1

        # append the first front to the current front
        fronts.append(current_front)

        # while not all solutions are assigned to a pareto front
        while n_ranked < n:

            next_front = []

            # for each individual in the current front
            for i in current_front:

                # all solutions that are dominated by this individuals
                for j in is_dominating[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranked[j] = 1.0
                        n_ranked += 1

            fronts.append(next_front)
            current_front = next_front

        return fronts
    ```

[Source repository](https://github.com/anyoptimization/pymoo){ .md-button }

---

### `python-chess`

Optimize chess engine algorithms for improved performance

**Category:** performance-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.6, performance: 0.4 &nbsp;·&nbsp; **Tags:** chess, game, algorithm, performance

??? example "View initial code (`chess/__init__.py`)"

    ```python
        def generate_pseudo_legal_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
            our_pieces = self.occupied_co[self.turn]

            # Generate piece moves.
            non_pawns = our_pieces & ~self.pawns & from_mask
            for from_square in scan_reversed(non_pawns):
                moves = self.attacks_mask(from_square) & ~our_pieces & to_mask
                for to_square in scan_reversed(moves):
                    yield Move(from_square, to_square)

            # Generate castling moves.
            if from_mask & self.kings:
                yield from self.generate_castling_moves(from_mask, to_mask)

            # The remaining moves are all pawn moves.
            pawns = self.pawns & self.occupied_co[self.turn] & from_mask
            if not pawns:
                return

            # Generate pawn captures.
            capturers = pawns
            for from_square in scan_reversed(capturers):
                targets = (
                    BB_PAWN_ATTACKS[self.turn][from_square] &
                    self.occupied_co[not self.turn] & to_mask)

                for to_square in scan_reversed(targets):
                    if square_rank(to_square) in [RANK_1, RANK_8]:
                        yield Move(from_square, to_square, QUEEN)
                        yield Move(from_square, to_square, ROOK)
                        yield Move(from_square, to_square, BISHOP)
                        yield Move(from_square, to_square, KNIGHT)
                    else:
                        yield Move(from_square, to_square)

            # Prepare pawn advance generation.
            if self.turn == WHITE:
                single_moves = pawns << 8 & ~self.occupied
                double_moves = single_moves << 8 & ~self.occupied & (BB_RANK_3 | BB_RANK_4)
            else:
                single_moves = pawns >> 8 & ~self.occupied
                double_moves = single_moves >> 8 & ~self.occupied & (BB_RANK_6 | BB_RANK_5)

            single_moves &= to_mask
            double_moves &= to_mask

            # Generate single pawn moves.
            for to_square in scan_reversed(single_moves):
                from_square = to_square + (8 if self.turn == BLACK else -8)

                if square_rank(to_square) in [RANK_1, RANK_8]:
                    yield Move(from_square, to_square, QUEEN)
                    yield Move(from_square, to_square, ROOK)
                    yield Move(from_square, to_square, BISHOP)
                    yield Move(from_square, to_square, KNIGHT)
                else:
                    yield Move(from_square, to_square)

            # Generate double pawn moves.
            for to_square in scan_reversed(double_moves):
                from_square = to_square + (16 if self.turn == BLACK else -16)
                yield Move(from_square, to_square)

            # Generate en passant captures.
            if self.ep_square:
                yield from self.generate_pseudo_legal_ep(from_mask, to_mask)
    ```

[Source repository](https://github.com/niklasf/python-chess){ .md-button }

---

### `python-pathfinding`

Optimize pathfinding algorithms for improved performance

**Category:** algorithm-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.5, performance: 0.5 &nbsp;·&nbsp; **Tags:** pathfinding, algorithm, performance, a-star

??? example "View initial code (`pathfinding/core/heap.py`)"

    ```python
        def pop_node(self):
            """
            Pops node off the heap. i.e. returns the one with the lowest f.

            Notes:
            1. Checks if that values is in removed_node_tuples first, if not tries
               again.
            2. We use this approach to avoid invalidating the heap structure.
            """
            node_tuple = heapq.heappop(self.open_list)
            while node_tuple in self.removed_node_tuples:
                node_tuple = heapq.heappop(self.open_list)

            if isinstance(self.grid, Graph):
                node = self.grid.node(node_tuple[2])
            elif isinstance(self.grid, Grid):
                node = self.grid.node(node_tuple[2], node_tuple[3])
            elif isinstance(self.grid, World):
                node = self.grid.grids[
                    node_tuple[4]].node(node_tuple[2], node_tuple[3])

            return node

    # ...

        def push_node(self, node):
            """
            Push node into heap.

            :param node: The node to push.
            """
            self.number_pushed = self.number_pushed + 1
            node_tuple = self._get_node_tuple(node, self.number_pushed)
            node_id = self._get_node_id(node)

            self.heap_order[node_id] = self.number_pushed

            heapq.heappush(self.open_list, node_tuple)

    # ...

        def remove_node(self, node, f):
            """
            Remove the node from the heap.

            This just stores it in a set and we just ignore the node if it does
            get popped from the heap.

            :param node: The node to remove.
            :param f: The old f value of the node.
            """
            node_id = self._get_node_id(node)
            heap_order = self.heap_order[node_id]
            node_tuple = self._get_node_tuple(node, heap_order)
            self.removed_node_tuples.add(node_tuple)
    ```

[Source repository](https://github.com/brean/python-pathfinding){ .md-button }

---

### `sympy`

Optimize SymPy Min/Max _find_localzeros algorithm from O(n²) to O(n log n) using transitivity and antichain detection to reduce redundant comparisons while preserving correctness.

**Category:** performance-optimization &nbsp;·&nbsp; **Evaluation weights:** correctness: 0.5, performance: 0.5 &nbsp;·&nbsp; **Tags:** sympy, algorithm, complexity-reduction, partial-ordering, performance

??? example "View initial code (`sympy_implementation.py`)"

    ```python
    class MinMaxBase:
        """
        Simplified MinMaxBase class focusing on the _find_localzeros algorithm.

        The goal is to optimize the O(n²) algorithm that sequentially allocates values
        to localzeros by finding which values are more extreme than others.
        """

        @classmethod
        def _find_localzeros(cls, values, **options):
            """
            Sequentially allocate values to localzeros.

            When a value is identified as being more extreme than another member it
            replaces that member; if this is never true, then the value is simply
            appended to the localzeros.

            CURRENT COMPLEXITY: O(n²) - compares each value against all existing localzeros
            TARGET COMPLEXITY: O(n log n) or O(n) using transitivity
            """
            localzeros = set()
            # This is the O(n²) bottleneck that needs optimization
            # The algorithm compares every new value v against all existing localzeros
            # Optimization opportunity: Use transitivity to avoid redundant comparisons
            # If (x, y) and (y, z) have been compared, then (x, z) doesn't need testing

            for v in values:
                is_newzero = True
                localzeros_ = list(localzeros)  # O(n) conversion on each iteration
                for z in localzeros_:            # Nested loop = O(n²) total
                    if id(v) == id(z):
                        is_newzero = False
                    else:
                        con = cls._is_connected(v, z)
                        if con:
                            is_newzero = False
                            if con is True or con == cls:
                                localzeros.remove(z)
                                localzeros.update([v])
                if is_newzero:
                    localzeros.update([v])

            return localzeros
    ```

[Source repository](https://github.com/sympy/sympy){ .md-button }

---

Evaluator approaches available for every task: handwritten, llm_generated, llm_judge.
