#!/usr/bin/env python3
"""
Initial program for OpenEvolve: difflib Differ._fancy_replace optimization

This file contains the Differ class with the _fancy_replace method that searches
for similar lines when replacing one block with another.

Optimization target: performance-fancy-replace
Current performance: Baseline score ~0.365, with RECURSION ERRORS at 500+ lines
Goal: Achieve score >0.8 through optimizing the WINDOW-based search algorithm

Key issues:
- O(N*WINDOW) complexity with expensive SequenceMatcher ratio computations
- Recursion errors still occur on pathological cases (500+ similar lines)
- Lines 929-946 in original: nested loops with three ratio checks per iteration
"""

from collections import namedtuple as _namedtuple
from types import GenericAlias

Match = _namedtuple('Match', 'a b size')

def _calculate_ratio(matches, length):
    if length:
        return 2.0 * matches / length
    return 1.0


class SequenceMatcher:
    """
    Simplified SequenceMatcher for standalone testing.
    Only includes methods needed by Differ._fancy_replace.
    """

    def __init__(self, isjunk=None, a='', b='', autojunk=True):
        self.isjunk = isjunk
        self.a = self.b = None
        self.autojunk = autojunk
        self.set_seqs(a, b)

    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        if a is self.a:
            return
        self.a = a
        self.matching_blocks = self.opcodes = None

    def set_seq2(self, b):
        if b is self.b:
            return
        self.b = b
        self.matching_blocks = self.opcodes = None
        self.fullbcount = None
        self.__chain_b()

    def __chain_b(self):
        b = self.b
        self.b2j = b2j = {}
        for i, elt in enumerate(b):
            indices = b2j.setdefault(elt, [])
            indices.append(i)
        self.bjunk = junk = set()
        isjunk = self.isjunk
        if isjunk:
            for elt in b2j.keys():
                if isjunk(elt):
                    junk.add(elt)
            for elt in junk:
                del b2j[elt]
        self.bpopular = popular = set()
        n = len(b)
        if self.autojunk and n >= 200:
            ntest = n // 100 + 1
            for elt, idxs in b2j.items():
                if len(idxs) > ntest:
                    popular.add(elt)
            for elt in popular:
                del b2j[elt]

    def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None):
        a, b, b2j, isbjunk = self.a, self.b, self.b2j, self.bjunk.__contains__
        if ahi is None:
            ahi = len(a)
        if bhi is None:
            bhi = len(b)
        besti, bestj, bestsize = alo, blo, 0
        j2len = {}
        nothing = []
        for i in range(alo, ahi):
            j2lenget = j2len.get
            newj2len = {}
            for j in b2j.get(a[i], nothing):
                if j < blo:
                    continue
                if j >= bhi:
                    break
                k = newj2len[j] = j2lenget(j-1, 0) + 1
                if k > bestsize:
                    besti, bestj, bestsize = i-k+1, j-k+1, k
            j2len = newj2len
        while besti > alo and bestj > blo and \
              not isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              not isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize += 1
        while besti > alo and bestj > blo and \
              isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize = bestsize + 1
        return Match(besti, bestj, bestsize)

    def get_matching_blocks(self):
        if self.matching_blocks is not None:
            return self.matching_blocks
        la, lb = len(self.a), len(self.b)
        queue = [(0, la, 0, lb)]
        matching_blocks = []
        while queue:
            alo, ahi, blo, bhi = queue.pop()
            i, j, k = x = self.find_longest_match(alo, ahi, blo, bhi)
            if k:
                matching_blocks.append(x)
                if alo < i and blo < j:
                    queue.append((alo, i, blo, j))
                if i+k < ahi and j+k < bhi:
                    queue.append((i+k, ahi, j+k, bhi))
        matching_blocks.sort()
        i1 = j1 = k1 = 0
        non_adjacent = []
        for i2, j2, k2 in matching_blocks:
            if i1 + k1 == i2 and j1 + k1 == j2:
                k1 += k2
            else:
                if k1:
                    non_adjacent.append((i1, j1, k1))
                i1, j1, k1 = i2, j2, k2
        if k1:
            non_adjacent.append((i1, j1, k1))
        non_adjacent.append( (la, lb, 0) )
        self.matching_blocks = list(map(Match._make, non_adjacent))
        return self.matching_blocks

    def get_opcodes(self):
        if self.opcodes is not None:
            return self.opcodes
        i = j = 0
        self.opcodes = answer = []
        for ai, bj, size in self.get_matching_blocks():
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'
            if tag:
                answer.append( (tag, i, ai, j, bj) )
            i, j = ai+size, bj+size
            if size:
                answer.append( ('equal', ai, i, bj, j) )
        return answer

    def ratio(self):
        matches = sum(triple[-1] for triple in self.get_matching_blocks())
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def quick_ratio(self):
        if self.fullbcount is None:
            self.fullbcount = fullbcount = {}
            for elt in self.b:
                fullbcount[elt] = fullbcount.get(elt, 0) + 1
        fullbcount = self.fullbcount
        avail = {}
        availhas, matches = avail.__contains__, 0
        for elt in self.a:
            if availhas(elt):
                numb = avail[elt]
            else:
                numb = fullbcount.get(elt, 0)
            avail[elt] = numb - 1
            if numb > 0:
                matches = matches + 1
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def real_quick_ratio(self):
        la, lb = len(self.a), len(self.b)
        return _calculate_ratio(min(la, lb), la + lb)


def _keep_original_ws(s, tag_s):
    """Replace whitespace with the original whitespace characters in `s`"""
    return ''.join(
        c if tag_c == " " and c.isspace() else tag_c
        for c, tag_c in zip(s, tag_s)
    )


class Differ:
    """
    Differ is a class for comparing sequences of lines of text.
    This standalone version includes only the methods needed for testing _fancy_replace.
    """

    def __init__(self, linejunk=None, charjunk=None):
        self.linejunk = linejunk
        self.charjunk = charjunk

    def compare(self, a, b):
        cruncher = SequenceMatcher(self.linejunk, a, b)
        for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
            if tag == 'replace':
                g = self._fancy_replace(a, alo, ahi, b, blo, bhi)
            elif tag == 'delete':
                g = self._dump('-', a, alo, ahi)
            elif tag == 'insert':
                g = self._dump('+', b, blo, bhi)
            elif tag == 'equal':
                g = self._dump(' ', a, alo, ahi)
            else:
                raise ValueError('unknown tag %r' % (tag,))
            yield from g

    def _dump(self, tag, x, lo, hi):
        """Generate comparison results for a same-tagged range."""
        for i in range(lo, hi):
            yield '%s %s' % (tag, x[i])

    def _plain_replace(self, a, alo, ahi, b, blo, bhi):
        assert alo < ahi and blo < bhi
        if bhi - blo < ahi - alo:
            first  = self._dump('+', b, blo, bhi)
            second = self._dump('-', a, alo, ahi)
        else:
            first  = self._dump('-', a, alo, ahi)
            second = self._dump('+', b, blo, bhi)
        for g in first, second:
            yield from g

    def _fancy_replace(self, a, alo, ahi, b, blo, bhi):
        r"""
        When replacing one block of lines with another, search the blocks
        for *similar* lines; the best-matching pair (if any) is used as a
        synch point, and intraline difference marking is done on the
        similar pair. Lots of work, but often worth it.
        """
        # EVOLVE-BLOCK-START id="performance-fancy-replace"
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
        # EVOLVE-BLOCK-END

    def _fancy_helper(self, a, alo, ahi, b, blo, bhi):
        g = []
        if alo < ahi:
            if blo < bhi:
                g = self._plain_replace(a, alo, ahi, b, blo, bhi)
            else:
                g = self._dump('-', a, alo, ahi)
        elif blo < bhi:
            g = self._dump('+', b, blo, bhi)
        yield from g

    def _qformat(self, aline, bline, atags, btags):
        atags = _keep_original_ws(aline, atags).rstrip()
        btags = _keep_original_ws(bline, btags).rstrip()
        yield "- " + aline
        if atags:
            yield f"? {atags}\n"
        yield "+ " + bline
        if btags:
            yield f"? {btags}\n"


if __name__ == "__main__":
    # Quick test to ensure the standalone version works
    print("Testing Differ._fancy_replace standalone program...")

    d = Differ()
    a = ["0123456789\n"] * 10
    b = ["01234a56789\n"] * 10

    result = list(d.compare(a, b))
    print(f"✅ Test passed: Generated {len(result)} output lines")
    print(f"   First 5 lines: {result[:5]}")
