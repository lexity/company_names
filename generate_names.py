#!/usr/bin/env python

"""
"""

import random
import time
import sys
from collections import deque, defaultdict
from itertools import islice, izip


class NGrams(defaultdict):
    """
    Simple n-grams container.
    """
    def __init__(self, n=1, data=None):
        """
        "n" is the n-gram size (defaults to 1).
        "data" is a dict containing initial n-gram data.
        """
        super(type(self), self).__init__(lambda: defaultdict(int))
        if 1 > n:
            raise ValueError('"n" must be one or greater')
        self.n = n
        for k, v in (data or {}).iteritems():
            self[k] = defaultdict(int, **v)

    def feed(self, corpus):
        """
        Add corpus (an iterable of symbols) to the n-grams.
        Return the initial n-gram (seed).
        """
        corpus = iter(corpus)  # to prevent restarting of sequence
        ngram = deque(islice(corpus, self.n), maxlen=self.n)
        seed = tuple(ngram)
        ngrams_found = False
        for symbol in corpus:
            self[tuple(ngram)][symbol] += 1
            ngram.append(symbol)
            ngrams_found = True
        if not ngrams_found:
            raise ValueError('insufficient number of symbols in corpus')
        return seed

    def choice(self, ngram):
        """
        Return a random follower based on their relative probabilities.
        """
        if ngram not in self:
            raise KeyError(repr(ngram))

        t_weight = self.count(ngram)
        rand_weight = random.randrange(t_weight)
        cw = 0
        for s, c in self.followers(ngram):
            cw += c
            if cw > rand_weight:
                return s

    def followers(self, ngram):
        """Return an iterator over (symbol, count) tuples."""
        return self[ngram].iteritems()

    def count(self, ngram):
        """Return the number of times the n-gram occured."""
        return sum(c for s,c in self.followers(ngram))

    def __repr__(self):
        return 'NGrams(n={0}, data={1})'.format(self.n,
            dict((k, dict(v)) for k,v in self.iteritems()))


def markov_chain(ngrams, seed=None):
    """
    Generate a (possibly) infinite Markov chain of symbols.  Use
    itertools.islice(), or the like, to control length.

    An end can be reached in cases where a corpus ends in a unique
    sequence, which raises a StopIteration error.

    "ngrams" is an NGrams-like object;
    "seed" is an optional initial n-gram, defaults to None (random n-gram);
    """
    if seed is None:
        seed = random.choice(list(ngrams))
    elif seed not in ngrams:
        raise ValueError('"%s" is an invalid key'%repr(seed))

    ngram = deque(seed, maxlen=ngrams.n)

    # Yield the seed symbols
    for sym in ngram:
        yield sym

    # ... then follow the chain, forever!
    while True:
        try:
            symbol = ngrams.choice(tuple(ngram))
            ngram.append(symbol)
            yield symbol
        except KeyError as e:
            raise StopIteration('"%s" is a dead end'%repr(str(e)))


def sequence_exists(ngrams, seq):
    """
    Return True if "seq" can be generated via the n-grams in "ngrams",
    False otherwise.
    """
    n = ngrams.n
    lseq = len(seq)
    if lseq < n:
        return False
    if lseq == n:
        return tuple(seq) in ngrams

    for i in xrange(n, lseq):
        if tuple(seq[i-n:i]) not in ngrams:
            return False
    if seq[-1] not in (sym for sym,c in ngrams.followers(tuple(seq[i-n:i]))):
        return False
    return True



def train_words(ngrams, corpus):
    """
    Feed individual words from "corpus" to "ngrams" to build term
    frequencies.
    """
    seeds = list()

    min_len = ngrams.n + 1
    for word in (w.lower() for w in corpus if len(w) > min_len):
        seed = ngrams.feed(word)
        seeds.append(seed)

    return seeds


def wordify_text(*corpora):
    """
    Yield from set of words from all files in "corpora".
    """
    words = set()
    for corpus in corpora:
        contents = open(corpus, 'r').read()
        words.update(contents.split())

    return ('%s\n'%w for w in words)


def generate_random(ngrams, seeds=None):
    """
    Yield random words.  "seeds" is a collection of initial n-grams.
    """
    while True:
        word = list()
        for c in markov_chain(ngrams,
                seed=(None if not seeds else random.choice(seeds))):
            if c == '\n':
                yield ''.join(word)
                break
            word.append(c)


def valid_length(words, min_len=1, max_len=9):
    """
    Yield words from iterable "words" that meet all criteria.
    "min_len"/"max_len" are word length criteria;
    """
    for word in words:
        if len(word) < min_len or len(word) > max_len:
            continue
        yield word


def skip_known(words, known):
    """
    Yield words from "words" that don't appear in "known".
    "known" is a callable (takes one "word" argument) that returns
    True if word exists in some lowercased collection.
    """
    for word in words:
        if known(word.lower()):
            continue
        yield word


def output_streams(streams, colw=9, delay=0):
    # make custom format string for a row
    rowfmt = '  '.join('{{{0}:<{1}}}'.format(i,colw) for i in xrange(len(streams)))

    # and print!
    for words in izip(*streams):
        print rowfmt.format(*words)
        time.sleep(delay)


def usage():
    print
    print 'Usage: {0} <corpus> [<corpus2> ...]'.format(sys.argv[0])
    print '\t"corpus" is a text file use for sampling'


if __name__ == '__main__':
    COL_WIDTH = 12

    if 2 > len(sys.argv):
        usage()
        sys.exit(0)

    corpora = sys.argv[1:]


    # To ignore generating dictionary words
    try:
        known = set(w.rstrip().lower() for w in wordify_text(*corpora))
    except IOError as e:
        print str(e)
        usage()
        sys.exit(1)

    def configure_ngrams(ngrams):

        seeds = train_words(ngrams, wordify_text(*corpora))

        words = generate_random(ngrams, seeds=seeds)
        words = valid_length(words, min_len=ngrams.n+1, max_len=COL_WIDTH)
        words = skip_known(words, known.__contains__)

        return words

    """
    ngrams = NGrams(3)
    words = configure_ngrams(ngrams)
    """
    ngrams_2 = NGrams(2)
    words_2 = configure_ngrams(ngrams_2)
    ngrams_3 = NGrams(3)
    words_3 = configure_ngrams(ngrams_3)
    ngrams_4 = NGrams(4)
    words_4 = configure_ngrams(ngrams_4)

    output_streams((words_2, words_3, words_4), colw=COL_WIDTH, delay=3)
    #output_streams((words_3,)*3, colw=COL_WIDTH, delay=3)




