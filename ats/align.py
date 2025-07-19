from sklearn.feature_extraction.text import TfidfVectorizer
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np


def compute_embeddings(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()


def fix_punc(text, segments, prepend, append, nopend):
    for l, s in enumerate(segments):
        if not s:
            continue
        t = text[l]
        for p, f in zip(s, s[1:] + [s[-1]]):
            connected = f[0] == p[1]
            loop = 0
            while True:
                if loop > 20:
                    break
                if p[1] < len(t) and t[p[1]] in append:
                    p[1] += 1
                elif t[p[1] - 1] in prepend:
                    p[1] -= 1
                elif (
                    (p[1] > 0 and t[p[1] - 1] in nopend)
                    or (p[1] < len(t) and t[p[1]] in nopend)
                    or (p[1] < len(t) - 1 and t[p[1] + 1] in nopend)
                ):
                    start, end = p[1] - 1, p[1]
                    if p[1] < len(t) - 1 and (
                        t[p[1] + 1] in nopend
                        and 0x4E00 > ord(t[p[1]])
                        or ord(t[p[1]]) > 0x9FAF
                    ):
                        end += 1
                    while start > 0 and t[start] in nopend:
                        start -= 1
                    while end < len(t) - 1 and t[end] in nopend:
                        end += 1

                    if t[start] in prepend:
                        p[1] = start
                    elif t[start] in append:
                        p[1] = start + 1
                    elif end < len(t) and t[end] in prepend:
                        p[1] = end
                    elif end < len(t) and t[end] in append:
                        p[1] = end + 1
                    else:
                        break
                else:
                    break
                loop += 1
            if connected:
                f[0] = p[1]


def align(model, lang, transcript, text, references, prepend, append, nopend):
    transcript_clean = [lang.clean(t) for t in transcript]
    text_clean = [lang.clean(t) for t in text]

    # Embedding computation
    transcript_embeddings = compute_embeddings(transcript_clean)
    text_embeddings = compute_embeddings(text_clean)

    # Apply DTW
    _, path = fastdtw(transcript_embeddings, text_embeddings, dist=euclidean)

    # Build segment alignment
    alignment_map = {}
    for t_idx, txt_idx in path:
        if txt_idx not in alignment_map:
            alignment_map[txt_idx] = []
        alignment_map[txt_idx].append(t_idx)

    segments = []
    for i, sentence in enumerate(text_clean):
        matched_idxs = alignment_map.get(i, [])
        segs = []
        if matched_idxs:
            combined = "".join([transcript_clean[j] for j in matched_idxs])
            start = 0
            end = len(combined)
            if end > 0:
                segs.append([start, end, matched_idxs[0]])
        segments.append(segs)

    # Apply punctuation heuristic
    fix_punc(text, segments, prepend, append, nopend)

    return segments, []  # references placeholder
