# Negotiation Dataset Visualizer

A small web app for exploring multi-issue negotiation corpora, as a first step
toward RLVR training. It currently bundles two datasets:

1. **Deal or No Deal** — FAIR
   [end-to-end-negotiator](https://github.com/facebookresearch/end-to-end-negotiator/tree/master/src/data/negotiate).
   Two agents split a pool of **books / hats / balls**. Counts are shared; each
   agent has private values summing to **10**. They chat, then each declares a
   split. If the splits are consistent (they partition every pool), each agent
   scores `sum(items_taken * value)`; otherwise it's **no deal**.

2. **CaSiNo (Campsite)** — Cornell
   [CaSiNo corpus](https://convokit.cornell.edu/documentation/casino-corpus.html).
   Two campers split **Food / Water / Firewood** (3 units each). Each privately
   ranks the issues High / Medium / Low priority, worth **5 / 4 / 3 points per
   unit** (max 36). They chat freely, then one submits a deal that the other
   accepts / rejects / walks away from. Includes rich metadata: per-issue
   argument reasons, per-utterance negotiation-strategy annotations, and
   self-reported satisfaction / opponent likeness.

Both expose an objective, verifiable score per agent — exactly what makes them
good RLVR targets.

## Layout

```
data/
  {train,val,test}.txt    raw Deal-or-No-Deal files (FAIR)
  casino-corpus/          ConvoKit CaSiNo corpus (utterances, conversations, ...)
parse_data.py             Deal-or-No-Deal .txt -> JSON  (public/data/dnd/)
parse_casino.py           CaSiNo ConvoKit -> JSON        (public/data/casino/)
build.py                  runs both parsers + writes public/data/manifest.json
public/
  index.html              app shell
  styles.css              styling
  app.js                  manifest-driven rendering + filtering + charts
  data/manifest.json      list of datasets, splits, items
  data/<dataset>/*.json   generated game files
serve.sh                  build (if needed) + start a static server
```

## Run

```bash
./serve.sh            # then open http://localhost:8791
# or pick a port:  ./serve.sh 9000
```

Deep-link a dataset with a hash, e.g. `http://localhost:8791/#casino`.

To regenerate all JSON after changing a parser:

```bash
python3 build.py
```

## Features

- Switch between **datasets** and their **splits** (DND: train/val/test · CaSiNo: all).
- Summary cards: game count, agreement rate, avg turns, avg / joint score, plus
  dataset-specific cards (DND efficient-deal rate · CaSiNo satisfaction & annotation coverage).
- Charts: dialogue-length histogram, per-agent score distribution (auto-scaled to
  the dataset's max), agreement donut.
- Per-game cards: item pool, both agents' private values (hover a CaSiNo value to
  read that camper's argument), the full dialogue as a chat transcript with
  strategy-annotation tags, and the final allocation with each side's score and
  (CaSiNo) subjective satisfaction / likeness.
- Search dialogue text; filter by outcome; sort by turns / joint score / lopsidedness.

## Adding another dataset

Write a parser that emits `public/data/<id>/<split>.json` with the shared game
schema (`counts`, `item_names`, `you_values`/`them_values`, `you_alloc`/`them_alloc`,
`you_score`/`them_score`, `you_max`/`them_max`, `agreed`, `valid_alloc`, `turns`,
optional `meta`) plus a `stats` block, then add an entry to `MANIFEST` in `build.py`.
